"""Model wrapper that exposes the residual stream and a differentiable
'rest-of-network' function, so we can compute Jacobian-lens readouts.

The heavy lifting for the lens itself lives in ``lens.py``; this module is only
responsible for (a) loading a HF causal-LM onto the GPU, (b) running a forward
pass whose intermediate residual-stream tensors are retained in the autograd
graph, and (c) exposing the final RMSNorm + unembedding so a readout can be
projected into vocabulary space.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Where the user's models already live.
HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")


# --------------------------------------------------------------- device utils
# NOTE: ROCm/HIP masquerades as `torch.cuda`, so an AMD GPU (e.g. Strix Halo
# gfx1151) is driven through exactly the same API as an NVIDIA one.

def pick_device(prefer: str | None = None) -> str:
    if prefer and prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def backend_name() -> str:
    if torch.cuda.is_available():
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_label() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mem_allocated_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def mem_total_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 8.0  # conservative guess for CPU boxes


def default_dtype(device: str) -> torch.dtype:
    # bf16 on accelerators; fp32 on CPU (bf16 CPU matmul is slow/patchy)
    return torch.float32 if device == "cpu" else torch.bfloat16


def move_to_device_fast(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Move a CPU model onto `device`, cloning each tensor into anonymous RAM first.

    `from_pretrained` hands back parameters that alias the mmap'd safetensors
    file. On ROCm a host->device copy straight from those file-backed pages runs
    at ~0.02 GB/s, versus ~15 GB/s once the tensor is cloned into ordinary
    memory -- a 1.7B model takes 388s vs <2s. (On CUDA the mmap path is fine;
    the extra CPU memcpy is cheap, so we take the same route everywhere.)

    Parameters are memoized on object identity so tied weights (e.g. Qwen3's
    lm_head <- embed_tokens) remain the *same* Parameter after the move.
    """
    if device == "cpu":
        return model

    seen_params: dict[int, torch.nn.Parameter] = {}
    seen_buffers: dict[int, torch.Tensor] = {}

    for mod in model.modules():
        for name, p in list(mod._parameters.items()):
            if p is None:
                continue
            new = seen_params.get(id(p))
            if new is None:
                new = torch.nn.Parameter(p.data.detach().clone().to(device),
                                         requires_grad=p.requires_grad)
                seen_params[id(p)] = new
            mod._parameters[name] = new
        for name, b in list(mod._buffers.items()):
            if b is None:
                continue
            new_b = seen_buffers.get(id(b))
            if new_b is None:
                new_b = b.detach().clone().to(device)
                seen_buffers[id(b)] = new_b
            mod._buffers[name] = new_b

    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


def resolve_local_model(model_id: str) -> str:
    """Turn 'Qwen/Qwen3-1.7B-Base' into the local snapshot dir if it is cached,
    otherwise return the id unchanged (letting HF download / resolve it)."""
    if os.path.isdir(model_id):
        return model_id
    cache_name = "models--" + model_id.replace("/", "--")
    snap_glob = os.path.join(HF_HUB, cache_name, "snapshots", "*")
    snaps = sorted(glob.glob(snap_glob))
    for snap in snaps:
        if os.path.exists(os.path.join(snap, "config.json")):
            return snap
    return model_id


@dataclass
class ForwardCache:
    """Tensors captured from a single forward pass, kept in the autograd graph."""

    input_ids: torch.Tensor                # [B, T]
    inputs_embeds: torch.Tensor            # [B, T, d]  (leaf, requires_grad)
    residuals: list[torch.Tensor]          # residuals[i] = output of layer i (pre-norm)
    final_resid: torch.Tensor              # input to the final RMSNorm (pre-norm)
    logits: torch.Tensor                   # [B, T, V]  model's own next-token logits
    tokens: list[str] = field(default_factory=list)


class LensModel:
    def __init__(self, model_id: str, device: str | None = None,
                 dtype: torch.dtype | None = None):
        self.model_id = model_id
        device = pick_device(device)
        dtype = dtype or default_dtype(device)
        local = resolve_local_model(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(local)
        # 'eager' attention is plain matmul+softmax, which (unlike SDPA/flash)
        # supports the double-backward we use to form Jacobian-vector products.
        model = AutoModelForCausalLM.from_pretrained(
            local, dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.model = move_to_device_fast(model, device).eval()
        # We never need gradients w.r.t. weights; only w.r.t. activations.
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.dtype = dtype

        # Locate the standard submodules. Works for Qwen3 / Llama-style models.
        self.inner = self.model.model
        self.layers = self.inner.layers
        self.n_layers = len(self.layers)
        self.norm = self.inner.norm
        self.embed = self.inner.get_input_embeddings() if hasattr(self.inner, "get_input_embeddings") else self.inner.embed_tokens
        self.lm_head = self.model.get_output_embeddings()

        cfg = self.model.config
        self.d_model = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size
        self.vocab_size = self.lm_head.weight.shape[0]

    # ------------------------------------------------------------------ utils
    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def token_strings(self, input_ids: torch.Tensor) -> list[str]:
        ids = input_ids[0].tolist()
        return [self.tokenizer.decode([i]) for i in ids]

    def decode_top(self, logits_row: torch.Tensor, k: int = 1):
        """logits_row: [V] -> list of (token_str, prob)."""
        probs = torch.softmax(logits_row.float(), dim=-1)
        top = torch.topk(probs, k)
        out = []
        for p, idx in zip(top.values.tolist(), top.indices.tolist()):
            out.append((self.tokenizer.decode([idx]), float(p), int(idx)))
        return out

    # --------------------------------------------------------------- forward
    def forward_with_graph(self, input_ids: torch.Tensor, batch: int = 1,
                           grad: bool = True) -> ForwardCache:
        """Run a forward pass that keeps every layer's residual output in the
        autograd graph. ``batch`` replicates the sequence so we can inject a
        different tangent per position in one shot.

        Uses forward hooks (version-robust) to grab: (1) each decoder layer's
        output residual, (2) the input to the final norm (the pre-norm final
        residual we differentiate through).

        ``grad=False`` skips graph construction entirely -- the logit lens only
        reads the residuals, so it has no reason to pay for a graph.
        """
        if batch > 1:
            input_ids = input_ids.expand(batch, -1).contiguous()

        inputs_embeds = self.embed(input_ids).detach().clone()
        if grad:
            inputs_embeds.requires_grad_(True)

        residuals: list[Optional[torch.Tensor]] = [None] * self.n_layers
        norm_input: dict[str, torch.Tensor] = {}

        handles = []

        def make_layer_hook(idx):
            def hook(_module, _inp, out):
                residuals[idx] = out[0] if isinstance(out, tuple) else out
            return hook

        def norm_pre_hook(_module, args):
            norm_input["x"] = args[0]

        for i, layer in enumerate(self.layers):
            handles.append(layer.register_forward_hook(make_layer_hook(i)))
        handles.append(self.norm.register_forward_pre_hook(norm_pre_hook))

        try:
            with (torch.enable_grad() if grad else torch.no_grad()):
                out = self.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
        finally:
            for h in handles:
                h.remove()

        return ForwardCache(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            residuals=[r for r in residuals],  # type: ignore
            final_resid=norm_input["x"],
            logits=out.logits,
        )

    def project(self, resid_vec: torch.Tensor) -> torch.Tensor:
        """Apply final RMSNorm + unembedding: (..., d) -> (..., V) logits."""
        return self.lm_head(self.norm(resid_vec))
