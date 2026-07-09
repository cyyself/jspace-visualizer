"""Jacobian-lens ("J-lens") and logit-lens readouts over the residual stream.

Logit lens
----------
    readout(t, l) = softmax( W_U . norm( h_l[t] ) )
The classic lens: unembed the raw residual at each (position, layer) cell.

Jacobian lens
-------------
    readout(t, l) = softmax( W_U . norm( J_l . h_l[t] ) )
where J_l = d(final_resid) / d(h_l) is the Jacobian mapping the layer-l
residual to the final-layer residual basis, and we use the *future-summed*
influence  sum_{t' >= t} d(final_resid[t']) / d(h_l[t]).  Intuitively J_l . h_l[t]
answers: "if this internal vector were nudged, which tokens would it push the
model toward saying downstream?"  This follows the J-lens formulation from
Anthropic's global-workspace work; here we compute a *context-specific*
Jacobian (exact for the current prompt) instead of a corpus-averaged one, which
removes the offline fitting step.

The J-v product is computed with the standard forward-over-reverse trick:
    g(u) = d<final_resid, u>/d h_l = J_l^T u        (first backward, create_graph)
    J_l . v = d<g, v>/du                            (second backward)
which needs no forward-mode AD and works with plain eager attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import ForwardCache, LensModel, empty_cache


@dataclass
class Cell:
    layer: int
    pos: int
    token: str
    prob: float
    token_id: int


def _grid_payload(lm: LensModel, tokens, layers, cells, kind):
    return {
        "kind": kind,
        "tokens": tokens,
        "layers": layers,           # layer indices, top row = deepest
        "n_layers": lm.n_layers,
        "cells": cells,             # cells[row][col] -> {token, prob, id}
    }


class Lens:
    def __init__(self, lm: LensModel):
        self.lm = lm

    # ------------------------------------------------------------ logit lens
    @torch.no_grad()
    def logit_lens_grid(self, input_ids: torch.Tensor, layers=None):
        lm = self.lm
        tokens = lm.token_strings(input_ids)
        cache = lm.forward_with_graph(input_ids, batch=1)
        if layers is None:
            layers = list(range(lm.n_layers))
        # top row = deepest layer
        rows = list(reversed(layers))
        cells = []
        for li in rows:
            resid = cache.residuals[li][0]              # [T, d]
            logits = lm.project(resid)                  # [T, V]
            row = []
            for t in range(resid.shape[0]):
                tok, prob, tid = lm.decode_top(logits[t], k=1)[0]
                row.append({"token": tok, "prob": prob, "id": tid})
            cells.append(row)
        return _grid_payload(lm, tokens, rows, cells, "logit")

    # --------------------------------------------------------- jacobian lens
    def jlens_grid(self, input_ids: torch.Tensor, layers=None, pos_chunk: int = 24):
        """Full position x layer J-lens grid for a static prompt."""
        lm = self.lm
        tokens = lm.token_strings(input_ids)
        T = input_ids.shape[1]
        if layers is None:
            layers = list(range(lm.n_layers))
        rows = list(reversed(layers))
        positions = list(range(T))

        # readout[layer_idx][pos] = (token, prob, id)
        readout = {li: [None] * T for li in rows}

        for start in range(0, T, pos_chunk):
            chunk = positions[start:start + pos_chunk]
            self._jlens_chunk(input_ids, rows, chunk, readout)

        cells = []
        for li in rows:
            row = [{"token": r[0], "prob": r[1], "id": r[2]} for r in readout[li]]
            cells.append(row)
        return _grid_payload(lm, tokens, rows, cells, "jacobian")

    def _jlens_chunk(self, input_ids, layers, positions, readout):
        """Compute J-lens readouts for a chunk of positions across all layers.

        A single batched forward (one sequence per position) lets us inject a
        distinct tangent per position; the forward graph is reused across layers.
        """
        lm = self.lm
        B = len(positions)
        cache = lm.forward_with_graph(input_ids, batch=B)
        y = cache.final_resid                            # [B, T, d]

        pos_idx = torch.tensor(positions, device=lm.device)
        batch_idx = torch.arange(B, device=lm.device)

        for li in layers:
            s = cache.residuals[li]                       # [B, T, d]
            # tangent: for batch row b, nonzero only at its own position.
            tangent = torch.zeros_like(s)
            tangent[batch_idx, pos_idx, :] = s[batch_idx, pos_idx, :].detach()

            u = torch.zeros_like(y, requires_grad=True)
            (g,) = torch.autograd.grad(y, s, grad_outputs=u,
                                       create_graph=True, retain_graph=True)
            (jvp,) = torch.autograd.grad(g, u, grad_outputs=tangent,
                                         retain_graph=True)
            # future-summed influence: sum over output positions.
            r = jvp.sum(dim=1)                            # [B, d]
            logits = lm.project(r)                        # [B, V]
            for bi, p in enumerate(positions):
                tok, prob, tid = lm.decode_top(logits[bi], k=1)[0]
                readout[li][p] = (tok, prob, tid)
            del u, g, jvp, r, logits, tangent
        del cache
        empty_cache()

    # ---------------------------------------- shared single-position readout
    def _readout_logits_at(self, cache: ForwardCache, pos: int, rows, kind: str):
        """Return {layer_index: logits[V]} for a single position `pos`."""
        lm = self.lm
        out = {}
        if kind == "logit":
            for li in rows:
                out[li] = lm.project(cache.residuals[li][0, pos])
            return out
        # jacobian: future-summed J_l . h_l[pos], one double-backward per layer
        y = cache.final_resid
        for li in rows:
            s = cache.residuals[li]
            tangent = torch.zeros_like(s)
            tangent[0, pos, :] = s[0, pos, :].detach()
            u = torch.zeros_like(y, requires_grad=True)
            (g,) = torch.autograd.grad(y, s, grad_outputs=u,
                                       create_graph=True, retain_graph=True)
            (jvp,) = torch.autograd.grad(g, u, grad_outputs=tangent,
                                         retain_graph=True)
            r = jvp.sum(dim=1)                            # [1, d]
            out[li] = lm.project(r)[0]
            del u, g, jvp, r, tangent
        return out

    def band_at(self, cache: ForwardCache, pos: int, kind: str, topk: int = 3, layers=None):
        """Workspace band: top-k readout at `pos` for every layer (deepest first)."""
        lm = self.lm
        rows = list(reversed(layers if layers is not None else range(lm.n_layers)))
        logits_map = self._readout_logits_at(cache, pos, rows, kind)
        band = []
        for li in rows:
            tops = lm.decode_top(logits_map[li], k=topk)
            band.append({
                "layer": li,
                "top": [{"token": tk, "prob": pr, "id": tid} for tk, pr, tid in tops],
            })
        return band

    # ------------------------------------------------- streaming generation
    def generate_stream(self, input_ids, kind="jacobian", max_new_tokens=24,
                        temperature=0.0, topk_band=3, layers=None):
        """Yield one event per generated token: the chosen token plus the full
        workspace band (per-layer readout at the generating position)."""
        lm = self.lm
        ids = input_ids
        eos = lm.tokenizer.eos_token_id
        for step in range(max_new_tokens):
            cache = lm.forward_with_graph(ids, batch=1)
            pos = ids.shape[1] - 1
            band = self.band_at(cache, pos, kind, topk_band, layers)
            logits = cache.logits[0, -1].float()
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            else:
                nxt = int(logits.argmax())
            tok_str = lm.tokenizer.decode([nxt])
            next_top = [{"token": t, "prob": p} for t, p, _ in lm.decode_top(logits, 5)]
            yield {"step": step, "token": tok_str, "token_id": nxt,
                   "band": band, "next_top": next_top}
            ids = torch.cat([ids, torch.tensor([[nxt]], device=lm.device)], dim=1)
            del cache
            empty_cache()
            if eos is not None and nxt == eos:
                break

    # -------------------------------------------------------- interventions
    def steering_vector(self, input_ids, layer: int, token_id: int, pos: int = -1):
        """Direction in layer-`layer` residual space that most increases the
        logit of `token_id` at `pos`: v = d logit_token / d h_layer[pos]."""
        lm = self.lm
        cache = lm.forward_with_graph(input_ids, batch=1)
        s = cache.residuals[layer]
        if pos < 0:
            pos = cache.final_resid.shape[1] + pos
        logit = lm.project(cache.final_resid[0, pos])[token_id]
        (grad,) = torch.autograd.grad(logit, s, retain_graph=False)
        v = grad[0, pos].detach()
        return v / (v.norm() + 1e-6)

    @torch.no_grad()
    def generate_plain(self, input_ids, max_new_tokens=24, temperature=0.0,
                       steer=None):
        """Greedy/temperature generation, optionally with a steering hook.
        `steer` = (layer, unit_vector, alpha). Returns decoded continuation."""
        lm = self.lm
        ids = input_ids.clone()
        handle = None
        if steer is not None:
            layer_i, vec, alpha = steer
            vec = vec.to(lm.dtype)

            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                scale = h.norm(dim=-1, keepdim=True).mean()
                h = h + alpha * scale * vec
                if isinstance(out, tuple):
                    return (h,) + tuple(out[1:])
                return h
            handle = lm.layers[layer_i].register_forward_hook(hook)
        try:
            out_ids = []
            for _ in range(max_new_tokens):
                logits = lm.model(input_ids=ids, use_cache=False).logits[0, -1].float()
                if temperature and temperature > 0:
                    nxt = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))
                else:
                    nxt = int(logits.argmax())
                out_ids.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]], device=lm.device)], dim=1)
                if lm.tokenizer.eos_token_id is not None and nxt == lm.tokenizer.eos_token_id:
                    break
        finally:
            if handle is not None:
                handle.remove()
        return lm.tokenizer.decode(out_ids)
