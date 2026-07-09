"""FastAPI server for the J-space visualizer.

Endpoints
  GET  /                     -> the web UI
  GET  /api/models           -> available local models + current
  POST /api/load             -> load/switch model {model}
  POST /api/slice            -> position x layer grid {prompt, kind, max_layers}
  POST /api/generate         -> SSE stream: per-token workspace band
  POST /api/intervene        -> steer generation toward a token at a layer
"""

from __future__ import annotations

import glob
import json
import os
import threading

import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .model import HF_HUB, LensModel
from .lens import Lens

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(HERE, "web")

# Models we know are plain (differentiable) causal LMs with local weights.
PREFERRED = [
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3.5-2B-Base",
]

app = FastAPI(title="J-space visualizer")

_state = {"lm": None, "lens": None, "model_id": None}
_lock = threading.Lock()


def _is_text_causal_lm(snap_dir: str, name: str) -> bool:
    cfgs = glob.glob(os.path.join(snap_dir, "config.json"))
    if not cfgs:
        return False
    try:
        cfg = json.load(open(cfgs[0]))
    except Exception:
        return False
    archs = cfg.get("architectures") or []
    mtype = cfg.get("model_type", "")
    # drop vision-language, seq2seq translation, etc.
    if any(x in mtype.lower() for x in ("vl", "marian", "whisper", "clip")):
        return False
    if any("VL" in a or "Vision" in a for a in archs):
        return False
    # allow anything exposing a causal-LM head, plus our known-good preferred set
    if name in PREFERRED:
        return True
    return any(a.endswith("ForCausalLM") for a in archs)


def available_models():
    found = []
    for d in sorted(glob.glob(os.path.join(HF_HUB, "models--*"))):
        name = os.path.basename(d).replace("models--", "").replace("--", "/", 1)
        if "GGUF" in name:
            continue
        snaps = glob.glob(os.path.join(d, "snapshots", "*"))
        snap = next((s for s in snaps if glob.glob(os.path.join(s, "*.safetensors"))), None)
        if snap and _is_text_causal_lm(snap, name):
            found.append(name)
    ordered = [m for m in PREFERRED if m in found] + [m for m in found if m not in PREFERRED]
    return ordered


def get_lens(model_id: str | None = None) -> Lens:
    with _lock:
        target = model_id or _state["model_id"] or PREFERRED[0]
        if _state["lens"] is None or target != _state["model_id"]:
            if _state["lm"] is not None:
                del _state["lm"], _state["lens"]
                _state["lm"] = _state["lens"] = None
                torch.cuda.empty_cache()
            lm = LensModel(target)
            _state.update(lm=lm, lens=Lens(lm), model_id=target)
        return _state["lens"]


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(WEB, "index.html")) as f:
        return f.read()


@app.get("/api/models")
def models():
    return {"models": available_models(), "current": _state["model_id"]}


@app.post("/api/load")
async def load(req: Request):
    body = await req.json()
    lens = get_lens(body.get("model"))
    lm = lens.lm
    return {"model": _state["model_id"], "n_layers": lm.n_layers,
            "d_model": lm.d_model, "vocab": lm.vocab_size,
            "gpu_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}


def _subset_layers(n_layers: int, max_layers: int | None):
    if not max_layers or max_layers >= n_layers:
        return list(range(n_layers))
    # evenly spaced subset, always including the last layer
    idx = torch.linspace(0, n_layers - 1, max_layers).round().long().tolist()
    return sorted(set(int(i) for i in idx))


@app.post("/api/slice")
async def slice_grid(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "The capital of France is")
    kind = body.get("kind", "jacobian")
    lens = get_lens(body.get("model"))
    lm = lens.lm
    ids = lm.tokenize(prompt)
    layers = _subset_layers(lm.n_layers, body.get("max_layers"))
    if kind == "logit":
        grid = lens.logit_lens_grid(ids, layers=layers)
    else:
        grid = lens.jlens_grid(ids, layers=layers)
    return JSONResponse(grid)


@app.post("/api/generate")
async def generate(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "The capital of France is")
    kind = body.get("kind", "jacobian")
    max_new = int(body.get("max_new_tokens", 24))
    temperature = float(body.get("temperature", 0.0))
    lens = get_lens(body.get("model"))
    lm = lens.lm
    ids = lm.tokenize(prompt)
    layers = _subset_layers(lm.n_layers, body.get("max_layers"))

    def event_stream():
        meta = {"type": "meta", "tokens": lm.token_strings(ids),
                "n_layers": lm.n_layers, "layers": list(reversed(layers)),
                "model": _state["model_id"]}
        yield f"data: {json.dumps(meta)}\n\n"
        try:
            for ev in lens.generate_stream(ids, kind=kind, max_new_tokens=max_new,
                                           temperature=temperature, layers=layers):
                ev["type"] = "token"
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:  # surface errors to the client instead of hanging
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/intervene")
async def intervene(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "The capital of France is")
    layer = int(body.get("layer", 0))
    token = body.get("token", " Paris")
    alpha = float(body.get("alpha", 8.0))
    max_new = int(body.get("max_new_tokens", 24))
    lens = get_lens(body.get("model"))
    lm = lens.lm
    ids = lm.tokenize(prompt)
    tok_ids = lm.tokenizer(token, add_special_tokens=False).input_ids
    if not tok_ids:
        return JSONResponse({"error": "token did not tokenize"}, status_code=400)
    token_id = tok_ids[0]
    vec = lens.steering_vector(ids, layer, token_id)
    baseline = lens.generate_plain(ids, max_new_tokens=max_new)
    steered = lens.generate_plain(ids, max_new_tokens=max_new,
                                  steer=(layer, vec, alpha))
    return {"prompt": prompt, "layer": layer, "token": token, "alpha": alpha,
            "baseline": baseline, "steered": steered}
