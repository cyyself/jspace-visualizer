"""Validate the core mechanism end-to-end on a small local model."""
import sys, time
import torch
sys.path.insert(0, "/home/cyy/jspace-visualizer")
from jspace.model import LensModel, backend_name, device_label, mem_allocated_gb
from jspace.lens import Lens

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B-Base"
PROMPT = "The capital of France is"

print(f"backend={backend_name()} device={device_label()} torch={torch.__version__} hip={torch.version.hip}", flush=True)
print(f"loading {MODEL} ...", flush=True)
t0 = time.time()
lm = LensModel(MODEL)
print(f"  loaded in {time.time()-t0:.1f}s | layers={lm.n_layers} d={lm.d_model} V={lm.vocab_size}", flush=True)
print(f"  gpu mem: {mem_allocated_gb():.1f} GB", flush=True)

ids = lm.tokenize(PROMPT)
print("tokens:", lm.token_strings(ids), flush=True)

# 1) Sanity: our captured final_resid -> project must equal the model's own logits.
cache = lm.forward_with_graph(ids, batch=1)
manual_logits = lm.project(cache.final_resid)
diff = (manual_logits - cache.logits).abs().max().item()
print(f"[check] project(final_resid) vs model logits max|diff| = {diff:.4e}", flush=True)
nxt = lm.decode_top(cache.logits[0, -1], k=5)
print("  model next-token top5:", [(t, round(p, 3)) for t, p, _ in nxt], flush=True)

lens = Lens(lm)

# 2) Logit lens grid.
t0 = time.time()
lg = lens.logit_lens_grid(ids)
print(f"[logit-lens] grid {len(lg['cells'])}x{len(lg['tokens'])} in {time.time()-t0:.2f}s", flush=True)
print("  last-column readout (deepest->shallow):",
      [row[-1]["token"] for row in lg["cells"]][:8], flush=True)

# 3) J-lens grid.
t0 = time.time()
jg = lens.jlens_grid(ids)
dt = time.time() - t0
print(f"[j-lens] grid {len(jg['cells'])}x{len(jg['tokens'])} in {dt:.2f}s", flush=True)
print("  last-column readout (deepest->shallow):",
      [row[-1]["token"] for row in jg["cells"]][:8], flush=True)

# 4) Bottom row (deepest layer) must match model prediction (J=identity).
bottom = jg["cells"][0][-1]["token"]
model_top = nxt[0][0]
print(f"[check] deepest J-lens last-col = {bottom!r}  vs model top = {model_top!r}", flush=True)
peak = torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0.0
print(f"  peak gpu mem: {peak:.1f} GB", flush=True)
print("OK", flush=True)
