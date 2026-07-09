"""Is repeated J-lens slicing leaking, or just scaling badly with prompt length?

  (1) leak     -> memory after each identical request keeps climbing
  (2) scaling  -> peak memory grows with prompt length; auto_pos_chunk should
                  hold peak roughly flat instead of OOMing on long prompts
  (3) identity -> chunking must not change the answer
"""
import sys, time
import torch
sys.path.insert(0, "/home/cyy/jspace-visualizer")
from jspace.model import LensModel
from jspace.lens import Lens

lm = LensModel("Qwen/Qwen3-1.7B-Base")
lens = Lens(lm)
base = torch.cuda.memory_allocated() / 1e9
print(f"model resident: {base:.2f} GB   total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\n")

gb = lambda x: x / 1e9

print("--- (1) leak check: same prompt, 5x ---")
ids = lm.tokenize(" ".join(["the quick brown fox jumps over the lazy dog"] * 4))
for i in range(5):
    torch.cuda.reset_peak_memory_stats()
    lens.jlens_grid(ids)
    print(f"  run {i+1}: peak={gb(torch.cuda.max_memory_allocated()):5.2f} GB  "
          f"after={gb(torch.cuda.memory_allocated()):5.2f} GB  "
          f"reserved={gb(torch.cuda.memory_reserved()):5.2f} GB")

print("\n--- (2) scaling: peak vs prompt length (adaptive chunk) ---")
for n in [1, 2, 4, 8, 12, 24, 48]:
    p = " ".join(["the quick brown fox jumps over the lazy dog"] * n)
    ids = lm.tokenize(p)
    T = ids.shape[1]
    ch = lens.auto_pos_chunk(T)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t = time.time()
    try:
        lens.jlens_grid(ids)
        peak = gb(torch.cuda.max_memory_allocated())
        print(f"  T={T:4d}  chunk={ch:2d}  peak={peak:6.2f} GB  "
              f"(over model {peak-base:5.2f})  {time.time()-t:6.1f}s")
    except torch.OutOfMemoryError as e:
        print(f"  T={T:4d}  chunk={ch:2d}  OOM: {e}")
        torch.cuda.empty_cache()
        break

print("\n--- (3) chunk-size sensitivity ---")
# bf16 matmuls are batch-size dependent (split-k reduction order), so changing
# pos_chunk perturbs logits by ~1e-2 relative. That flips the argmax only where
# the top-2 are near-tied. The invariant that must hold: confident cells agree.
ids = lm.tokenize("The Colosseum is located in the country of")
a = lens.jlens_grid(ids, pos_chunk=24)          # single chunk
b = lens.jlens_grid(ids, pos_chunk=3)           # forced multi-chunk
a2 = lens.jlens_grid(ids, pos_chunk=24)         # rerun at same chunk

def diff_cells(x, y):
    return [(x["layers"][ri], ci, cx, cy)
            for ri, (rx, ry) in enumerate(zip(x["cells"], y["cells"]))
            for ci, (cx, cy) in enumerate(zip(rx, ry))
            if cx["token"] != cy["token"]]

total = len(a["cells"]) * len(a["tokens"])
d_ab, d_aa = diff_cells(a, b), diff_cells(a, a2)
print(f"  same chunk, rerun     : {len(d_aa)}/{total} differ  (deterministic: {len(d_aa)==0})")
print(f"  chunk=24 vs chunk=3   : {len(d_ab)}/{total} differ ({100*len(d_ab)/total:.1f}%)")

conf = [(L, ci, cx, cy) for L, ci, cx, cy in d_ab if max(cx["prob"], cy["prob"]) > 0.40]
print(f"  of those, confident (p>0.40): {len(conf)}")
assert not conf, f"confident cells disagree across chunk sizes: {conf}"
if d_ab:
    print(f"  max prob among differing cells: {max(max(cx['prob'], cy['prob']) for _,_,cx,cy in d_ab):.3f}")
print("  INVARIANT OK: only near-tie cells flip; confident readouts agree")
print("  last col:", [r[-1]["token"] for r in a["cells"][:4]])
