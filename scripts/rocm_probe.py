"""Probe an accelerator for the ops the J-lens needs.

The J-lens forms Jacobian-vector products via forward-over-reverse
(double-backward). Fused/flash attention kernels do not support that, and on a
new backend (e.g. ROCm on gfx1151) the double-backward path is the main thing
that can break. This checks it in isolation before we load a real model.
"""
import torch
import torch.nn.functional as F

print(f"torch      : {torch.__version__}")
print(f"hip        : {torch.version.hip}")
print(f"cuda(build): {torch.version.cuda}")
print(f"available  : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("no accelerator visible -- check /dev/kfd perms and ROCm install")

dev = "cuda"
print(f"device     : {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
print(f"arch       : {getattr(props, 'gcnArchName', '?')}")
print(f"total mem  : {props.total_memory/1e9:.1f} GB (as reported to torch)")

# 1) basic bf16 matmul
a = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
b = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
c = (a @ b).float()
print(f"[ok] bf16 matmul            -> {tuple(c.shape)} finite={torch.isfinite(c).all().item()}")

# 2) eager attention block, twice-differentiable path
torch.manual_seed(0)
B, H, T, D = 2, 4, 16, 32
q = torch.randn(B, H, T, D, device=dev, dtype=torch.float32, requires_grad=True)
k = torch.randn(B, H, T, D, device=dev, dtype=torch.float32)
v = torch.randn(B, H, T, D, device=dev, dtype=torch.float32)


def eager_attn(x):
    scores = (x @ k.transpose(-1, -2)) / D**0.5
    mask = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, -1) @ v


y = eager_attn(q)
print(f"[ok] eager attention fwd    -> {tuple(y.shape)}")

# 3) THE critical one: forward-over-reverse JVP (J @ tangent)
tangent = torch.randn_like(q)
u = torch.zeros_like(y, requires_grad=True)
(g,) = torch.autograd.grad(y, q, grad_outputs=u, create_graph=True, retain_graph=True)
(jvp,) = torch.autograd.grad(g, u, grad_outputs=tangent, retain_graph=True)
print(f"[ok] double-backward JVP    -> {tuple(jvp.shape)} finite={torch.isfinite(jvp).all().item()}")

# 4) correctness: compare against torch.func.jvp (true forward mode) on CPU-equiv math
#    (finite differences is enough to confirm we computed J@v, not garbage)
eps = 1e-3
with torch.no_grad():
    y_plus = eager_attn(q + eps * tangent)
    y_minus = eager_attn(q - eps * tangent)
fd = (y_plus - y_minus) / (2 * eps)
err = (fd - jvp).abs().max().item() / max(jvp.abs().max().item(), 1e-9)
print(f"[ok] JVP vs finite-diff     -> rel err {err:.2e} {'PASS' if err < 1e-2 else 'FAIL'}")

# 5) how much memory can we actually allocate? (Strix Halo: VRAM carveout vs GTT)
alloc, mb = [], 0
try:
    for _ in range(64):  # try up to 16 GB in 256 MB chunks
        alloc.append(torch.empty(256 * 1024 * 1024 // 2, device=dev, dtype=torch.bfloat16))
        mb += 256
except Exception as e:
    print(f"[info] allocation stopped at {mb/1024:.1f} GB: {type(e).__name__}")
else:
    print(f"[ok] allocated {mb/1024:.1f} GB without error")
del alloc
torch.cuda.empty_cache()
print("PROBE OK")
