#!/usr/bin/env python3
# sdpa256_repro.py — isolate the head_dim=256 fused-SDPA garbage at the REAL
# Ornith shape (GQA 16/4 + causal), which #3660's own test never covered
# (heads=1, no GQA, no mask). Run with the FORK's mlx python:
#
#   cd lib/mlx
#   python3 -m venv /tmp/mlxdev && source /tmp/mlxdev/bin/activate
#   pip install -U pip setuptools wheel nanobind
#   CMAKE_ARGS="-DMLX_BUILD_METAL=ON" pip install -e . -v     # ~20-30 min first time
#   python3 ../../scripts/ornith/sdpa256_repro.py
#
# It prints a 4-row matrix. The row(s) that FAIL tell us whether the bug is the
# causal mask, the GQA head-mapping, or both — i.e. exactly where to look in the
# steel kernel. (qL=16>8 and kL=20000>16384 force the fused steel hd256 path.)
import mlx.core as mx
import math


def ref(q, k, v, scale, causal):  # q:[B,Hq,Tq,D]  k,v:[B,Hkv,Tk,D]
    B, Hq, Tq, D = q.shape
    Hkv, Tk = k.shape[1], k.shape[2]
    rep = Hq // Hkv
    kk = mx.repeat(k, rep, axis=1).astype(mx.float32)  # repeat_interleave: q i -> kv i//rep
    vv = mx.repeat(v, rep, axis=1).astype(mx.float32)
    s = (q.astype(mx.float32) @ kk.swapaxes(-1, -2)) * scale
    if causal:  # query block is the LAST Tq positions (offset Tk-Tq)
        qi = mx.arange(Tq).reshape(Tq, 1) + (Tk - Tq)
        kj = mx.arange(Tk).reshape(1, Tk)
        s = mx.where(kj <= qi, s, mx.array(-1e30, mx.float32))
    return (mx.softmax(s, axis=-1) @ vv).astype(v.dtype)


D, Tq, Tk = 256, 16, 20000  # qL>8 and kL>16384 => fused steel hd256 path
print(f"head_dim={D} Tq={Tq} Tk={Tk}  (fused steel hd256 path)")
print(f"device: {mx.default_device()}")
for (Hq, Hkv, causal, name) in [
    (1, 1, False, "1/1   no-mask   (== the PR's own test)"),
    (1, 1, True, "1/1   causal"),
    (16, 4, False, "16/4  GQA no-mask"),
    (16, 4, True, "16/4  GQA causal   <-- your real shape"),
]:
    q = mx.random.normal((1, Hq, Tq, D)).astype(mx.float16)
    k = mx.random.normal((1, Hkv, Tk, D)).astype(mx.float16)
    v = mx.random.normal((1, Hkv, Tk, D)).astype(mx.float16)
    sc = 1.0 / math.sqrt(D)
    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=sc, mask=("causal" if causal else None)
    )
    r = ref(q, k, v, sc, causal)
    mx.eval(out, r)
    d = mx.abs(out.astype(mx.float32) - r.astype(mx.float32))
    nan = int(mx.isnan(out).sum().item())
    mx_abs = d.max().item()
    ok = "OK  " if (mx_abs < 2e-2 and nan == 0) else "FAIL"
    print(f"  [{ok}] {name:36s} max|Δ|={mx_abs:8.4f}  mean|Δ|={d.mean().item():.6f}  NaNs={nan}")
