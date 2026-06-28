#!/bin/bash
# bench_ornith.sh — tailored prefill sweep for the Ornith-1.0-9B-4bit MTP model
# on a 16 GB Mac. Thin wrapper over bench_prefill_sweep.sh that holds the
# production KV / long-context / fused-attn config constant and sweeps the
# prefill knobs (chunk size, SSM stride, MTP on/off) that the audit flagged.
#
# Usage:
#   zig build -Doptimize=ReleaseFast
#   BINARY=./zig-out/bin/mlx-serve tests/bench_ornith.sh                 # default model path
#   BINARY=./mlx-serve-bin tests/bench_ornith.sh ~/Models/<other-model>  # override binary + model
#
# SAFETY: this runs ONE server at a time with a BOUNDED context (~20k, NOT your
# 190k production value). The bounded ctx is the OOM guard — at ~20k the model +
# KV + one prefill chunk fit in 16 GB, so the sweep can safely use
# --skip-mem-preflight (which it needs, else the conservative auto-budget rejects
# valid bench prompts). What crashed the Mac was 190k ctx + skip-preflight in a
# loop; capping ctx removes that. Prefill *throughput* characteristics (chunk
# count, clear_cache cost, MTP tax) transfer fine from a modest ctx. The sweep
# owns --ctx-size/--skip-mem-preflight/--prefix-cache-mem. Stop any running
# mlx-serve first.
#
# Why these sweeps (June-2026 prefill audit — see docs/PERF_TUNING.md):
#   - prefill-chunk: your production launch uses 256, which splits a long prompt
#     into many chunks, each paying an mlx_eval + per-chunk mlx_clear_cache
#     barrier. Larger chunks (per-chunk activation cost is independent of the KV
#     size) should cut that dramatically — this finds the sweet spot.
#   - mtp on/off: this checkpoint ships an MTP sidecar, so MTP is ON by default
#     and taxes PREFILL (full-prompt hidden capture + per-chunk history append)
#     for a DECODE-only speedup. The off cells quantify that prefill tax.
set -uo pipefail

MODEL="${1:-$HOME/Models/Ornith-1.0-9B-4bit-MTP-MLX-Serve}"
OUT="${2:-docs/perf-csvs/ornith-prefill-sweep-$(date +%Y%m%d-%H%M%S).csv}"

# Hold the asymmetric-KV + fused-attn config constant (low memory anyway). NO
# --ctx-size / --skip-mem-preflight here — the sweep manages a safe ctx + keeps
# the OOM guard on.
export EXTRA_FLAGS="${EXTRA_FLAGS:--ctk 8 -ctv turbo4 --kv-attn-mode fused}"
# Bounded context — fits the prompt sizes below with headroom; ~20k as suggested.
export CTX_SIZE="${CTX_SIZE:-20480}"

# Sweep the prefill knobs. Conservative defaults after the crash; override via env.
export PREFILL_CHUNKS="${PREFILL_CHUNKS:-256 1024 2048}"
export SSM_STRIDES="${SSM_STRIDES:-256 2048}"
export MTP_MODES="${MTP_MODES:-on off}"
export PROMPT_SIZES="${PROMPT_SIZES:-1000 4000 8000 12000}"
# Prefix cache ON so we also get the WARM-reuse number per config.
export PREFIX_CACHE="${PREFIX_CACHE:-on}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "# Ornith prefill sweep — model=$MODEL"
echo "# EXTRA_FLAGS=$EXTRA_FLAGS"
echo "# chunks=[$PREFILL_CHUNKS] ssm=[$SSM_STRIDES] mtp=[$MTP_MODES] sizes=[$PROMPT_SIZES]"
exec "$DIR/bench_prefill_sweep.sh" "$MODEL" "$OUT"
