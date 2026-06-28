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
# IMPORTANT: this stops every running mlx-serve it can find between configs, so
# don't run it against the same machine where your production server is live.
# The sweep forces --prefill-trace and parses the server's own trace line for
# accurate COLD + WARM prefill tok/s.
#
# Why these sweeps (from the June-2026 prefill audit — see docs/PERF_TUNING.md):
#   - prefill-chunk: your production launch uses 256, which at 190k ctx splits a
#     long prompt into hundreds of chunks, each paying an mlx_eval + per-chunk
#     mlx_clear_cache barrier. Larger chunks (the activation cost is independent
#     of the 190k KV) should cut that dramatically IF they fit — this finds the
#     sweet spot.
#   - mtp on/off: this checkpoint ships an MTP sidecar, so MTP is ON by default
#     and taxes PREFILL (full-prompt hidden capture + per-chunk history append)
#     for a DECODE-only speedup. The off cells quantify that prefill tax.
set -uo pipefail

MODEL="${1:-$HOME/Models/Ornith-1.0-9B-4bit-MTP-MLX-Serve}"
OUT="${2:-docs/perf-csvs/ornith-prefill-sweep-$(date +%Y%m%d-%H%M%S).csv}"

# Hold the production KV / context / memory config constant across every cell.
# (matches: --ctx-size 190000 --skip-mem-preflight -ctk 8 -ctv turbo4 --kv-attn-mode fused)
export EXTRA_FLAGS="${EXTRA_FLAGS:---ctx-size 190000 --skip-mem-preflight -ctk 8 -ctv turbo4 --kv-attn-mode fused}"

# Sweep the prefill knobs. Defaults chosen for the 9B/16 GB long-context case;
# override any via env.
export PREFILL_CHUNKS="${PREFILL_CHUNKS:-256 1024 2048 4096}"
export SSM_STRIDES="${SSM_STRIDES:-256 2048}"
export MTP_MODES="${MTP_MODES:-on off}"
export PROMPT_SIZES="${PROMPT_SIZES:-4000 16000 32000}"
# Prefix cache ON so we also get the WARM-reuse number per config.
export PREFIX_CACHE="${PREFIX_CACHE:-on}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "# Ornith prefill sweep — model=$MODEL"
echo "# EXTRA_FLAGS=$EXTRA_FLAGS"
echo "# chunks=[$PREFILL_CHUNKS] ssm=[$SSM_STRIDES] mtp=[$MTP_MODES] sizes=[$PROMPT_SIZES]"
exec "$DIR/bench_prefill_sweep.sh" "$MODEL" "$OUT"
