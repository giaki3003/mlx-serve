#!/bin/bash
# bench_prefill_profile.sh — per-component prefill profile (GDN vs attn vs FFN).
#
# Starts ONE server with --prefill-profile and reports, per prompt size, the
# split of prefill GPU time across:
#   gdn  : the 24 GatedDeltaNet (linear-attention) mixers
#   attn : the 8 full-attention mixers
#   mlp  : the dense FFN + norms + residuals (all layers)
#
# Answers the open question: is the ~660 tok/s prefill bound by the hand-rolled
# GatedDeltaNet kernel (gdn% large -> a vectorized GDN kernel is worth it) or by
# the FFN 4-bit matmul (mlp% dominant -> MLX's qmm is already tuned, that's the
# ceiling)?
#
# NOTE: --prefill-profile forces an mlx eval per component, which SERIALIZES the
# layer pipeline. So the absolute prefill time is INFLATED — this script reports
# only the RATIO (ms + %), which is the meaningful part. Single server, capped
# ctx, prefix cache off, one prompt at a time.
#
# Usage:
#   zig build -Doptimize=ReleaseFast
#   git fetch origin && git reset --hard origin/perf/m5-prefill-optimizations
#   BINARY=./zig-out/bin/mlx-serve tests/bench_prefill_profile.sh
#
# Env: MODEL, BINARY, PORT, EXTRA_FLAGS, SIZES, CHUNK, MAX_CTX.
set -uo pipefail

MODEL="${1:-$HOME/Models/Ornith-1.0-9B-4bit-MTP-MLX-Serve}"
OUT="${2:-docs/perf-csvs/prefill-profile-$(date +%Y%m%d-%H%M%S).csv}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PORT="${PORT:-11262}"
EXTRA_FLAGS="${EXTRA_FLAGS:--ctk 8 -ctv turbo4 --kv-attn-mode fused}"
SIZES="${SIZES:-4000 8000 16000}"
CHUNK="${CHUNK:-1024}"
MAX_CTX="${MAX_CTX:-32768}"

# The script owns these — refuse them in EXTRA_FLAGS.
for _bad in '--ctx-size' '--skip-mem-preflight' '--prefill-chunk' '--prefill-profile' '--prefill-trace' '--prefix-cache-entries'; do
  if printf '%s' "$EXTRA_FLAGS" | grep -q -- "$_bad"; then
    echo "REFUSING: the profiler manages $_bad itself (use CHUNK/SIZES/MAX_CTX env, not EXTRA_FLAGS)." >&2; exit 1
  fi
done
MAX_SZ=0; for _s in $SIZES; do (( _s > MAX_SZ )) && MAX_SZ=$_s; done
CTX_SIZE="${CTX_SIZE:-$(( MAX_SZ + 8192 ))}"
if (( CTX_SIZE > MAX_CTX )); then
  echo "REFUSING: CTX_SIZE=$CTX_SIZE > MAX_CTX=$MAX_CTX. Lower SIZES (or raise MAX_CTX only if you KNOW it fits)." >&2; exit 1
fi
[[ -x "$BINARY" ]] || { echo "build first: zig build -Doptimize=ReleaseFast (binary=$BINARY)" >&2; exit 1; }
command -v jq >/dev/null || { echo "need jq on PATH" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

LOG="$(mktemp -t mlxserve-prof.XXXXXX)"; SERVER_PID=""; MODEL_ID="mlx-serve"
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$SERVER_PID" 2>/dev/null
  fi
  rm -f "$LOG"
}
trap cleanup EXIT INT TERM

make_prompt() { awk -v n="$1" -v salt="$2" 'BEGIN{ printf "prof%s ", salt; for (i=0;i<n;i++) printf "the " }'; }
send() {
  echo "===MARK===" >> "$LOG"
  jq -nc --arg p "$1" --arg m "$MODEL_ID" \
    '{model:$m,messages:[{role:"user",content:$p}],max_tokens:1,temperature:0,stream:false}' \
  | curl -s -m 1200 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @-
}
prof_line() { awk '/===MARK===/{b=""} /\[prefill-profile\]/{b=$0} END{print b}' "$LOG"; }
pf_ms()  { printf '%s' "$1" | sed -nE "s/.*[[:space:]]$2=([0-9]+)ms.*/\1/p"; }
pf_pct() { printf '%s' "$1" | sed -nE "s/.*[[:space:]]$2=[0-9]+ms \(([0-9]+)%\).*/\1/p"; }

if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "REFUSING: a server is already on port $PORT — stop it first." >&2; exit 1
fi
echo "# starting server (ctx=$CTX_SIZE chunk=$CHUNK) ..."
# shellcheck disable=SC2086
"$BINARY" --model "$MODEL" --serve --port "$PORT" --ctx-size "$CTX_SIZE" --skip-mem-preflight \
  --prefix-cache-entries 0 --prefill-chunk "$CHUNK" --prefill-profile $EXTRA_FLAGS >>"$LOG" 2>&1 &
SERVER_PID=$!
tries=0
while (( tries < 300 )); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "  ! server died; log tail:" >&2; tail -n 15 "$LOG" >&2; exit 1; }
  sleep 0.5; tries=$((tries+1))
done
(( tries >= 300 )) && { echo "  ! health timeout" >&2; exit 1; }
MODEL_ID="$(curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | jq -r '.data[0].id // empty' 2>/dev/null)"
[[ -z "$MODEL_ID" ]] && MODEL_ID="mlx-serve"
echo "# model=$MODEL_ID  EXTRA_FLAGS=$EXTRA_FLAGS"

HDR="size|gdn_ms|gdn_pct|attn_ms|attn_pct|mlp_ms|mlp_pct"
echo "$HDR" > "$OUT"; printf '%s\n' "$HDR" | tr '|' '\t'
for sz in $SIZES; do
  p="$(make_prompt "$sz" "$sz")"
  send "$p" >/dev/null 2>&1            # warm the chunk-shape kernels
  r="$(send "$p")"                     # measured (profile accumulators reset per request)
  if ! printf '%s' "$r" | jq -e '.choices' >/dev/null 2>&1; then
    echo "  ! size=$sz failed: $(printf '%s' "$r" | jq -r '.error.message // "no-response"' 2>/dev/null | head -c 60)" >&2; continue
  fi
  line="$(prof_line)"
  [[ -z "$line" ]] && { echo "  ! no [prefill-profile] line for size=$sz (build current? --prefill-profile honored?)" >&2; continue; }
  row="$sz|$(pf_ms "$line" gdn)|$(pf_pct "$line" gdn)|$(pf_ms "$line" attn)|$(pf_pct "$line" attn)|$(pf_ms "$line" mlp)|$(pf_pct "$line" mlp)"
  echo "$row" >> "$OUT"; printf '%s\n' "$row" | tr '|' '\t'
done
echo "# done -> $OUT"
echo "# read: mlp% dominant -> FFN 4-bit-matmul ceiling (MLX qmm already tuned, little room)."
echo "#       gdn% large (>~35%) -> a vectorized GatedDeltaNet kernel is worth building."
echo "#       attn% should be small (the A/B already cleared the attention path)."
