#!/bin/bash
# bench_prefill_ab.sh — A/B the head_dim-256 prefill fix on Apple Silicon.
#
# Compares, across --prefill-chunk values, on ONE long prompt:
#   baseline : dense prefill attention (MLX SDPA — UNFUSED at head_dim 256, so it
#              materializes the full [H_q, chunk, kv] score matrix)
#   tiled    : --kv-attn-prefill-tiled (K-tiled flash-2: per-block [H_q, chunk, block])
#
# Reports prefill tok/s (server [prefill-trace]) AND peak GPU memory (/props) —
# peak memory is the metric that settles the question: if tiled wins, its peak
# stays ~flat as you raise --prefill-chunk while baseline's climbs (chunk*kv).
#
# SAFETY (this crashed a 16 GB Mac once — read this):
#   * One server at a time; waits for full exit + settle before the next.
#   * Context is CAPPED at MAX_CTX (default 32768).
#   * It ESTIMATES each config's attention transient (baseline: H_Q*chunk*kv*2;
#     tiled: H_Q*chunk*block*2) and SKIPS any config above ATTN_CAP_GB (default
#     2 GB) — so a crash-prone dense+big-chunk config is never even launched.
#     A skipped baseline next to a running tiled IS a result ("tiled runs where
#     dense can't").
#   * --skip-mem-preflight is used (so valid prompts run) but only because the
#     ctx cap + the attention cap bound the worst case.
#
# Usage:
#   zig build -Doptimize=ReleaseFast
#   git fetch origin && git reset --hard origin/perf/m5-prefill-optimizations   # if you had the old branch
#   BINARY=./zig-out/bin/mlx-serve tests/bench_prefill_ab.sh
#   # then: open docs/perf-csvs/prefill-ab-*.csv
#
# Env overrides: MODEL, BINARY, PORT, EXTRA_FLAGS, CHUNKS, SIZE, MAX_CTX,
#   ATTN_CAP_GB, H_Q (attention heads for the estimate; Ornith=16), KV_ATTN_BLOCK.
set -uo pipefail

MODEL="${1:-$HOME/Models/Ornith-1.0-9B-4bit-MTP-MLX-Serve}"
OUT="${2:-docs/perf-csvs/prefill-ab-$(date +%Y%m%d-%H%M%S).csv}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PORT="${PORT:-11261}"

# Hold the production KV / fused-attn config constant across BOTH arms.
EXTRA_FLAGS="${EXTRA_FLAGS:--ctk 8 -ctv turbo4 --kv-attn-mode fused}"
CHUNKS="${CHUNKS:-256 1024 2048 4096}"
SIZE="${SIZE:-24000}"            # one long prompt (~tokens); the regime that matters
MAX_CTX="${MAX_CTX:-32768}"
ATTN_CAP_GB="${ATTN_CAP_GB:-2}"  # skip any config whose est. attention transient exceeds this
H_Q="${H_Q:-16}"                 # query heads, for the transient estimate (Ornith=16)
KV_ATTN_BLOCK="${KV_ATTN_BLOCK:-4096}"
PREFIX_CACHE_MEM="${PREFIX_CACHE_MEM:-256MB}"

# The A/B owns these flags — refuse them in EXTRA_FLAGS.
for _bad in '--ctx-size' '--skip-mem-preflight' '--prefix-cache-mem' '--prefix-cache-entries' '--prefill-chunk' '--kv-attn-prefill-tiled' '--kv-attn-block'; do
  if printf '%s' "$EXTRA_FLAGS" | grep -q -- "$_bad"; then
    echo "REFUSING: the A/B manages $_bad itself (use the matching env var, not EXTRA_FLAGS)." >&2; exit 1
  fi
done
CTX_SIZE="${CTX_SIZE:-$(( SIZE + 8192 ))}"
if (( CTX_SIZE > MAX_CTX )); then
  echo "REFUSING: CTX_SIZE=$CTX_SIZE > MAX_CTX=$MAX_CTX. Lower SIZE, or raise MAX_CTX only if you KNOW it fits (16 GB: <= ~32768)." >&2; exit 1
fi
[[ -x "$BINARY" ]] || { echo "build first: zig build -Doptimize=ReleaseFast (binary=$BINARY)" >&2; exit 1; }
command -v jq >/dev/null || { echo "need jq on PATH" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

CAP_BYTES=$(( ATTN_CAP_GB * 1024 * 1024 * 1024 ))
# Estimated attention transient for an arm at a given chunk (bytes, bf16 scores).
est_bytes() { # arm chunk
  local arm="$1" chunk="$2" span
  if [[ "$arm" == "tiled" ]]; then span="$KV_ATTN_BLOCK"; else span="$SIZE"; fi
  echo $(( H_Q * chunk * span * 2 ))
}

LOG="$(mktemp -t mlxserve-ab.XXXXXX)"
SERVER_PID=""; MODEL_ID="mlx-serve"
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$SERVER_PID" 2>/dev/null
  fi
  rm -f "$LOG"
}
trap cleanup EXIT INT TERM

# ~1 token/word filler ("the"), unique prefix per (arm,chunk) so nothing leaks.
make_prompt() { awk -v n="$1" -v salt="$2" 'BEGIN{ printf "ab%s ", salt; for (i=0;i<n;i++) printf "the " }'; }

start_server() { # extra flags...
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "  ! a server is already on port $PORT — stop it first (won't kill foreign servers)." >&2; return 1
  fi
  : > "$LOG"
  # shellcheck disable=SC2086
  "$BINARY" --model "$MODEL" --serve --port "$PORT" \
    --ctx-size "$CTX_SIZE" --skip-mem-preflight --prefix-cache-entries 0 \
    --prefix-cache-mem "$PREFIX_CACHE_MEM" --kv-attn-block "$KV_ATTN_BLOCK" \
    --log-level info --prefill-trace $EXTRA_FLAGS "$@" >>"$LOG" 2>&1 &
  SERVER_PID=$!
  local tries=0
  while (( tries < 300 )); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      MODEL_ID="$(curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | jq -r '.data[0].id // empty' 2>/dev/null)"
      [[ -z "$MODEL_ID" ]] && MODEL_ID="mlx-serve"
      return 0
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  ! server died; log tail:" >&2; tail -n 15 "$LOG" >&2; return 1; }
    sleep 0.5; tries=$((tries+1))
  done
  echo "  ! health timeout" >&2; return 1
}
stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null
    local t=0; while kill -0 "$SERVER_PID" 2>/dev/null; do sleep 0.5; t=$((t+1)); (( t>60 )) && { kill -9 "$SERVER_PID" 2>/dev/null; break; }; done
  fi
  SERVER_PID=""; sleep 3   # let macOS reclaim wired memory before the next server
}
send() { # prompt
  echo "===MARK===" >> "$LOG"
  jq -nc --arg p "$1" --arg m "$MODEL_ID" \
    '{model:$m,messages:[{role:"user",content:$p}],max_tokens:1,temperature:0,stream:false}' \
  | curl -s -m 1200 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @-
}
trace() { awk '/===MARK===/{b=""} /\[prefill-trace\]/{b=$0} END{print b}' "$LOG"; }
fld() { printf '%s' "$1" | sed -nE "s/.*[[:space:]]$2=([0-9]+).*/\1/p"; }
peak_mb() { curl -sf "http://127.0.0.1:$PORT/props" 2>/dev/null | jq -r '((.memory.peak_bytes // 0)/1048576) | floor' 2>/dev/null; }

emit() { echo "$*" >> "$OUT"; }
HDR="arm|chunk|size|est_attn_mb|tokens|chunks|total_ms|prefill_tps|peak_mb|status"
emit "$HDR"; printf '%s\n' "$HDR" | tr '|' '\t'

run() { # arm chunk extra...
  local arm="$1" chunk="$2"; shift 2
  local est; est="$(est_bytes "$arm" "$chunk")"
  local est_mb=$(( est / 1048576 ))
  if (( est > CAP_BYTES )); then
    local row="$arm|$chunk|$SIZE|$est_mb|-|-|-|-|-|skipped-cap(${ATTN_CAP_GB}GB)"
    emit "$row"; printf '%s\n' "$row" | tr '|' '\t'; return
  fi
  if ! start_server --prefill-chunk "$chunk" "$@"; then
    local row="$arm|$chunk|$SIZE|$est_mb|-|-|-|-|-|launch-fail"
    emit "$row"; printf '%s\n' "$row" | tr '|' '\t'; return
  fi
  local prompt; prompt="$(make_prompt "$SIZE" "$arm$chunk")"
  send "$prompt" >/dev/null 2>&1        # run 1: warm the chunk-shape kernels
  local resp; resp="$(send "$prompt")"  # run 2: measured (cold KV — prefix cache off; warm kernels)
  local status="ok"
  if ! printf '%s' "$resp" | jq -e '.choices' >/dev/null 2>&1; then
    status="$(printf '%s' "$resp" | jq -r '.error.message // "no-response"' 2>/dev/null | tr '|' '/' | head -c 40)"
  fi
  local line tok ch tms tps pk
  line="$(trace)"; tok="$(fld "$line" tokens)"; ch="$(fld "$line" chunks)"; tms="$(fld "$line" total)"
  pk="$(peak_mb)"
  tps="$(awk -v t="${tok:-0}" -v ms="${tms:-0}" 'BEGIN{ if (ms+0>0) printf "%.1f", t*1000/ms; else print "0" }')"
  local row="$arm|$chunk|$SIZE|$est_mb|${tok:-?}|${ch:-?}|${tms:-?}|$tps|${pk:-?}|$status"
  emit "$row"; printf '%s\n' "$row" | tr '|' '\t'
  stop_server
}

echo "# A/B prefill — model=$MODEL"
echo "# ctx=$CTX_SIZE size=$SIZE chunks=[$CHUNKS] kv_attn_block=$KV_ATTN_BLOCK attn_cap=${ATTN_CAP_GB}GB H_Q=$H_Q"
echo "# EXTRA_FLAGS=$EXTRA_FLAGS  (prefix cache OFF; run 2 is the measured cold-KV prefill at warm kernels)"
echo "# baseline = dense SDPA prefill; tiled = --kv-attn-prefill-tiled"
for chunk in $CHUNKS; do
  run baseline "$chunk"
  run tiled    "$chunk" --kv-attn-prefill-tiled
done
echo "# done -> $OUT"
echo "# read: per chunk, compare baseline vs tiled on prefill_tps AND peak_mb."
echo "#  - if tiled wins: peak_mb stays ~flat as chunk grows (chunk*block) while baseline climbs (chunk*kv)"
echo "#    and baseline hits 'skipped-cap' first -> tiled lets you raise --prefill-chunk for more tok/s."
echo "#  - if not: prefill_tps is similar/worse -> the dense path is fine for your ctx; report back."
