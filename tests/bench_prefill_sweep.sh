#!/bin/bash
# bench_prefill_sweep.sh — prefill-focused config sweep for mlx-serve (Apple Silicon).
#
# Measures COLD and WARM prompt-prefill throughput across prompt sizes and the
# perf tuning knobs, by parsing the server's OWN `[prefill-trace]` line (enabled
# here with --prefill-trace). That line is far more accurate for prefill than a
# client-side TTFT measurement, because it excludes network, request parsing,
# tokenization, and queueing — it times only the chunked forward + eval + the
# final lm_head forward on the GPU.
#
# It also records the compiled-forward status field (added by the perf/m5
# instrumentation commit): for each config you can SEE whether the compiled
# fast path actually ran, or which gate condition blocked it (normally
# "per-slot-cache" in --serve mode — see docs/PERF_TUNING.md).
#
# Usage:
#   zig build -Doptimize=ReleaseFast        # REQUIRED — a Debug binary is 2-4x slower
#   tests/bench_prefill_sweep.sh <model-dir-or-path> [out.csv]
#
# Sweep dimensions (override via env, space-separated lists):
#   PROMPT_SIZES   approx prompt token counts          (default: 1000 4000 8000 16000 32000)
#   PREFILL_CHUNKS --prefill-chunk values              (default: 8192 16384)
#   SSM_STRIDES    --ssm-checkpoint-stride values      (default: 256 2048)
#   MTP_MODES      on|off  (off => --no-mtp)           (default: on off)
#   COMPILE_FWD    0|1 for MLX_SERVE_COMPILE_FORWARD   (default: 0)
#   PREFIX_CACHE   on|off (off => --prefix-cache-entries 0; warm phase skipped)
#                                                       (default: on)
#   MAX_CONCURRENT --max-concurrent                    (default: 1)
#   DECODE_TOKENS  if >0, also report approx decode tok/s with this max_tokens (default: 0)
#   PORT           server port                          (default: 11260)
#   BINARY         mlx-serve path                       (default: ./zig-out/bin/mlx-serve)
#   EXTRA_FLAGS    appended to every server launch (e.g. "--kv-quant 8")
#
# Output: a CSV (default docs/perf-csvs/prefill-sweep-<ts>.csv) plus a live table
# on stdout. Memory pressure is NOT captured here — watch Activity Monitor's GPU
# memory or the server's `[hot-cache] resident=...` log line during the run.
set -uo pipefail

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  echo "usage: $0 <model-dir-or-path> [out.csv]" >&2
  exit 2
fi
TS="$(date +%Y%m%d-%H%M%S)"
OUT="${2:-docs/perf-csvs/prefill-sweep-$TS.csv}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PORT="${PORT:-11260}"

PROMPT_SIZES="${PROMPT_SIZES:-1000 4000 8000 16000 32000}"
PREFILL_CHUNKS="${PREFILL_CHUNKS:-8192 16384}"
SSM_STRIDES="${SSM_STRIDES:-256 2048}"
MTP_MODES="${MTP_MODES:-on off}"
COMPILE_FWD="${COMPILE_FWD:-0}"
PREFIX_CACHE="${PREFIX_CACHE:-on}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
DECODE_TOKENS="${DECODE_TOKENS:-0}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

[[ -x "$BINARY" ]] || { echo "build first: zig build -Doptimize=ReleaseFast (binary=$BINARY missing/not executable)" >&2; exit 1; }
command -v jq  >/dev/null || { echo "need jq on PATH" >&2; exit 1; }
command -v curl >/dev/null || { echo "need curl on PATH" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

LOG="$(mktemp -t mlxserve-sweep.XXXXXX)"
SERVER_PID=""
cleanup() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
  pkill -9 -x mlx-serve 2>/dev/null
  rm -f "$LOG"
}
trap cleanup EXIT INT TERM

# ~1 token per "tokN " word for ASCII BPE; distinct words so the prompt isn't
# collapsed by the n-gram spec-gate. The CSV records the EXACT tokens the trace
# reports, so approximate generation is fine.
make_prompt() { awk -v n="$1" 'BEGIN{ for (i=0;i<n;i++) printf "tok%d ", i }'; }

start_server() { # extra launch flags as args
  pkill -9 -x mlx-serve 2>/dev/null; sleep 1
  : > "$LOG"
  # shellcheck disable=SC2086
  MLX_SERVE_COMPILE_FORWARD="$CF" "$BINARY" --model "$MODEL" --serve --port "$PORT" \
    --log-level info --prefill-trace --max-concurrent "$MAX_CONCURRENT" $EXTRA_FLAGS "$@" >>"$LOG" 2>&1 &
  SERVER_PID=$!
  local tries=0
  while (( tries < 240 )); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  ! server died on launch; log tail:" >&2; tail -n 20 "$LOG" >&2; return 1; }
    sleep 0.5; tries=$((tries+1))
  done
  echo "  ! server failed health check" >&2; return 1
}
stop_server() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  pkill -9 -x mlx-serve 2>/dev/null; sleep 1
}

# Send one non-streaming chat request; echoes the response JSON. Marks the log
# with a sentinel first so we only read THIS request's trace line.
send() { # prompt max_tokens
  echo "===SWEEP-MARK===" >> "$LOG"
  jq -nc --arg p "$1" --argjson mt "$2" \
    '{model:"mlx-serve",messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0,stream:false}' \
  | curl -sf -m 900 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'content-type: application/json' -d @-
}

# Last [prefill-trace] line since the most recent sentinel.
trace_since_mark() { awk '/===SWEEP-MARK===/{buf=""} /\[prefill-trace\]/{buf=$0} END{print buf}' "$LOG"; }
field()    { printf '%s' "$1" | sed -nE "s/.*[[:space:]]$2=([0-9]+).*/\1/p"; }
# The compiled status is the LAST field and may contain spaces/parens
# (e.g. "skipped: per-slot-cache (ctx.cache != &xfm.cache)"), with the optional
# [mtp]/[drafter]/[pld]/[capture-hidden] flags appended after it. Capture from
# `compiled=` to end of line, then strip the trailing flag tokens.
compiled() {
  printf '%s' "$1" \
    | sed -nE 's/.* compiled=(.*)$/\1/p' \
    | sed -E 's/ \[(mtp|drafter|pld|capture-hidden)\].*$//'
}
tps() { # tokens total_ms  -> tok/s (0 if total_ms==0)
  awk -v t="$1" -v ms="$2" 'BEGIN{ if (ms+0>0) printf "%.1f", t*1000.0/ms; else print "0" }'
}

emit() { echo "$*" >> "$OUT"; }
HDR="prompt_req|chunk|ssm_stride|mtp|compile_fwd|prefix_cache|max_conc|phase|tokens|chunks|ssm_cps|warm_off|chunked_ms|eval_ms|clear_ms|last_ms|total_ms|prefill_tps|compiled|cached_tokens"
emit "$HDR"
printf '%s\n' "$HDR" | tr '|' '\t'

run_phase() { # prompt max_tokens phase chunk ssm mtp pc size
  local prompt="$1" mt="$2" phase="$3" chunk="$4" ssm="$5" mtp="$6" pc="$7" size="$8"
  local resp; resp="$(send "$prompt" "$mt")"
  [[ -z "$resp" ]] && { echo "  ! empty response ($phase)" >&2; return 1; }
  local cached; cached="$(printf '%s' "$resp" | jq -r '.usage.prompt_tokens_details.cached_tokens // .usage.cached_tokens // 0' 2>/dev/null)"
  local line; line="$(trace_since_mark)"
  [[ -z "$line" ]] && { echo "  ! no [prefill-trace] for $phase (is --prefill-trace honored?)" >&2; return 1; }
  local tok ch cps wo cms ems clms lms tms cmp pt
  tok="$(field "$line" tokens)"; ch="$(field "$line" chunks)"; cps="$(field "$line" ssm_cps)"
  wo="$(field "$line" warm_off)"; cms="$(field "$line" chunked)"; ems="$(field "$line" eval)"
  clms="$(field "$line" clear)"; lms="$(field "$line" last_token)"; tms="$(field "$line" total)"
  cmp="$(compiled "$line")"
  pt="$(tps "${tok:-0}" "${tms:-0}")"
  # Columns match $HDR: prompt_req|chunk|ssm_stride|mtp|compile_fwd|prefix_cache|max_conc|phase|...
  local row="$size|$chunk|$ssm|$mtp|$CF|$pc|$MAX_CONCURRENT|$phase|${tok:-?}|${ch:-?}|${cps:-?}|${wo:-?}|${cms:-?}|${ems:-?}|${clms:-?}|${lms:-?}|${tms:-?}|$pt|${cmp:-?}|${cached:-0}"
  emit "$row"
  printf '%s\n' "$row" | tr '|' '\t'
}

echo "# mlx-serve prefill sweep  model=$MODEL  out=$OUT"
echo "# binary=$BINARY  $(du -h "$BINARY" 2>/dev/null | cut -f1) (must be ReleaseFast ~4.5MB; Debug ~2x is a silent 2-4x perf regression)"

for chunk in $PREFILL_CHUNKS; do
  for ssm in $SSM_STRIDES; do
    for mtp in $MTP_MODES; do
      for CF in $COMPILE_FWD; do
        flags=( --prefill-chunk "$chunk" --ssm-checkpoint-stride "$ssm" )
        [[ "$mtp" == "off" ]] && flags+=( --no-mtp )
        pc="$PREFIX_CACHE"
        [[ "$pc" == "off" ]] && flags+=( --prefix-cache-entries 0 )
        echo ">> config chunk=$chunk ssm=$ssm mtp=$mtp compile_fwd=$CF prefix_cache=$pc max_conc=$MAX_CONCURRENT"
        start_server "${flags[@]}" || { echo "  skipping config (launch failed)"; continue; }
        for sz in $PROMPT_SIZES; do
          prompt="$(make_prompt "$sz")"
          # COLD: first time this prompt is seen this server-lifetime -> cache miss.
          run_phase "$prompt" 1 cold "$chunk" "$ssm" "$mtp" "$pc" "$sz"
          if [[ "$pc" != "off" ]]; then
            # WARM: identical prompt -> hot prefix cache restores most of it.
            run_phase "$prompt" 1 warm "$chunk" "$ssm" "$mtp" "$pc" "$sz"
          fi
          if (( DECODE_TOKENS > 0 )); then
            # Approx decode tok/s = completion_tokens / (client_elapsed - prefill_total).
            t0=$(python3 -c 'import time;print(time.time())')
            resp="$(send "$prompt" "$DECODE_TOKENS")"
            t1=$(python3 -c 'import time;print(time.time())')
            line="$(trace_since_mark)"; tms="$(field "$line" total)"
            comp="$(printf '%s' "$resp" | jq -r '.usage.completion_tokens // 0')"
            dtps="$(awk -v c="$comp" -v dt="$t1" -v st="$t0" -v pf="${tms:-0}" 'BEGIN{ w=(dt-st)-pf/1000.0; if (w>0) printf "%.1f", c/w; else print "0" }')"
            row="$sz|$chunk|$ssm|$mtp|$CF|$pc|$MAX_CONCURRENT|decode|$comp|-|-|-|-|-|-|-|${tms:-?}|$dtps|-|-"
            emit "$row"; printf '%s\n' "$row" | tr '|' '\t'
          fi
        done
        stop_server
      done
    done
  done
done

echo "# done -> $OUT"
echo "# tip: sort warm vs cold prefill_tps, and check the 'compiled' column to confirm whether MLX_SERVE_COMPILE_FORWARD engaged."
