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

# ── SAFETY ─────────────────────────────────────────────────────────────────
# The crash that motivated this was 190k ctx + --skip-mem-preflight in a LOOP:
# each server reserved a ~190k-token KV cache with the OOM guard off, exhausting
# 16 GB and swap-locking the Mac. The real safety lever is the BOUNDED context,
# not the preflight gate — at ctx <= MAX_CTX the worst-case allocation (model +
# KV + one prefill chunk) fits in 16 GB even with the gate off.
#
# So this sweep:
#   1. Caps the context: CTX_SIZE (default = largest prompt + 8192) is rejected
#      if it exceeds MAX_CTX (default 32768). This is what prevents the crash.
#   2. Runs ONE server at a time and waits for each to fully exit (release wired
#      memory) before the next starts — no overlapping ~12 GB allocations.
#   3. Adds --skip-mem-preflight ON PURPOSE: the conservative auto-budget (which
#      reserves the full 2 GB prefix-cache up front) otherwise rejects perfectly
#      valid bench prompts with "prompt exceeds maximum context length". Safe
#      here precisely because ctx is capped. Also caps the prefix-cache bytes.
# The script OWNS --ctx-size / --skip-mem-preflight / --prefix-cache-mem; don't
# pass them in EXTRA_FLAGS.
for _bad in '--ctx-size' '--skip-mem-preflight' '--prefix-cache-mem'; do
  if printf '%s' "$EXTRA_FLAGS" | grep -q -- "$_bad"; then
    echo "REFUSING: the sweep manages $_bad itself (use CTX_SIZE / PREFIX_CACHE_MEM env, not EXTRA_FLAGS)." >&2
    exit 1
  fi
done
MAX_CTX="${MAX_CTX:-32768}"
PREFIX_CACHE_MEM="${PREFIX_CACHE_MEM:-256MB}"
MAX_SZ=0; for _s in $PROMPT_SIZES; do (( _s > MAX_SZ )) && MAX_SZ=$_s; done
CTX_SIZE="${CTX_SIZE:-$(( MAX_SZ + 8192 ))}"
if (( CTX_SIZE > MAX_CTX )); then
  echo "REFUSING: CTX_SIZE=$CTX_SIZE exceeds MAX_CTX=$MAX_CTX — a large ctx with skip-preflight is exactly what crashed the Mac." >&2
  echo "          Lower CTX_SIZE/PROMPT_SIZES, or raise MAX_CTX explicitly if you KNOW it fits (16 GB: keep <= ~32768)." >&2
  exit 1
fi

LOG="$(mktemp -t mlxserve-sweep.XXXXXX)"
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$SERVER_PID" 2>/dev/null
  fi
  rm -f "$LOG"
}
trap cleanup EXIT INT TERM

# ~1 token per "tokN " word for ASCII BPE; distinct words so the prompt isn't
# collapsed by the n-gram spec-gate. The CSV records the EXACT tokens the trace
# reports, so approximate generation is fine.
make_prompt() { awk -v n="$1" 'BEGIN{ for (i=0;i<n;i++) printf "tok%d ", i }'; }

start_server() { # extra launch flags as args
  # Never run two servers at once — overlapping ~12 GB wired allocations are
  # what crash the machine. Refuse if anything is already serving on PORT.
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "  ! a server is already responding on port $PORT — stop it first (this sweep won't kill foreign servers)." >&2
    return 1
  fi
  : > "$LOG"
  # ctx is CAPPED (<= MAX_CTX) so --skip-mem-preflight is safe here; it's needed
  # so the conservative auto-budget doesn't reject valid bench prompts. The
  # prefix-cache bytes are capped too. The ctx cap is the real OOM guard.
  # shellcheck disable=SC2086
  MLX_SERVE_COMPILE_FORWARD="$CF" "$BINARY" --model "$MODEL" --serve --port "$PORT" \
    --ctx-size "$CTX_SIZE" --skip-mem-preflight --prefix-cache-mem "$PREFIX_CACHE_MEM" \
    --log-level info --prefill-trace \
    --max-concurrent "$MAX_CONCURRENT" $EXTRA_FLAGS "$@" >>"$LOG" 2>&1 &
  SERVER_PID=$!
  local tries=0
  while (( tries < 240 )); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      # Use the ACTUAL loaded model id for requests. A wrong model id makes the
      # server 404, which (with -f) shows up as an empty response — the cause of
      # the "every run empty" symptom. Fall back to "mlx-serve" if discovery fails.
      MODEL_ID="$(curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | jq -r '.data[0].id // empty' 2>/dev/null)"
      [[ -z "$MODEL_ID" ]] && MODEL_ID="mlx-serve"
      echo "  model id = $MODEL_ID  (ctx=$CTX_SIZE)" >&2
      return 0
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  ! server died on launch; log tail:" >&2; tail -n 20 "$LOG" >&2; return 1; }
    sleep 0.5; tries=$((tries+1))
  done
  echo "  ! server failed health check" >&2; return 1
}
stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null
    # Wait for the process to actually exit so its wired GPU memory is released
    # BEFORE the next server tries to grab another ~12 GB (overlap == crash).
    local t=0
    while kill -0 "$SERVER_PID" 2>/dev/null; do
      sleep 0.5; t=$((t+1)); (( t > 60 )) && { kill -9 "$SERVER_PID" 2>/dev/null; break; }
    done
    wait "$SERVER_PID" 2>/dev/null
  fi
  SERVER_PID=""
  # Settle: give macOS a moment to reclaim the wired allocation + free the port.
  sleep 3
}

# Send one non-streaming chat request; echoes the response JSON. Marks the log
# with a sentinel first so we only read THIS request's trace line. NOTE: no
# `-f` — we want the error BODY on a non-2xx (e.g. a 400 "prompt exceeds context
# length") instead of an empty string, so failures are diagnosable. The body is
# returned either way; the caller checks for `.choices`.
MODEL_ID="mlx-serve"
send() { # prompt max_tokens
  echo "===SWEEP-MARK===" >> "$LOG"
  local body
  body="$(jq -nc --arg p "$1" --arg model "$MODEL_ID" --argjson mt "$2" \
    '{model:$model,messages:[{role:"user",content:$p}],max_tokens:$mt,temperature:0,stream:false}' \
  | curl -s -m 900 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'content-type: application/json' -d @-)"
  # If the server returned an error object (or nothing), surface it for the caller.
  if ! printf '%s' "$body" | jq -e '.choices' >/dev/null 2>&1; then
    local err; err="$(printf '%s' "$body" | jq -r '.error.message // .error // .detail // empty' 2>/dev/null)"
    echo "  ! request failed: ${err:-<empty/non-JSON: ${body:0:200}>}" >&2
    return 1
  fi
  printf '%s' "$body"
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
