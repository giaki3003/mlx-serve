# Prefill performance tuning (Apple Silicon)

This document covers **prompt-prefill / TTFT** tuning for the native MLX engine —
chunking, SSM-checkpoint stride, caches, the wired-memory limit, and the
`--perf-preset` bundles. It also summarizes a June-2026 prefill audit and tracks
the experimental, behavior-changing optimizations that are gated behind opt-in
flags pending on-device benchmarking.

> All numbers depend on the model, prompt shape, and Mac. **Build ReleaseFast**
> (`zig build -Doptimize=ReleaseFast`) before measuring — a Debug binary is
> 2-4× slower and makes everything look like a regression.

## TL;DR

```bash
# Raw cold-prefill throughput (single user, benchmark):
mlx-serve --model <dir> --serve --perf-preset cold-prefill

# Claude Code / OpenCode style agent (warm prefix reuse):
mlx-serve --model <dir> --serve --perf-preset coding-agent

# Conservative on a 16 GB Mac:
mlx-serve --model <dir> --serve --perf-preset low-memory

# Sweep configs and read accurate cold/warm prefill tok/s from the trace:
tests/bench_prefill_sweep.sh <model-dir>
```

## Measuring prefill: the `[prefill-trace]` line

`--prefill-trace` (or `--log-level debug`) emits one line per request:

```
[prefill-trace] tokens=8001 chunks=1 chunk_size=16384 ssm_stride=2048 ssm_cps=4 \
  warm_off=0 chunked=410ms eval=31ms clear=7ms last_token=12ms total=460ms \
  compiled=skipped: per-slot-cache (ctx.cache != &xfm.cache) [mtp] [pld]
```

- `chunks` / `chunk_size` — how many pieces the prefill split into, and the
  configured chunk. More chunks ⇒ more eval+clear barriers.
- `ssm_stride` / `ssm_cps` — effective SSM-checkpoint stride and checkpoints
  captured (hybrid GatedDeltaNet/Mamba models only).
- `warm_off` — how many leading tokens the hot prefix cache restored (warm reuse).
- `chunked` / `eval` / `clear` / `last_token` / `total` — time decomposition.
  `clear` is the per-chunk `mlx_clear_cache` cost (see the experimental work below).
- `compiled=` — **whether the compiled full-forward fast path ran, or why not.**
  In `--serve` mode this reads `per-slot-cache` by default (each request gets its
  own KV cache, so the gate `ctx.cache == &xfm.cache` is false). See the audit.

`tests/bench_prefill_sweep.sh <model> [out.csv]` automates a sweep over prompt
sizes × `{prefill-chunk, ssm-stride, mtp on/off, compile-forward, …}` and writes
a CSV of cold + warm prefill tok/s parsed from this line. Override the sweep with
env vars (`PROMPT_SIZES`, `PREFILL_CHUNKS`, `SSM_STRIDES`, `MTP_MODES`,
`COMPILE_FWD`, `PREFIX_CACHE`, `MAX_CONCURRENT`, `DECODE_TOKENS`, `EXTRA_FLAGS`).

### Per-component split: `--prefill-profile`

To find *where* prefill compute goes — the GatedDeltaNet mixers (24 layers), the
full-attention mixers (8 layers), or the dense FFN — run with `--prefill-profile`
(or `MLX_SERVE_PREFILL_PROFILE=1`). It emits a second line:

```
[prefill-profile] gdn=…ms (…%) attn=…ms (…%) mlp=…ms (…%) [serialized — ratio only]
```

It forces an `mlx_eval` after each layer's mixer and after the rest of the layer,
so the timers capture real per-component GPU time. **This serializes the layer
pipeline, so the total/`prefill_tps` in `[prefill-trace]` is inflated while
profiling — read the percentages as a RATIO, not a speed.** Use it on a single
request (benchmark), not under serving load. The split tells you whether a
vectorized GDN kernel is worth it (gdn dominates) or whether you're at the FFN
4-bit-matmul ceiling (mlp dominates, little room — MLX's qmm is already tuned).

## Tuning flags

| Flag | Default | What it does |
|---|---|---|
| `--prefill-chunk <n>` | 8192 | Tokens per prefill chunk. Larger keeps mid-size prompts single-chunk (fewer eval/clear barriers) at higher peak activation memory (~`8·chunk·max(hidden,ffn)·2` bytes). `MLX_SERVE_PREFILL_CHUNK` env wins. |
| `--ssm-checkpoint-stride <n>` | 256 | Tokens between SSM/conv-state snapshots during hybrid prefill. Smaller = finer warm prefix reuse but more sub-chunking; `0` disables (hybrid then bypasses the hot cache). MoE is auto-coarsened to ≥ `prefill_chunk`. |
| `--ssm-checkpoint-max <n>` | 32 | Max SSM checkpoints retained per request (memory bound). `0` = unlimited. |
| `--prefix-cache-entries <n>` | 32 | Hot prefix cache capacity (entries). `0` disables. On hybrid SSM models the entry count is the reliable memory lever (the byte budget under-counts SSM state — see audit). |
| `--prefix-cache-mem <n>{KB,MB,GB}` | 2GB | Hot prefix cache KV-byte budget. `0`/`off` disables the byte cap. |
| `--tokenize-cache-entries <n>` | 4 | Per-model chat-render + tokenize LRU. `0` disables. |
| `--wired-limit <n>{KB,MB,GB}\|auto\|off` | `auto` (= `min(0.9·RAM, RAM−2GB)`) | GPU memory ceiling. `applyGpuLimit` sets BOTH the wired limit (capped at the device working set, ~11.9 GB on a 16 GB Mac — MLX rejects larger) AND the MLX memory limit (the real lever; can exceed the working set, the excess being pageable). Raises the ceiling for long-context prefill on a 16 GB Mac; `off` = device default. |
| `--no-mtp` | MTP on for dense | Disables the Qwen native MTP head. **Relevant to prefill:** MTP is *decode* speculation but currently taxes *prefill* (see audit). `--no-mtp` removes that tax. |
| `--max-concurrent <n>` | 1 | Submit-queue size for continuous batching. Batches *decode*, not prefill. |
| `--prefill-trace` | off | Force the trace line at info level. |
| `MLX_SERVE_COMPILE_FORWARD=1` | off | Experiment: compile the full forward. Currently unreachable in `--serve` (per-slot cache) — see audit + the staged single-slot fast path below. |

## Presets

`--perf-preset <name>` bundles the above into a workload starting point. **These
are starting points for on-device benchmarking, not proven optima.** Individual
flags passed *after* `--perf-preset` override it (left-to-right parse). When
`--perf-preset` is absent, nothing changes.

| preset | prefill-chunk | ssm-stride / max | prefix entries / mem | tokenize | max-conc | mtp | pld | kv-quant | trace |
|---|---|---|---|---|---|---|---|---|---|
| `cold-prefill` | 16384 | 0 / 0 | 0 / 0 | 0 | 1 | off | off | off | on |
| `coding-agent` | 16384 | 2048 / 16 | 32 / 2GB | 16 | 1 | on | on | off | off |
| `low-memory` | 8192 | 4096 / 8 | 8 / 512MB | 8 | 1 | on | on | 8 | off |
| `max-throughput` | 16384 | 2048 / 16 | 32 / 2GB | 32 | 4 | on | on | off | off |

## Audit summary (June 2026)

A multi-agent audit of the prefill path on a base M5 (16 GB) verified the
following (every finding cross-checked against the MLX 0.32 source + docs):

1. **MTP taxes prefill on dense models.** Native MTP defaults ON for dense
   Qwen3.5/3.6. When active, every prefill chunk runs `forwardWithCaptureAll`
   (materializing the full-prompt hidden state) + `appendHistory`, **and** it
   disables the compiled fast path — all paid during prefill to speed *decode*.
   Immediate lever: `--no-mtp` (or `cold-prefill` preset). A lazy/last-K MTP
   history is staged below.
2. **Per-chunk `mlx_clear_cache()` hurts multi-chunk prefills.** It frees the
   entire MLX buffer cache every chunk, defeating the allocator's free
   `reuse_from_cache` between same-shaped chunks (confirmed in MLX
   `allocator.cpp`). MLX already reclaims under pressure. Single-chunk (≤ chunk)
   prefills pay it once (negligible); long prompts pay it per chunk.
3. **`ssm_checkpoint_stride=256` over-chunks dense GDN prefill.** A 32k prompt
   becomes ~128 chunks (each an eval+clear barrier) vs ~16 at stride 2048.
   Naive coarsening to `prefill_chunk` was tried and reverted (it killed sub-8k
   prefix reuse); an *adaptive* dense coarsening is staged below.
4. **The compiled full-forward is dead code in `--serve`.** The gate requires
   `ctx.cache == &xfm.cache`, but the scheduler gives every slot its own cache.
   `compileForward()` only runs from the global load path, so
   `MLX_SERVE_COMPILE_FORWARD=1` currently does nothing in server mode. **And**
   even if made reachable: mlx-c's `mlx_compile(fun, shapeless)` exposes no
   input/output state threading, so a compiled forward that mutates the KV/SSM
   cache as a side effect may be *incorrect* (MLX purity rule) — it needs a
   byte-equivalence test before it can be trusted. A single-slot reachability
   path + that test are staged below.
5. **Keep the custom GatedDeltaNet Metal kernel.** MLX 0.32 has no
   chunked-scan / SSM / delta-rule primitive; the kernel is a single-dispatch
   fused recurrence, not a per-token relaunch.
6. **Hot prefix-cache byte budget under-counts retained SSM state ~3.4×** on
   hybrid models — use `--prefix-cache-entries` (not the byte cap) as the
   memory lever there.

## Experimental / staged optimizations (need on-device verification)

These behavior-changing fixes from the audit are implemented behind **opt-in
flags, default OFF**, each with a test. They must be built + benchmarked +
(where noted) equivalence-checked on a Mac before any default is flipped.
Changing a default also requires updating the Swift `ServerOptions.toCLIArgs`
mirror (see CLAUDE.md "ServerOptions defaults must mirror the Zig server
defaults").

<!-- Updated as each staged commit lands on perf/m5-prefill-optimizations. -->

### `--kv-attn-prefill-tiled` — head_dim-256 prefill experiment (needs on-device A/B)

**The finding.** MLX 0.32's *fused* attention kernel is only built for head_dim
∈ {64, 80, 128} (`steel_attention.metal`), and `use_fallback` hardcodes the same
set for the prefill path (`scaled_dot_product_attention.cpp:626-633`). Qwen3.5
(Ornith) uses **head_dim 256**, so **dense prefill attention falls to MLX's
unfused SDPA** (transformer.zig calls `mlx_fast_scaled_dot_product_attention`),
which materializes the full `[H_q, chunk, kv]` score matrix — slow, and the reason
you must keep `--prefill-chunk` tiny (small chunks bound that `chunk·kv`
transient). Decode (seq≤8) at 256 *is* fused. This — not chunking (~5%) or MTP
(~1–2%) — is the real prefill ceiling on this model.

**Two candidate fixes, one open question.** `tiledCausalAttention` (the K-tiled
flash-2 loop in `quantAttention`) avoids the full score matrix, but it **only
tiles the KEY axis** — the per-block scores are `[H_q, chunk, block]`, so the
QUERY axis (`chunk`) stays untiled (see commit `dd1a7d0`). So:
- vs dense at head_dim 256 (which is **unfused**, `chunk·kv`): K-tiling is
  `chunk·block`, **smaller at long context** (block≪kv) — likely a win for the
  190k regime, *if* you keep `--prefill-chunk` moderate.
- the teammate's `dd1a7d0` note assumes the dense path is flash-tiled (~245 MB);
  that holds for head_dim {64,80,128} but **not 256**, so for *this* model the
  comparison favors K-tiling more than that note implies. **This is the open
  question to settle empirically.**

`--kv-attn-prefill-tiled` (opt-in, default OFF) routes prefill through the K-tiled
path so you can **measure** it. The fully flat fix (any chunk size) is a **Q-tile
loop** wrapping `tiledCausalAttention` (each query block independent; no softmax
state crosses Q-blocks) — deferred; coordinate with the teammate who owns this code.

- **Requires:** `--kv-attn-mode fused` + a quantized KV (`-ctk/-ctv` / `--kv-quant`).
- **Keep `--prefill-chunk` moderate** (e.g. 1024–2048) — the query axis is untiled,
  so a huge chunk makes the `chunk·block` tile large.
- **Validate first:** `zig build test` — the `tiled fused: …` parity tests compare
  the tiled path to dense SDPA, including the new **head_dim-256**, prefill-shaped
  (T_q>1) multi-block case. Then A/B prefill tok/s AND peak memory with the flag
  on vs off at your real ctx — that's what decides whether this helps for 256.
- **Default stays OFF** so it composes with the in-flight teammate work; if the
  A/B shows a win, the durable fix is the Q-tile loop, not this flag.

### `--gdn-vectorized` — value-vectorized GatedDeltaNet prefill kernel

The `--prefill-profile` split showed GatedDeltaNet is ~27% of prefill, and it
runs on the **scalar** lanes (no matrix engine), not the FFN's `qmm_nax`. The
scalar GDN kernel gives every `Dv` output-dim its own simdgroup, so each step
**re-reads q/k `Dv`× over**. `--gdn-vectorized` (opt-in, default OFF) swaps in a
kernel where each thread carries `--gdn-vec-nv` value-dims (default 4) and loads
each q/k element **once** per step, reusing it across them — cutting that
redundant traffic ~NV×. Prefill-only (T>1); decode and spec-decode keep the
scalar kernel. **Uncertain payoff** (~1.2× *if* GDN prefill is bandwidth-bound on
those reads — the q/k may already be L2-resident, in which case it's a wash).

mlx-lm has **no** chunked-matmul GDN to port (its `gated_delta_ops` is a
sequential per-token loop, and its kernel is the same scalar one); the
chunked-matmul form's per-head matmuls are too small (`~128×128`) to feed the
matrix engine well, so this q/k-reuse kernel — not chunking — is the lever.

- **Validate first:** `zig build test` → "GDN value-vectorized kernel matches the
  scalar kernel (parity)" compares both kernels' `y` + state on random inputs.
  Then A/B prefill tok/s with `--gdn-vectorized` on vs off (try `--gdn-vec-nv 4`
  and `8`). The `[prefill-profile]` `gdn` ms before/after shows the real effect.
- A Metal compile error surfaces as `error.MetalKernelCompileFailed` at first use
  (the Zig still builds — the kernel source compiles at runtime).

### Deprioritized by the on-device sweep (June 2026, Ornith-9B, 16 GB)

A sweep (`bench_ornith.sh`) showed prefill is **compute-bound**: chunk size moved
throughput ~5% (and only when `--prefill-chunk` *and* `--ssm-checkpoint-stride`
are raised together — the effective chunk is their min), MTP on/off ~1–2%, and
per-chunk `clear_cache` ~0.3% of total. So the earlier staged orchestration
patches (drop per-chunk clear_cache, MTP-lazy-history, async prefill eval,
adaptive SSM stride, compiled-forward single-slot) are **not worth their
risk/complexity for this model** and are shelved. The real lever is
`--kv-attn-prefill-tiled` above (attention compute), not orchestration.
