# Building mlx-serve against our MLX fork (flash-256 / quant-grail)

This branch (`build/vendored-mlx`) vendors **giaki3003/mlx** and **giaki3003/mlx-c**
as submodules (`lib/mlx`, `lib/mlx-c`) and builds them into `lib/mlx-dist`, which
`build.zig` links via `addMlxLib()` — same pattern as `lib/llama` / `addLlamaLib`.

## Quickstart (Apple Silicon)
```bash
git clone --recursive -b build/vendored-mlx https://github.com/giaki3003/mlx-serve.git
cd mlx-serve
./scripts/ornith/build-mlx.sh     # cmake-builds lib/mlx + lib/mlx-c -> lib/mlx-dist
zig build                          # links lib/mlx-dist (no Homebrew mlx/mlx-c)
```
Already cloned without `--recursive`? `git submodule update --init --recursive`.
Re-run `build-mlx.sh` after bumping a submodule.

## What changed vs Homebrew build
- `build.zig`: `addMlxLib()` links `lib/mlx-dist`; `mlx`/`mlx-c` dropped from the
  `brew list` preflight (webp still Homebrew). Applied by
  `patch-mlx-serve-build.py` (idempotent).
- `lib/mlx` → `feat/flash256-quant-grail` (PR #3660 flash-256 merged; the grail TODO).
- `lib/mlx-c` → `fba4470` (bindings for MLX 0.31.2).

## The work (grail) + full design
See **`lib/mlx/HANDOVER.md`** (after submodule init) — flash-256, the fused
quantized-flash kernel design, version matching, and the pitfalls already paid for.

## UNVERIFIED on macOS
Authored on Linux (no Metal). First build will need confirmation: static-vs-shared
(`SHARED=` in build-mlx.sh), where `mlx.metallib` lands, and whether mlx-c `fba4470`
compiles against our mlx (v0.31.2+73) — if not, rebase #3660 onto the `v0.31.2` tag
(HANDOVER §3 "Version matching").
