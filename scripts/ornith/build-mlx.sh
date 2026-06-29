#!/usr/bin/env bash
# Build the vendored MLX + mlx-c submodules into lib/mlx-dist/{include,lib}.
# Mirrors mlx-serve's scripts/fetch-llama.sh (which stages libllama into lib/llama).
# RUN FROM the mlx-serve checkout root. Apple Silicon / macOS only.
#
# Requires: cmake, ninja, git, Xcode CLT (Metal). After this, `zig build` links
# lib/mlx-dist via the addMlxLib() helper from mlx-serve-build.zig.patch.
#
# UNVERIFIED on Mac by the author (built on Linux, no Metal) — expect to confirm
# the flags below against your mlx-c/mlx versions on first run. See HANDOVER.md §3.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

MLX_SUBMODULE="${MLX_SUBMODULE:-lib/mlx}"        # our mlx fork (flash-256 [+ grail])
MLXC_SUBMODULE="${MLXC_SUBMODULE:-lib/mlx-c}"    # our mlx-c fork (0.6.0 [+ grail binding])
DIST="${DIST:-$ROOT/lib/mlx-dist}"               # staging prefix build.zig links against
BUILD="${BUILD:-$ROOT/.mlx-build}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
SHARED="${SHARED:-ON}"                            # ON => dylibs (matches mlxc dylib link model; rpath added by addMlxLib).
                                                  # OFF => static .a, but then build.zig must also link libmlx + Metal frameworks.

echo "[build-mlx] init submodules"
git submodule update --init --recursive "$MLX_SUBMODULE" "$MLXC_SUBMODULE"

echo "[build-mlx] configure (mlx-c drives the build; FetchContent_SOURCE points at our mlx fork)"
# Key flag: FETCHCONTENT_SOURCE_DIR_MLX makes mlx-c use OUR vendored mlx instead of
# downloading the pinned v0.31.2 tag. One build tree -> libmlx + libmlxc, consistent.
cmake -S "$MLXC_SUBMODULE" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$DIST" \
  -DFETCHCONTENT_SOURCE_DIR_MLX="$ROOT/$MLX_SUBMODULE" \
  -DBUILD_SHARED_LIBS="$SHARED" \
  -DMLX_BUILD_METAL=ON \
  -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_C_BUILD_EXAMPLES=OFF

echo "[build-mlx] build + install -> $DIST"
cmake --build "$BUILD" -j"$JOBS"
cmake --install "$BUILD"

echo
echo "[build-mlx] staged:"
ls -1 "$DIST/lib"/libmlx*  2>/dev/null || echo "  (no libmlx* — check install rules / SHARED setting)"
echo "  headers: $DIST/include/mlx , $DIST/include/mlx/c"
test -f "$DIST/include/mlx/c/fast.h" && echo "  mlx-c headers present OK" || echo "  WARN: mlx/c headers missing"
# Locate the metallib (needed at runtime for the Metal backend).
find "$DIST" "$BUILD" -name '*.metallib' 2>/dev/null | sed 's/^/  metallib: /' | head -3
echo "[build-mlx] done. Now: zig build   (links lib/mlx-dist via addMlxLib)"
