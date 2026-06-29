#!/usr/bin/env bash
# collect-build-info.sh — gather the facts needed to start Phase 2 (the grail).
#
# Run from the mlx-serve checkout root, AFTER:
#   ./scripts/ornith/build-mlx.sh   (builds lib/mlx + lib/mlx-c -> lib/mlx-dist)
#   zig build -Doptimize=ReleaseFast
# It tolerates partial/failed state and reports what it finds. Paste the WHOLE
# output back — it answers: did mlx-c compile against our mlx fork? shared/static?
# where's the metallib? does zig link the vendored libs (not Homebrew)? GPU arch?
set +e
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT" || exit 1
line(){ printf '\n========== %s ==========\n' "$1"; }

line "ENV"
sw_vers 2>/dev/null
echo "xcode CLT : $(xcode-select -p 2>/dev/null)"
echo "cmake     : $(cmake --version 2>/dev/null | head -1)"
echo "ninja     : $(ninja --version 2>/dev/null)"
echo "zig       : $(zig version 2>/dev/null)"
echo "cpu       : $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
echo "cores/mem : $(sysctl -n hw.ncpu 2>/dev/null) cores / $(( $(sysctl -n hw.memsize 2>/dev/null) / 1073741824 )) GB"

line "GPU ARCH (via your installed mlx python)"
python3 -c "import mlx.core as mx; d=mx.metal.device_info(); print('architecture =', d.get('architecture')); print('max_recommended_working_set_size =', d.get('max_recommended_working_set_size'))" 2>&1 | head

line "SUBMODULE PINS"
git submodule status lib/mlx lib/mlx-c 2>/dev/null
echo "lib/mlx   : $(git -C lib/mlx rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git -C lib/mlx describe --tags --always 2>/dev/null)"
echo "lib/mlx-c : $(git -C lib/mlx-c rev-parse --short HEAD 2>/dev/null) \"$(git -C lib/mlx-c log -1 --format=%s 2>/dev/null)\""

line "MLX-DIST (did build-mlx.sh succeed?)"
ls -la lib/mlx-dist/lib/ 2>/dev/null | grep -iE "libmlx|\.dylib|\.a" || echo "  (lib/mlx-dist/lib not built -> build-mlx.sh did NOT complete; see its output)"
if ls lib/mlx-dist/lib/libmlx*.dylib >/dev/null 2>&1; then echo "  link model : SHARED (dylibs)"
elif ls lib/mlx-dist/lib/libmlx*.a >/dev/null 2>&1; then echo "  link model : STATIC (.a)"
else echo "  link model : NONE"; fi
echo "  metallib :"; find lib/mlx-dist .mlx-build -name '*.metallib' 2>/dev/null | sed 's/^/    /' | head
test -f lib/mlx-dist/include/mlx/c/fast.h && echo "  mlx-c headers: OK" || echo "  mlx-c headers: MISSING"

line "ZIG BUILD (tail; full errors if it failed)"
zig build -Doptimize=ReleaseFast 2>&1 | tail -30
echo "[zig build exit: ${PIPESTATUS[0]}]"

line "BINARY LINKAGE (must point at lib/mlx-dist, NOT /opt/homebrew)"
BIN="$(ls zig-out/bin/mlx-serve 2>/dev/null | head -1)"
echo "binary: ${BIN:-<not built>}"
[ -n "$BIN" ] && { otool -L "$BIN" 2>/dev/null | grep -iE "mlx|rpath|homebrew" | sed 's/^/  /' | head -12; du -h "$BIN" 2>/dev/null; }

line "RUNTIME SMOKE (version only — confirms the dylib + metallib resolve)"
[ -n "$BIN" ] && "$BIN" --version 2>&1 | head -3

line "DONE — paste this whole block back"
