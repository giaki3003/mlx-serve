#!/usr/bin/env python3
"""Patch mlx-serve's build.zig to link the vendored mlx/mlx-c (lib/mlx-dist)
instead of Homebrew. Idempotent; fails loudly if an anchor isn't found exactly
once (so it never half-applies). Run from the mlx-serve root, or pass the path:

    python3 scripts/ornith/patch-mlx-serve-build.py [path/to/build.zig]

Mirrors how mlx-serve vendors llama.cpp (addLlamaLib + lib/llama). After patching,
run scripts/ornith/build-mlx.sh then `zig build`. webp stays a Homebrew dep.
"""
import sys, pathlib

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "build.zig")
src = path.read_text()

if "addMlxLib(" in src:
    print(f"[patch] {path} already patched — nothing to do.")
    sys.exit(0)

def replace_once(s, old, new, label):
    n = s.count(old)
    if n != 1:
        sys.exit(f"[patch] ERROR: anchor '{label}' found {n}x (expected 1). "
                 f"build.zig drifted — patch by hand per HANDOVER.md §3.")
    return s.replace(old, new)

# R1 — drop mlx + mlx-c from the brew preflight (keep webp)
src = replace_once(src,
    '    .{ .name = "mlx", .min = .{ .major = 0, .minor = 31, .patch = 2 } },\n'
    '    .{ .name = "mlx-c", .min = .{ .major = 0, .minor = 6, .patch = 0 } },\n'
    '    .{ .name = "webp", .min = .{ .major = 1, .minor = 6, .patch = 0 } },\n',
    '    .{ .name = "webp", .min = .{ .major = 1, .minor = 6, .patch = 0 } },\n',
    "brew-deps")

# R2 — main module: vendored mlx-c, keep webp from Homebrew
src = replace_once(src,
    '    // mlx-c include/lib paths (homebrew)\n'
    '    mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });\n'
    '    mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });\n'
    '    mod.linkSystemLibrary("mlxc", .{});\n'
    '    mod.linkSystemLibrary("webp", .{});\n',
    '    // mlx + mlx-c from vendored submodules, staged into lib/mlx-dist by\n'
    '    // scripts/ornith/build-mlx.sh (mirrors addLlamaLib / lib/llama).\n'
    '    addMlxLib(b, mod);\n'
    '    // webp is still a Homebrew dep\n'
    '    mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });\n'
    '    mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });\n'
    '    mod.linkSystemLibrary("webp", .{});\n',
    "mod-mlxc")

# R3 — test module: same
src = replace_once(src,
    '    test_mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });\n'
    '    test_mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });\n'
    '    test_mod.linkSystemLibrary("mlxc", .{});\n'
    '    test_mod.linkSystemLibrary("webp", .{});\n',
    '    addMlxLib(b, test_mod);\n'
    '    test_mod.addIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });\n'
    '    test_mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });\n'
    '    test_mod.linkSystemLibrary("webp", .{});\n',
    "test_mod-mlxc")

# R4 — define addMlxLib() just before BrewDep
src = replace_once(src,
    'const BrewDep = struct { name: []const u8, min: std.SemanticVersion };\n',
    'fn addMlxLib(b: *std.Build, module: *std.Build.Module) void {\n'
    '    // libmlx + libmlxc from the lib/mlx and lib/mlx-c submodules, built by\n'
    '    // scripts/ornith/build-mlx.sh into lib/mlx-dist/{include,lib}. One include\n'
    '    // dir carries both mlx/*.h and mlx/c/*.h. use_pkg_config=.no so a stray\n'
    '    // Homebrew mlx-c cannot hijack the link; rpath covers shared libs\n'
    '    // (harmless for static).\n'
    '    module.addIncludePath(b.path("lib/mlx-dist/include"));\n'
    '    module.addLibraryPath(b.path("lib/mlx-dist/lib"));\n'
    '    module.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });\n'
    '    module.addRPath(b.path("lib/mlx-dist/lib"));\n'
    '}\n\n'
    'const BrewDep = struct { name: []const u8, min: std.SemanticVersion };\n',
    "BrewDep")

path.write_text(src)
print(f"[patch] {path} patched: addMlxLib() links lib/mlx-dist; mlx/mlx-c dropped "
      f"from brew preflight. Run scripts/ornith/build-mlx.sh, then `zig build`.")
