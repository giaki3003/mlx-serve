// KV-cache quantization backend.
//
// This module owns the storage / dispatch contract between the cache
// (`KVCache` in `transformer.zig`) and the attention call sites. SDPA always
// reads dense `[B, H, T, head_dim]` tensors via `KVCache.denseView` — what the
// cache buffers actually hold is decided here.
//
// v1 ships a single non-trivial scheme: affine group-wise quantization at
// 4 or 8 bits, identical mathematically to mlx-c's existing `mlx_quantize`
// path used for weight quantization. The buffers grow to 3 arrays per K and V
// (q, scales, biases); attention is unchanged because `denseView` calls
// `dequantizeAffine` before returning.
//
// ── Adding a new scheme later (e.g. TurboQuant) ──
//
// The contract is intentionally small so a future session can drop in a new
// scheme without touching `transformer.zig`'s SDPA call sites. Steps:
//
//  1. Add an enum variant to `Scheme` (e.g. `turboquant_1`, `turboquant_2`).
//  2. (Optional) Add per-cache state on `KVCache` for things like rotation
//     matrices. For TurboQuant: `quant_state: ?TurboState` carrying one
//     `[head_dim, head_dim]` orthogonal/Hadamard matrix per K and V per
//     layer (~10 MB total at Gemma 4 E4B). Initialize once at
//     `KVCache.init` from a deterministic seed.
//  3. Add two functions here mirroring `quantizeAffine`/`dequantizeAffine`:
//        quantizeTurbo  : (s, dense_x, R, bits) → QuantizedKV
//                         { q = quantizeAffine(R @ dense_x, …), … }
//        dequantizeTurbo: (s, q, scales, biases, R, bits) → dense_x
//                         { y = dequantizeAffine(…); return y @ R^T }
//  4. Extend the `switch (config.scheme)` blocks in `KVCache.update`
//     (quantize on write) and `KVCache.denseView` (dequantize on read)
//     with one case arm each. SDPA call sites do not change; the cache
//     contract holds.
//
// The same dispatch point is also where a future fused-quant-SDPA Metal
// kernel (see "Fused quant-attention Metal kernel" in TODO.md) would slot
// in: `denseView` becomes a no-op stub for that scheme and SDPA call sites
// grow a parallel quant-path. v1 commits to Path A (dense view + standard
// SDPA).

const std = @import("std");
const mlx = @import("mlx.zig");

/// KV-cache storage scheme.
///   * `off`      — dense bf16 (legacy).
///   * `affine`   — group-wise affine quant via `mlx_quantize`/`mlx_dequantize`.
///   * `turboquant_2` — Hadamard-rotated 2-bit affine quant. Each cache
///                     carries one `[head_dim, head_dim]` rotation matrix per
///                     K and V per layer (see `TurboState`). On write we
///                     compute `q = quantizeAffine(x @ H, group, 2)`; on read
///                     we recover `x ≈ dequantizeAffine(q, …) @ H` (Hadamard
///                     matrices are symmetric and self-inverse modulo a
///                     scalar, so `H = H^T = H^{-1}` after normalization).
///   * `turboquant_4` — same rotation idea at 4-bit. Useful when the bit
///                     budget can spare a couple of bits in exchange for
///                     reduced rotation overhead at the cost ceiling.
///                     (Compared to plain `affine` at 4-bit, TurboQuant 4
///                     spends a `[D,D]` matmul per layer per token; the
///                     rotation breaks the worst-case correlation patterns
///                     that hurt straight affine at long context.)
///
/// 1-bit TurboQuant from the Path B roadmap requires a custom 1-bit
/// pack/unpack — `mlx_quantize`/`mlx_dequantize` only support 2/4/8 bits in
/// mlx 0.31.2. Deferred to a follow-up that pairs with the fused-kernel work.
pub const Scheme = enum { off, affine, turboquant_2, turboquant_4 };

/// Configuration for the cache's storage backend. Stored on `KVCache.config`
/// and switched on at every read/write boundary.
pub const KVQuantConfig = struct {
    scheme: Scheme,
    /// Affine: 4 or 8. TurboQuant: 2 or 4. Ignored when `scheme == .off`.
    bits: u8,
    /// Affine group size — number of consecutive elements that share one
    /// scale+bias pair along the last axis. mlx-c convention is 64 for
    /// 4-bit and 8-bit weights; we match that.
    group_size: u32,

    pub const dense: KVQuantConfig = .{ .scheme = .off, .bits = 0, .group_size = 0 };

    pub fn affine(bits: u8) KVQuantConfig {
        std.debug.assert(bits == 4 or bits == 8);
        return .{ .scheme = .affine, .bits = bits, .group_size = 64 };
    }

    pub fn turboquant(bits: u8) KVQuantConfig {
        // Bits 2 and 4 ride mlx-c's native packing. We intentionally do NOT
        // accept 1 here — adding 1-bit requires a custom pack/unpack on top
        // of the rotation; see Scheme docstring.
        std.debug.assert(bits == 2 or bits == 4);
        return .{
            .scheme = if (bits == 2) .turboquant_2 else .turboquant_4,
            .bits = bits,
            .group_size = 64,
        };
    }

    pub fn isQuant(self: KVQuantConfig) bool {
        return self.scheme != .off;
    }
};

/// A (K, V) pair of `KVQuantConfig`. The cache applies `.k` to keys and `.v`
/// to values independently (llama.cpp `-ctk`/`-ctv` style), so e.g. K can be
/// affine-8 while V is turbo-4 — the quality-safe long-context combo (K errors
/// hurt attention more than V errors). Used as the cache-identity key wherever
/// a single `KVQuantConfig` used to be (cache config, snapshot, prefix-cache
/// match), so warm-reuse never mixes incompatible buffer layouts.
pub const KVQuantPair = struct {
    k: KVQuantConfig,
    v: KVQuantConfig,

    pub const dense: KVQuantPair = .{ .k = KVQuantConfig.dense, .v = KVQuantConfig.dense };

    /// Same config on both sides — the legacy symmetric case.
    pub fn uniform(c: KVQuantConfig) KVQuantPair {
        return .{ .k = c, .v = c };
    }

    /// True when both sides carry the identical config (lets `update` take the
    /// proven symmetric fast-paths instead of the per-side mixed path).
    pub fn isUniform(self: KVQuantPair) bool {
        return std.meta.eql(self.k, self.v);
    }

    /// True when at least one side is quantized (cache needs scale/bias
    /// buffers and, if any side is turbo, a `TurboState`).
    pub fn isQuant(self: KVQuantPair) bool {
        return self.k.isQuant() or self.v.isQuant();
    }

    /// True when at least one side uses a TurboQuant scheme (needs rotation
    /// matrices).
    pub fn anyTurbo(self: KVQuantPair) bool {
        return self.k.scheme == .turboquant_2 or self.k.scheme == .turboquant_4 or
            self.v.scheme == .turboquant_2 or self.v.scheme == .turboquant_4;
    }
};

/// One quantized K or V triple. Layout for input shape `[..., D]`:
///   q      : `[..., D * bits / 32]` uint32   (packed)
///   scales : `[..., D / group_size]` bf16
///   biases : `[..., D / group_size]` bf16
///
/// Owns its three array handles. Caller frees via `deinit`.
pub const QuantizedKV = struct {
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,

    pub fn deinit(self: *QuantizedKV) void {
        _ = mlx.mlx_array_free(self.q);
        _ = mlx.mlx_array_free(self.scales);
        _ = mlx.mlx_array_free(self.biases);
        self.q = mlx.mlx_array_new();
        self.scales = mlx.mlx_array_new();
        self.biases = mlx.mlx_array_new();
    }
};

/// Affine quantize `dense_x` group-wise along the last axis. Returns a
/// `QuantizedKV` triple owned by the caller. Caller's input `dense_x` is
/// not consumed (refcount semantics — the caller still owns it).
pub fn quantizeAffine(
    s: mlx.mlx_stream,
    dense_x: mlx.mlx_array,
    group_size: u32,
    bits: u8,
) !QuantizedKV {
    var vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);

    try mlx.check(mlx.mlx_quantize(
        &vec,
        dense_x,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        .{}, // global_scale (null)
        s,
    ));

    // mlx-c convention: the vector contains [q, scales, biases] in that order.
    // Mirror the unpack pattern used elsewhere when consuming a
    // `mlx_vector_array` (e.g. concatenate fan-out).
    const n = mlx.mlx_vector_array_size(vec);
    if (n != 3) return error.UnexpectedQuantizeOutput;

    var out: QuantizedKV = .{
        .q = mlx.mlx_array_new(),
        .scales = mlx.mlx_array_new(),
        .biases = mlx.mlx_array_new(),
    };
    errdefer out.deinit();

    try mlx.check(mlx.mlx_vector_array_get(&out.q, vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.scales, vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.biases, vec, 2));
    return out;
}

/// Per-cache rotation state for the TurboQuant schemes. Holds one symmetric
/// Hadamard matrix per K and V per layer — `[n, n]` bf16 where `n` is the
/// actual K (resp. V) last-dim observed at first write. Built lazily because
/// the cached K/V last-dim is NOT always `config.head_dim` — Gemma 4 stores
/// K at `2 * head_dim` due to partial-RoPE / split-rotary, and some archs
/// have K and V dims that differ from each other. The matrices are
/// constructed deterministically via Sylvester construction + per-layer
/// column-sign flips (no RNG seed; reproducible across restarts).
///
/// Construction is gated on the last-dim being a power of two. The
/// scheduler's load path validates `head_dim` is pow2 at cache init via
/// `validatePowerOfTwoOrFail`, but the actual K/V dim is allowed to differ
/// by an integer factor (Gemma 4: 2x); we re-check pow2 at lazy-init time
/// and return `error.NonPowerOfTwoHeadDim` if violated.
pub const TurboState = struct {
    /// One matrix per K per layer. Slot may be empty (`.ctx == null`) until
    /// the first `updateTurboQuant` call for that layer, at which point the
    /// real K tensor's last-dim is observed and the matrix is built.
    rk: []mlx.mlx_array,
    /// Per-K-layer dim, recorded at lazy-init time so subsequent calls can
    /// assert shape consistency. 0 = not yet initialized.
    rk_dim: []u32,
    /// One matrix per V per layer. Distinct from rk so K and V get
    /// uncorrelated rotations (matters for arches where Q/K and V share
    /// little structure, e.g. GQA + value-only sliding-window). May also
    /// have a different last-dim from K.
    rv: []mlx.mlx_array,
    rv_dim: []u32,
    allocator: std.mem.Allocator,

    /// Allocate empty slots for `num_layers`. Matrices are NOT built here —
    /// the first `ensureKLayer`/`ensureVLayer` call per layer triggers
    /// construction from the observed K/V tensor shape.
    pub fn initLazy(allocator: std.mem.Allocator, num_layers: u32) !TurboState {
        const rk = try allocator.alloc(mlx.mlx_array, num_layers);
        errdefer allocator.free(rk);
        const rv = try allocator.alloc(mlx.mlx_array, num_layers);
        errdefer allocator.free(rv);
        const rk_dim = try allocator.alloc(u32, num_layers);
        errdefer allocator.free(rk_dim);
        const rv_dim = try allocator.alloc(u32, num_layers);
        errdefer allocator.free(rv_dim);
        for (rk) |*a| a.* = mlx.mlx_array_new();
        for (rv) |*a| a.* = mlx.mlx_array_new();
        for (rk_dim) |*d| d.* = 0;
        for (rv_dim) |*d| d.* = 0;
        return .{
            .rk = rk,
            .rk_dim = rk_dim,
            .rv = rv,
            .rv_dim = rv_dim,
            .allocator = allocator,
        };
    }

    /// Deprecated: kept so existing unit tests (which use a single-layer
    /// fixed-dim setup) keep working. Real load path uses `initLazy` and
    /// builds matrices on first write.
    pub fn initHadamard(allocator: std.mem.Allocator, s: mlx.mlx_stream, num_layers: u32, head_dim: u32) !TurboState {
        if (!std.math.isPowerOfTwo(head_dim)) return error.NonPowerOfTwoHeadDim;
        var state = try initLazy(allocator, num_layers);
        errdefer state.deinit();
        const h_arr = try buildHadamardArray(allocator, s, head_dim);
        defer _ = mlx.mlx_array_free(h_arr);
        for (state.rk, 0..) |*a, i| {
            a.* = try cloneWithSignFlip(s, h_arr, head_dim, @intCast(0x9E37 ^ i));
            state.rk_dim[i] = head_dim;
        }
        for (state.rv, 0..) |*a, i| {
            a.* = try cloneWithSignFlip(s, h_arr, head_dim, @intCast(0x85EB ^ i));
            state.rv_dim[i] = head_dim;
        }
        return state;
    }

    /// Lazy-init the K rotation matrix for `layer` from the observed dim `n`.
    /// Subsequent calls for the same layer assert `n` matches; mismatched
    /// shapes (which would indicate an arch-specific layer-shape change
    /// mid-decode) return `error.TurboShapeMismatch`.
    pub fn ensureKLayer(self: *TurboState, s: mlx.mlx_stream, layer: u32, n: u32) !mlx.mlx_array {
        const li: usize = @intCast(layer);
        if (self.rk_dim[li] != 0) {
            if (self.rk_dim[li] != n) return error.TurboShapeMismatch;
            return self.rk[li];
        }
        if (!std.math.isPowerOfTwo(n)) return error.NonPowerOfTwoHeadDim;
        const h = try buildHadamardArray(self.allocator, s, n);
        defer _ = mlx.mlx_array_free(h);
        self.rk[li] = try cloneWithSignFlip(s, h, n, @as(u64, 0x9E37) ^ @as(u64, li));
        self.rk_dim[li] = n;
        return self.rk[li];
    }

    pub fn ensureVLayer(self: *TurboState, s: mlx.mlx_stream, layer: u32, n: u32) !mlx.mlx_array {
        const li: usize = @intCast(layer);
        if (self.rv_dim[li] != 0) {
            if (self.rv_dim[li] != n) return error.TurboShapeMismatch;
            return self.rv[li];
        }
        if (!std.math.isPowerOfTwo(n)) return error.NonPowerOfTwoHeadDim;
        const h = try buildHadamardArray(self.allocator, s, n);
        defer _ = mlx.mlx_array_free(h);
        self.rv[li] = try cloneWithSignFlip(s, h, n, @as(u64, 0x85EB) ^ @as(u64, li));
        self.rv_dim[li] = n;
        return self.rv[li];
    }

    pub fn deinit(self: *TurboState) void {
        for (self.rk) |*a| _ = mlx.mlx_array_free(a.*);
        for (self.rv) |*a| _ = mlx.mlx_array_free(a.*);
        self.allocator.free(self.rk);
        self.allocator.free(self.rv);
        self.allocator.free(self.rk_dim);
        self.allocator.free(self.rv_dim);
    }
};

/// Deterministic normalized Hadamard matrix `[N, N]` bf16. Sylvester
/// construction: `H_{2N} = [[H_N, H_N], [H_N, -H_N]] / sqrt(2)`. Result is
/// orthogonal (`H^T H = I`) and symmetric (`H^T = H`).
fn buildHadamardArray(allocator: std.mem.Allocator, s: mlx.mlx_stream, n: u32) !mlx.mlx_array {
    const N: usize = @intCast(n);
    const buf = try allocator.alloc(f32, N * N);
    defer allocator.free(buf);
    // Build sign matrix recursively, then normalize at the end. We carry
    // unnormalized ±1 entries through the recursion and divide by sqrt(N) once.
    var size: usize = 1;
    buf[0] = 1.0;
    while (size < N) : (size *= 2) {
        // Quadruple block expansion: top-right = top-left, bottom-left =
        // top-left, bottom-right = -top-left.
        for (0..size) |r| {
            for (0..size) |c| {
                const tl = buf[r * N + c];
                buf[r * N + (c + size)] = tl;
                buf[(r + size) * N + c] = tl;
                buf[(r + size) * N + (c + size)] = -tl;
            }
        }
    }
    const inv_sqrt_n: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(N)));
    for (buf) |*v| v.* *= inv_sqrt_n;
    const shape = [_]c_int{ @intCast(N), @intCast(N) };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
    return bf16_arr;
}

/// Multiply each column of `h_arr` by ±1 according to bits of `seed`. The
/// result is still an orthogonal symmetric matrix (column sign flips of an
/// orthogonal symmetric matrix preserve orthogonality; symmetry preserved
/// because we apply the same flip to row i and column i — actually no, this
/// preserves the absolute values but breaks symmetry unless we also flip
/// row i. We do.). Used to give each layer (and K vs V) its own rotation.
fn cloneWithSignFlip(s: mlx.mlx_stream, h_arr: mlx.mlx_array, n: u32, seed: u64) !mlx.mlx_array {
    const N: usize = @intCast(n);
    // Build a column-sign vector `[N]` with entries ±1 from `seed`.
    var sign_buf: [4096]f32 = undefined;
    if (N > sign_buf.len) return error.HeadDimTooLarge;
    var s_state = seed *% 6364136223846793005;
    for (sign_buf[0..N]) |*v| {
        s_state = s_state *% 6364136223846793005 +% 1442695040888963407;
        v.* = if ((s_state >> 33) & 1 == 0) 1.0 else -1.0;
    }
    const shape_v = [_]c_int{@intCast(N)};
    const sign_f32 = mlx.mlx_array_new_data(&sign_buf, &shape_v, 1, .float32);
    defer _ = mlx.mlx_array_free(sign_f32);
    var sign_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sign_bf16);
    try mlx.check(mlx.mlx_astype(&sign_bf16, sign_f32, .bfloat16, s));

    // R_flipped = diag(sign) @ h @ diag(sign). For Hadamard `h`:
    // multiply rows by sign (broadcast over columns), then multiply columns
    // by sign (broadcast over rows).
    var row_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row_scaled);
    {
        // reshape sign to [N, 1] for row-broadcast
        const sh = [_]c_int{ @intCast(N), 1 };
        var sign_col = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sign_col);
        try mlx.check(mlx.mlx_reshape(&sign_col, sign_bf16, &sh, 2, s));
        try mlx.check(mlx.mlx_multiply(&row_scaled, h_arr, sign_col, s));
    }
    var out = mlx.mlx_array_new();
    {
        const sh = [_]c_int{ 1, @intCast(N) };
        var sign_row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sign_row);
        try mlx.check(mlx.mlx_reshape(&sign_row, sign_bf16, &sh, 2, s));
        try mlx.check(mlx.mlx_multiply(&out, row_scaled, sign_row, s));
    }
    return out;
}

/// Rotate `dense_x` by `R` along its last axis: out = dense_x @ R. Caller
/// owns the returned array. mlx_matmul broadcasts the leading dims of
/// `dense_x` (`[B, H, T, D]`) against `R` (`[D, D]`), so we get
/// `[B, H, T, D]` out.
pub fn rotateLastDim(s: mlx.mlx_stream, dense_x: mlx.mlx_array, R: mlx.mlx_array) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_matmul(&out, dense_x, R, s));
    return out;
}

/// TurboQuant write path: rotate then affine-quantize. The caller's
/// `dense_x` is not consumed.
pub fn quantizeTurbo(
    s: mlx.mlx_stream,
    dense_x: mlx.mlx_array,
    R: mlx.mlx_array,
    group_size: u32,
    bits: u8,
) !QuantizedKV {
    const rotated = try rotateLastDim(s, dense_x, R);
    defer _ = mlx.mlx_array_free(rotated);
    return try quantizeAffine(s, rotated, group_size, bits);
}

/// TurboQuant read path: affine-dequantize then rotate back. Caller owns
/// the returned dense array.
pub fn dequantizeTurbo(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    R: mlx.mlx_array,
    group_size: u32,
    bits: u8,
) !mlx.mlx_array {
    const deq = try dequantizeAffine(s, q, scales, biases, group_size, bits);
    defer _ = mlx.mlx_array_free(deq);
    // `R` is symmetric in our Hadamard+sign-flip construction (we flip rows
    // and columns by the same `sign` vector, so the matrix stays symmetric).
    // Therefore R = R^T = R^{-1}, and the inverse rotation is just `@ R`.
    return try rotateLastDim(s, deq, R);
}

/// Affine dequantize a `(q, scales, biases)` triple to dense bf16. Caller
/// owns the returned array.
pub fn dequantizeAffine(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    group_size: u32,
    bits: u8,
) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_dequantize(
        &out,
        q,
        scales,
        biases,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        .{}, // global_scale (null)
        .{ .value = .bfloat16, .has_value = true },
        s,
    ));
    return out;
}

// ── Fused quant-attention path (opt-in via --kv-attn-mode fused) ──
//
// `quantAttention` reads K and V directly from their quantized triples,
// avoiding the dense materialization that `KVCache.denseView` performs in
// the default path. The fusion comes from `mlx_quantized_matmul`, Apple's
// kernel that internally dequantizes one operand and multiplies in a
// single Metal pass — same primitive `qmatmulBits` already uses for weight
// quantization throughout `transformer.zig`.
//
// Memory shape. For `causal` and `""` (the attention that grows with context)
// this runs a K-TILED online-softmax loop (`tiledCausalAttention`, the manual
// flash-attention-2 interim for the "Fused quant-attention Metal kernel" TODO):
// K/V are walked in `kv_attn_block` chunks, consumed per block via
// quantized_matmul (the dense K/V is NEVER materialized) with a running
// (max, sum, output) in f32 — the full `[T_q, T_k]` scores never form either.
// This tiles the KEY axis only; the per-block scores tile is
// `[B, H_q, T_q, block]`, so the peak is O(T_q · block):
//   * DECODE (T_q==1): O(block) — flat in context. THE WIN; the call sites gate
//     fused to decode for exactly this reason. A 120k decode tick never spikes.
//   * PREFILL (T_q==chunk): O(chunk · block) — multi-GB per layer at chunk=8192,
//     so prefill is NOT routed here; it uses flash SDPA over the (small,
//     per-chunk-eval-bounded) dequantized K/V, which tiles BOTH axes in-kernel.
//     Making fused prefill flat needs a Q-tile loop wrapping this one (each
//     query block is independent — no online-softmax state crosses Q blocks).
// The `array` (explicit/sliding-window) path keeps the single-pass form: the
// surviving context is bounded by the window, and tiling an arbitrary additive
// mask means slicing it per block (no current consumer).
//
// Shape contract:
//   q_dense      : [B, H,    T_q, D] bf16 (Q already scaled or not; we apply scale below)
//   k_q/sc/bi    : K affine-quantized along last axis (D). Shape of the
//                  triple is whatever `mlx_quantize` produced.
//   v_q/sc/bi    : V same as K.
// GQA: H_q > H_kv is handled by FOLDING the `repeats` group into Q's row axis
// (see `quantAttention`) — K/V pass through `mlx_quantized_matmul` untouched at
// their native H_kv, no expansion or copy.

/// A read-only borrow of a `QuantizedKV` triple — the cache owns the
/// arrays; the call site borrows them for the duration of one attention.
/// Used to thread quant triples through `DenseKVView` without altering
/// refcount semantics.
pub const BorrowedTriple = struct {
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
};

/// Process-wide K-tile size for the fused-quant online-softmax attention
/// (`--kv-attn-block`). The per-block transient is H_kv·(repeats·T_q)·block, so
/// this trades flatness vs dispatch count; 4096 is a safe 16 GB default. Bigger
/// = fewer dispatches but a larger (still context-INDEPENDENT) transient — do
/// not pair a large value with a large `--prefill-chunk`.
pub var kv_attn_block: u32 = 4096;

/// Absolute ceiling (MB) on the tiled-attention scores transient before a
/// prefill chunk is refused the tiled path regardless of the tiled-vs-dense
/// comparison (0 = no ceiling). Safety override for `--kv-attn-tiled-budget`.
pub var kv_attn_tiled_budget_mb: u32 = 2048;

/// f32-equivalent copies of the per-block scores tile that coexist within ONE
/// block iteration. Zig `defer`s free at SCOPE (iteration) end, so every
/// `[.., rT_q, blk]` temporary stacks until the loop body closes:
/// s_bf(½)+sf+sc+s_mn+p+p_bf(½) ≈ 5, plus sc5+sc5m+sm on a straddling block ≈ 8.
/// Use the straddle figure as the conservative peak.
const TILED_SCORE_COPIES: u64 = 8;

/// Peak bytes of the K-tiled attention's per-block scores transient:
/// `[B=1, H_q, T_q, min(block, t_k)]` f32 × the coexisting-copy factor. This is
/// the WHOLE-prefill peak — block tiles are freed each iteration and the
/// per-layer carry (acc/l/m, ~T_q·D) is negligible, so layers do NOT stack
/// (unlike the dense dequant path). Flat in t_k beyond one block; grows with
/// T_q, which is why it's cheap on a warm turn (small T_q) and not on a cold
/// chunk (T_q == prefill chunk).
pub fn tiledScoreTransientBytes(h_q: u64, t_q: u64, block_k: u64, t_k: u64) u64 {
    const eff_block = @min(block_k, @max(@as(u64, 1), t_k));
    return TILED_SCORE_COPIES * 4 * h_q * t_q * eff_block;
}

/// Peak bytes of the DENSE path's prefill dequant transient: every
/// full-attention layer materializes the full f16 K AND V to feed flash SDPA,
/// and the async-eval'd chunk graph can hold all of them at once — they STACK,
/// hence the `full_attn_layers` factor. Grows with t_k (the full context), so
/// it's the term that dominates a warm continuation even when only a few tokens
/// are new.
pub fn denseDequantTransientBytes(full_attn_layers: u64, t_k: u64, h_kv: u64, hdim: u64) u64 {
    return full_attn_layers * 2 * t_k * h_kv * hdim * 2;
}

/// Route a PREFILL attention chunk through the K-tiled fused path iff its
/// bounded scores transient is both cheaper than the dense dequant it would
/// replace AND under the absolute ceiling. The discriminator is T_q:
///   * warm continuation (small T_q): tiled tile ≪ dense's O(t_k) dequant → TILED.
///     This is the case the plain `!is_prefill` gate wrongly forced to dense,
///     re-dequantizing the whole context every turn and rejecting multi-turn.
///   * cold prefill (T_q == full chunk): `[H_q, chunk, block]` dwarfs the dense
///     dequant → DENSE (flash SDPA, which tiles both axes in-kernel).
/// Decode (T_q==1) is gated separately at the call sites (always tiled) and does
/// not consult this. Callers must also confirm fused mode + a quantized cache.
pub fn preferTiledPrefill(
    h_q: u64,
    h_kv: u64,
    t_q: u64,
    block_k: u64,
    t_k: u64,
    full_attn_layers: u64,
    hdim: u64,
) bool {
    const tiled = tiledScoreTransientBytes(h_q, t_q, block_k, t_k);
    if (kv_attn_tiled_budget_mb != 0 and
        tiled > @as(u64, kv_attn_tiled_budget_mb) * 1024 * 1024) return false;
    return tiled < denseDequantTransientBytes(full_attn_layers, t_k, h_kv, hdim);
}

/// Slice a `[B, H, T, X]` array to `[B, H, t0:t1, X]` along the sequence axis.
/// Caller owns the result. (The K/V quant triples are already strided cache
/// views; `mlx_quantized_matmul` consumes a further slice fine.)
pub fn sliceSeq(a: mlx.mlx_array, t0: c_int, t1: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    if (sh.len < 4) return error.UnexpectedShape;
    const start = [_]c_int{ 0, 0, t0, 0 };
    const stop = [_]c_int{ sh[0], sh[1], t1, sh[3] };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, a, &start, 4, &stop, 4, &strides, 4, s));
    return out;
}

/// Flash-attention-2 online softmax over K/V blocks for the fused-quant path —
/// the real fix for the O(T_k) attention transient. Walks K/V in `kv_attn_block`
/// chunks, accumulating a running (max `m`, denom `l`, numerator `acc`) in f32,
/// never materializing the full `[.., T_q, T_k]` scores or the full dense K/V.
/// Peak is O(block), CONTEXT-INDEPENDENT. `q_folded` is `[B, H_kv, repeats*T_q,
/// D]` (GQA folded into rows); K/V are read per block straight from the quant
/// triples. `is_causal` applies the right-anchored causal mask per block (and
/// stops at the first fully-future block); false = full attention (decode / no
/// mask). Returns the folded output `[B, H_kv, repeats*T_q, D]` (bf16).
fn tiledCausalAttention(
    q_folded: mlx.mlx_array,
    k_triple: BorrowedTriple,
    v_triple: BorrowedTriple,
    k_bits: u8,
    k_group_size: u32,
    v_bits: u8,
    v_group_size: u32,
    scale: f32,
    B: c_int,
    H_kv: c_int,
    repeats: c_int,
    T_q: c_int,
    t_k: c_int,
    D: c_int,
    is_causal: bool,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const rT_q: c_int = repeats * T_q;
    const block: c_int = @intCast(@max(@as(u32, 1), kv_attn_block));
    const q0: c_int = t_k - T_q; // absolute position of query row 0 (causal)

    const acc_shape = [_]c_int{ B, H_kv, rT_q, D };
    const md_shape = [_]c_int{ B, H_kv, rT_q, 1 };

    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    const zero_arr = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero_arr);
    const scale_arr = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(scale_arr);

    // f32 accumulators (persist across blocks; reassigned in-loop, final freed
    // by these defers).
    var acc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(acc);
    try mlx.check(mlx.mlx_zeros(&acc, &acc_shape, 4, .float32, s));
    var l = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l);
    try mlx.check(mlx.mlx_zeros(&l, &md_shape, 4, .float32, s));
    var m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_full(&m, &md_shape, 4, neg_inf, .float32, s));

    var t0: c_int = 0;
    while (t0 < t_k) : (t0 += block) {
        const t1: c_int = @min(t0 + block, t_k);
        const blk_len: c_int = t1 - t0;
        // Block entirely in the future of EVERY query (causal) → stop (t0 only
        // grows). With K capped at the chunk end this never trips; insurance.
        if (is_causal and t0 > q0 + T_q - 1) break;

        // Per-block K/V slices (freed at iteration end).
        const kq = try sliceSeq(k_triple.q, t0, t1, s);
        defer _ = mlx.mlx_array_free(kq);
        const ksc = try sliceSeq(k_triple.scales, t0, t1, s);
        defer _ = mlx.mlx_array_free(ksc);
        const kbi = try sliceSeq(k_triple.biases, t0, t1, s);
        defer _ = mlx.mlx_array_free(kbi);
        const vq = try sliceSeq(v_triple.q, t0, t1, s);
        defer _ = mlx.mlx_array_free(vq);
        const vsc = try sliceSeq(v_triple.scales, t0, t1, s);
        defer _ = mlx.mlx_array_free(vsc);
        const vbi = try sliceSeq(v_triple.biases, t0, t1, s);
        defer _ = mlx.mlx_array_free(vbi);

        // scores = (Q @ Kblkᵀ) · scale → [B, H_kv, rT_q, blk_len], in f32.
        var s_bf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(s_bf);
        try mlx.check(mlx.mlx_quantized_matmul(&s_bf, q_folded, kq, ksc, kbi, true, mlx.mlx_optional_int.some(@intCast(k_group_size)), mlx.mlx_optional_int.some(@intCast(k_bits)), "affine", s));
        var sf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sf);
        try mlx.check(mlx.mlx_astype(&sf, s_bf, .float32, s));
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        try mlx.check(mlx.mlx_multiply(&sc, sf, scale_arr, s));

        // Causal mask for STRADDLING blocks only. Expose T_q first (the fold
        // interleaves repeats, so a triu over folded rows masks wrong cells);
        // build `[T_q, blk_len]` once, broadcast over B/H_kv/repeats. Visible
        // iff (t0+col) <= (q0+row) ⟺ col-row <= q0-t0; mask the strictly-greater
        // cells via triu at k = (q0-t0)+1.
        var sm = sc;
        var owns_sm = false;
        defer {
            if (owns_sm) _ = mlx.mlx_array_free(sm);
        }
        if (is_causal and (t1 - 1 > q0)) {
            var sc5 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sc5);
            {
                const sh = [_]c_int{ B, H_kv, repeats, T_q, blk_len };
                try mlx.check(mlx.mlx_reshape(&sc5, sc, &sh, 5, s));
            }
            var ones2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ones2);
            const sh2 = [_]c_int{ T_q, blk_len };
            try mlx.check(mlx.mlx_ones(&ones2, &sh2, 2, .float32, s));
            var upper = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(upper);
            try mlx.check(mlx.mlx_triu(&upper, ones2, (q0 - t0) + 1, s));
            var mask2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(mask2);
            try mlx.check(mlx.mlx_multiply(&mask2, upper, neg_inf, s));
            var sc5m = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sc5m);
            try mlx.check(mlx.mlx_add(&sc5m, sc5, mask2, s));
            sm = mlx.mlx_array_new();
            owns_sm = true;
            const sh4 = [_]c_int{ B, H_kv, rT_q, blk_len };
            try mlx.check(mlx.mlx_reshape(&sm, sc5m, &sh4, 4, s));
        }

        // Online-softmax update (flash-2). m←new max last; c rescales carried
        // l/acc; p is the block at the new max. c guards a row still at -inf
        // (no key seen yet) → 0, avoiding exp(-inf − -inf)=NaN.
        var mb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mb);
        try mlx.check(mlx.mlx_max_axis(&mb, sm, -1, true, s));
        var mn = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_maximum(&mn, m, mb, s));

        var m_eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(m_eq);
        try mlx.check(mlx.mlx_equal(&m_eq, m, neg_inf, s));
        var m_mn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(m_mn);
        try mlx.check(mlx.mlx_subtract(&m_mn, m, mn, s));
        var exp_mm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(exp_mm);
        try mlx.check(mlx.mlx_exp(&exp_mm, m_mn, s));
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_where(&c, m_eq, zero_arr, exp_mm, s));

        var s_mn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(s_mn);
        try mlx.check(mlx.mlx_subtract(&s_mn, sm, mn, s));
        var p = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(p);
        try mlx.check(mlx.mlx_exp(&p, s_mn, s));

        // Commit m now (c and p captured what they needed from the old m / mn).
        _ = mlx.mlx_array_free(m);
        m = mn;

        // l = l·c + rowsum(p)
        var lc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lc);
        try mlx.check(mlx.mlx_multiply(&lc, l, c, s));
        var psum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(psum);
        try mlx.check(mlx.mlx_sum_axis(&psum, p, -1, true, s));
        var l_new = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&l_new, lc, psum, s));
        _ = mlx.mlx_array_free(l);
        l = l_new;

        // acc = acc·c + (p @ Vblk).  P→bf16 for the V matmul (matches the
        // existing fused V-matmul precision); accumulate in f32.
        var p_bf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(p_bf);
        try mlx.check(mlx.mlx_astype(&p_bf, p, .bfloat16, s));
        var pv = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pv);
        try mlx.check(mlx.mlx_quantized_matmul(&pv, p_bf, vq, vsc, vbi, false, mlx.mlx_optional_int.some(@intCast(v_group_size)), mlx.mlx_optional_int.some(@intCast(v_bits)), "affine", s));
        var pv_f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pv_f);
        try mlx.check(mlx.mlx_astype(&pv_f, pv, .float32, s));
        var acc_c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(acc_c);
        try mlx.check(mlx.mlx_multiply(&acc_c, acc, c, s));
        var acc_new = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&acc_new, acc_c, pv_f, s));
        _ = mlx.mlx_array_free(acc);
        acc = acc_new;
    }

    // Finalize: out = acc / max(l, eps). The eps guards any row that never saw
    // an unmasked key (can't happen in causal with K capped, but cheap).
    const eps = mlx.mlx_array_new_float(1e-9);
    defer _ = mlx.mlx_array_free(eps);
    var l_safe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l_safe);
    try mlx.check(mlx.mlx_maximum(&l_safe, l, eps, s));
    var out_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_f);
    try mlx.check(mlx.mlx_divide(&out_f, acc, l_safe, s));
    var out_folded = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out_folded);
    try mlx.check(mlx.mlx_astype(&out_folded, out_f, .bfloat16, s));
    return out_folded;
}

/// Hand-rolled attention that consumes K and V triples directly.
/// `scale` is the standard 1/sqrt(D) factor SDPA applies before softmax.
/// `mask_mode` mirrors `mlx_fast_scaled_dot_product_attention`:
///   * "causal": apply a Q-aligned causal mask (upper triangular -inf).
///     For T_q == 1 (decode tick) this is a no-op.
///   * "":       no mask.
///   * "array":  add `mask_arr` to the pre-softmax scores. Must be
///               additive (mlx convention: -inf for masked positions).
///
/// `rk`/`rv` are the optional per-side Hadamard rotation matrices for the
/// TurboQuant schemes (fused-turbo path). When K is turbo, its triple holds
/// `quantize(K @ Rk)`, and the score `Q·Kᵀ` is rotation-invariant, so we
/// rotate `Q` by `Rk` before the K matmul: `(Q@Rk)·(K@Rk) = Q·K`. When V is
/// turbo, its triple holds `quantize(V @ Rv)`, so the fused product
/// `P @ dequant(qV) = (P@V) @ Rv`; we undo it by rotating the output through
/// `Rvᵀ` (= `Rv`, since the matrices are symmetric+orthogonal). Pass
/// `.{ .ctx = null }` for either when that side is plain affine (no rotation).
/// Returns a `[B, H, T_q, D]` bf16 array; caller owns and frees.
pub fn quantAttention(
    q_dense: mlx.mlx_array,
    k_triple: BorrowedTriple,
    v_triple: BorrowedTriple,
    k_bits: u8,
    k_group_size: u32,
    v_bits: u8,
    v_group_size: u32,
    rk: mlx.mlx_array,
    rv: mlx.mlx_array,
    scale: f32,
    mask_mode: []const u8,
    mask_arr: mlx.mlx_array,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    // GQA WITHOUT materializing K/V. `mlx_quantized_matmul` doesn't broadcast
    // across the head dim the way fast SDPA does, so H_q > H_kv (Gemma 4: 8/2,
    // Qwen 3.5: 16/4) must be reconciled. The previous version did this with
    // `mlx_repeat_axis` on all SIX K/V quant components — but `repeat` PHYSICALLY
    // copies (it is NOT the stride-0 view the old comment claimed; that's
    // `broadcast_to`), so at long context it duplicated the entire quantized
    // K/V cache `repeats`× per layer per forward — a multi-GB transient that
    // OOM-aborted prefill and defeated the whole point of the fused path.
    //
    // Instead, fold the `repeats` group into Q's ROW axis so K/V stay at their
    // native H_kv and pass through `quantized_matmul` untouched (no copy, no
    // reliance on quant-matmul broadcast semantics). Q [B, H_q, T_q, D] ->
    // [B, H_kv, repeats*T_q, D] is a zero-copy contiguous reshape: q-head
    // `h_kv*repeats + j` lands at row `j*T_q + t_q` of kv-head `h_kv`, so each
    // q-head still dots against its own kv-head. Scores/probs are reshaped to
    // expose the real T_q axis for masking, then folded back for the V matmul,
    // and the output is reshaped back to [B, H_q, T_q, D]. All reshapes are
    // contiguous (zero-copy).
    const q_shape = mlx.getShape(q_dense);
    if (q_shape.len < 4) return error.UnexpectedQShape;
    const k_q_shape = mlx.getShape(k_triple.q);
    if (k_q_shape.len < 4) return error.UnexpectedKShape;
    const B: c_int = q_shape[0];
    const H_q: c_int = q_shape[1];
    const T_q: c_int = q_shape[2];
    const D: c_int = q_shape[3];
    const H_kv: c_int = k_q_shape[1];
    const repeats: c_int = @divExact(H_q, H_kv);

    // 0) fused-turbo: rotate Q into K's stored (Hadamard) basis. No-op when
    //    `rk.ctx == null` (affine K). `rk` is `[D, D]`, broadcast over heads.
    var q_rot = q_dense;
    var owns_q_rot = false;
    defer {
        if (owns_q_rot) _ = mlx.mlx_array_free(q_rot);
    }
    if (rk.ctx != null) {
        q_rot = try rotateLastDim(s, q_dense, rk);
        owns_q_rot = true;
    }

    // Fold the head group into Q's rows (zero-copy).
    var q_folded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_folded);
    {
        const sh = [_]c_int{ B, H_kv, repeats * T_q, D };
        try mlx.check(mlx.mlx_reshape(&q_folded, q_rot, &sh, 4, s));
    }

    // K length (full cache) from the K triple's sequence axis.
    const t_k: c_int = blk: {
        const ks = mlx.getShape(k_triple.q);
        if (ks.len < 3) return error.UnexpectedKShape;
        break :blk ks[2];
    };

    var out_folded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_folded);

    if (std.mem.eql(u8, mask_mode, "array")) {
        // Explicit (sliding-window) mask: the surviving context is bounded by
        // the window, so the single-pass O(T_k) form is fine — and tiling an
        // arbitrary additive mask would mean slicing it per block (extra
        // surface, no current consumer). Keep the single-pass path.
        var scores = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores);
        try mlx.check(mlx.mlx_quantized_matmul(&scores, q_folded, k_triple.q, k_triple.scales, k_triple.biases, true, mlx.mlx_optional_int.some(@intCast(k_group_size)), mlx.mlx_optional_int.some(@intCast(k_bits)), "affine", s));
        const scale_arr = mlx.mlx_array_new_float(scale);
        defer _ = mlx.mlx_array_free(scale_arr);
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_multiply(&scaled, scores, scale_arr, s));
        var scaled5 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled5);
        {
            const sh = [_]c_int{ B, H_kv, repeats, T_q, t_k };
            try mlx.check(mlx.mlx_reshape(&scaled5, scaled, &sh, 5, s));
        }
        // Position mask (e.g. sliding window), shape `[1,1,T_q,T_k]` /
        // `[1,1,1,T_k]` — broadcasts over the leading B/H_kv/repeats dims.
        var pre = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pre);
        try mlx.check(mlx.mlx_add(&pre, scaled5, mask_arr, s));
        var attn5 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn5);
        try mlx.check(mlx.mlx_softmax_axis(&attn5, pre, -1, true, s));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        {
            const sh = [_]c_int{ B, H_kv, repeats * T_q, t_k };
            try mlx.check(mlx.mlx_reshape(&attn, attn5, &sh, 4, s));
        }
        try mlx.check(mlx.mlx_quantized_matmul(&out_folded, attn, v_triple.q, v_triple.scales, v_triple.biases, false, mlx.mlx_optional_int.some(@intCast(v_group_size)), mlx.mlx_optional_int.some(@intCast(v_bits)), "affine", s));
    } else {
        // causal or "" (no mask): K-TILED online softmax. Never materializes
        // the full scores or dense KV; peak is O(T_q · block) — O(block) and
        // context-flat at DECODE (T_q==1, the gated use), O(chunk · block) at
        // prefill (why the call sites keep prefill on flash SDPA — see header).
        const is_causal = std.mem.eql(u8, mask_mode, "causal");
        _ = mlx.mlx_array_free(out_folded);
        out_folded = try tiledCausalAttention(q_folded, k_triple, v_triple, k_bits, k_group_size, v_bits, v_group_size, scale, B, H_kv, repeats, T_q, t_k, D, is_causal, s);
    }

    // 8) un-fold the head group: [B, H_kv, repeats*T_q, D] -> [B, H_q, T_q, D]
    //    (zero-copy; reverses the step-0 fold).
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    {
        const sh = [_]c_int{ B, H_q, T_q, D };
        try mlx.check(mlx.mlx_reshape(&out, out_folded, &sh, 4, s));
    }

    // 9) fused-turbo: undo V's stored rotation. `out = (P@V) @ Rv`, so rotating
    //    by `Rv` (= `Rvᵀ`) recovers `P@V`. No-op when `rv.ctx == null`.
    if (rv.ctx != null) {
        const out_rot = try rotateLastDim(s, out, rv);
        _ = mlx.mlx_array_free(out);
        return out_rot;
    }
    return out;
}

// ── Tests ──

const testing = std.testing;

/// Build a `[1, 1, 8, head_dim]` bf16 tensor whose values vary smoothly per
/// position so quantization error is non-trivial but bounded.
fn buildSmoothBf16(s: mlx.mlx_stream, head_dim: c_int) !mlx.mlx_array {
    const T: usize = 8;
    const D: usize = @intCast(head_dim);
    const buf = try testing.allocator.alloc(f32, T * D);
    defer testing.allocator.free(buf);
    for (0..T) |t| {
        for (0..D) |d| {
            // Range roughly [-1, 1]; smooth so adjacent group elements are
            // close (best case for affine), with some variation across groups.
            const fi: f32 = @floatFromInt(t * D + d);
            const denom: f32 = @floatFromInt(T * D);
            buf[t * D + d] = (fi / denom) * 2.0 - 1.0;
        }
    }
    const shape = [_]c_int{ 1, 1, @intCast(T), head_dim };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
    return bf16_arr;
}

/// Read a flat float32 host buffer for a small array (eval-and-copy via
/// `mlx_astype` to float32 then reshape to 1D).
pub fn readF32Flat(s: mlx.mlx_stream, arr: mlx.mlx_array, allocator: std.mem.Allocator) ![]f32 {
    var f32_view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32_view);
    try mlx.check(mlx.mlx_astype(&f32_view, arr, .float32, s));

    const n: c_int = @intCast(mlx.mlx_array_size(f32_view));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    {
        const sh = [_]c_int{n};
        try mlx.check(mlx.mlx_reshape(&flat, f32_view, &sh, 1, s));
    }
    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, flat);
        try mlx.check(mlx.mlx_eval(ev));
    }
    const ptr = mlx.mlx_array_data_float32(flat) orelse return error.NullData;
    const out = try allocator.alloc(f32, @intCast(n));
    @memcpy(out, ptr[0..@intCast(n)]);
    return out;
}

test "quantizeAffine + dequantizeAffine round-trip at 4 bits" {
    const s = mlx.gpuStream();
    const src = try buildSmoothBf16(s, 256);
    defer _ = mlx.mlx_array_free(src);

    var qkv = try quantizeAffine(s, src, 64, 4);
    defer qkv.deinit();

    // Shape sanity: q is [..., D * bits / 32] = [..., 256 * 4 / 32] = [..., 32]
    const q_shape = mlx.getShape(qkv.q);
    try testing.expectEqual(@as(c_int, 32), q_shape[q_shape.len - 1]);
    // scales/biases last dim = D / group_size = 256 / 64 = 4
    const sc_shape = mlx.getShape(qkv.scales);
    try testing.expectEqual(@as(c_int, 4), sc_shape[sc_shape.len - 1]);

    const deq = try dequantizeAffine(s, qkv.q, qkv.scales, qkv.biases, 64, 4);
    defer _ = mlx.mlx_array_free(deq);

    const orig = try readF32Flat(s, src, testing.allocator);
    defer testing.allocator.free(orig);
    const got = try readF32Flat(s, deq, testing.allocator);
    defer testing.allocator.free(got);

    try testing.expectEqual(orig.len, got.len);
    var max_err: f32 = 0;
    for (orig, got) |o, g| {
        const e = @abs(o - g);
        if (e > max_err) max_err = e;
    }
    // 4-bit affine on smooth data with group=64: empirical ceiling well under 0.05.
    try testing.expect(max_err < 0.05);
}

test "quantizeAffine + dequantizeAffine round-trip at 8 bits" {
    const s = mlx.gpuStream();
    const src = try buildSmoothBf16(s, 256);
    defer _ = mlx.mlx_array_free(src);

    var qkv = try quantizeAffine(s, src, 64, 8);
    defer qkv.deinit();

    // Shape: q = [..., 256 * 8 / 32] = [..., 64]
    const q_shape = mlx.getShape(qkv.q);
    try testing.expectEqual(@as(c_int, 64), q_shape[q_shape.len - 1]);

    const deq = try dequantizeAffine(s, qkv.q, qkv.scales, qkv.biases, 64, 8);
    defer _ = mlx.mlx_array_free(deq);

    const orig = try readF32Flat(s, src, testing.allocator);
    defer testing.allocator.free(orig);
    const got = try readF32Flat(s, deq, testing.allocator);
    defer testing.allocator.free(got);

    var max_err: f32 = 0;
    for (orig, got) |o, g| {
        const e = @abs(o - g);
        if (e > max_err) max_err = e;
    }
    // 8-bit affine: ~256x finer steps than 4-bit; expect < 0.005 on smooth data.
    try testing.expect(max_err < 0.01);
}

test "KVQuantConfig.affine builds a sane config" {
    const c4 = KVQuantConfig.affine(4);
    try testing.expectEqual(Scheme.affine, c4.scheme);
    try testing.expectEqual(@as(u8, 4), c4.bits);
    try testing.expectEqual(@as(u32, 64), c4.group_size);

    const c8 = KVQuantConfig.affine(8);
    try testing.expectEqual(@as(u8, 8), c8.bits);

    const cd = KVQuantConfig.dense;
    try testing.expectEqual(Scheme.off, cd.scheme);
}

test "KVQuantConfig.turboquant routes 2 and 4 to distinct schemes" {
    const t2 = KVQuantConfig.turboquant(2);
    try testing.expectEqual(Scheme.turboquant_2, t2.scheme);
    try testing.expectEqual(@as(u8, 2), t2.bits);
    try testing.expectEqual(@as(u32, 64), t2.group_size);

    const t4 = KVQuantConfig.turboquant(4);
    try testing.expectEqual(Scheme.turboquant_4, t4.scheme);
    try testing.expectEqual(@as(u8, 4), t4.bits);
}

test "buildHadamardArray produces a valid Hadamard at N=8" {
    const s = mlx.gpuStream();
    const h = try buildHadamardArray(testing.allocator, s, 8);
    defer _ = mlx.mlx_array_free(h);

    // H is `[8, 8]` and H @ H = I (entries already normalized by 1/sqrt(8)).
    var hh = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hh);
    try mlx.check(mlx.mlx_matmul(&hh, h, h, s));
    const got = try readF32Flat(s, hh, testing.allocator);
    defer testing.allocator.free(got);
    try testing.expectEqual(@as(usize, 64), got.len);
    // Diagonal ~1, off-diagonal ~0. Use a loose tolerance because we cast
    // to bf16 and back; rounding error is non-trivial.
    for (0..8) |r| {
        for (0..8) |c| {
            const expected: f32 = if (r == c) 1.0 else 0.0;
            try testing.expect(@abs(got[r * 8 + c] - expected) < 0.05);
        }
    }
}

test "quantizeTurbo + dequantizeTurbo round-trip at 4 bits with Hadamard" {
    const s = mlx.gpuStream();
    const src = try buildSmoothBf16(s, 256);
    defer _ = mlx.mlx_array_free(src);

    // Build a single Hadamard matrix and use it as R.
    var ts = try TurboState.initHadamard(testing.allocator, s, 1, 256);
    defer ts.deinit();
    const R = ts.rk[0];

    var qkv = try quantizeTurbo(s, src, R, 64, 4);
    defer qkv.deinit();

    const deq = try dequantizeTurbo(s, qkv.q, qkv.scales, qkv.biases, R, 64, 4);
    defer _ = mlx.mlx_array_free(deq);

    const orig = try readF32Flat(s, src, testing.allocator);
    defer testing.allocator.free(orig);
    const got = try readF32Flat(s, deq, testing.allocator);
    defer testing.allocator.free(got);
    try testing.expectEqual(orig.len, got.len);

    // Smoke check: no NaN/Inf in the dequantized output. The smooth ramp
    // is actually a worst-case input for TurboQuant (the rotation mixes a
    // tight local range into a wider global one, increasing per-group step
    // size). We only assert finiteness + that the mean absolute error
    // stays sub-input-range — that catches kernel bugs (wrong matmul
    // shape, NaN propagation, packing off-by-one) without overspecifying.
    var sum_abs: f64 = 0;
    var input_range: f32 = 0;
    for (orig, got) |o, g| {
        try testing.expect(std.math.isFinite(g));
        sum_abs += @abs(o - g);
        input_range = @max(input_range, @abs(o));
    }
    const mean_err: f32 = @floatCast(sum_abs / @as(f64, @floatFromInt(orig.len)));
    // Mean error < input range (i.e. the output bears resemblance to the input).
    try testing.expect(mean_err < input_range);
}

test "quantizeTurbo + dequantizeTurbo round-trip at 2 bits" {
    const s = mlx.gpuStream();
    const src = try buildSmoothBf16(s, 256);
    defer _ = mlx.mlx_array_free(src);

    var ts = try TurboState.initHadamard(testing.allocator, s, 1, 256);
    defer ts.deinit();
    const R = ts.rk[0];

    var qkv = try quantizeTurbo(s, src, R, 64, 2);
    defer qkv.deinit();

    const deq = try dequantizeTurbo(s, qkv.q, qkv.scales, qkv.biases, R, 64, 2);
    defer _ = mlx.mlx_array_free(deq);

    const orig = try readF32Flat(s, src, testing.allocator);
    defer testing.allocator.free(orig);
    const got = try readF32Flat(s, deq, testing.allocator);
    defer testing.allocator.free(got);

    // 2-bit on a smooth-ramp input is heavily lossy — only 4 quant levels
    // per group. We just want finiteness + bounded values; per-element
    // tightness is meaningless on this input shape.
    var input_range: f32 = 0;
    for (orig, got) |o, g| {
        try testing.expect(std.math.isFinite(g));
        input_range = @max(input_range, @abs(o));
        // Output should stay within ~2x the input range — a generous bound
        // that rejects NaN, runaway, or wrong-bias accumulation.
        try testing.expect(@abs(g) < 2.0 * input_range + 1.0);
    }
}

test "TurboState.initHadamard rejects non-power-of-2 head_dim" {
    const s = mlx.gpuStream();
    const result = TurboState.initHadamard(testing.allocator, s, 1, 96);
    try testing.expectError(error.NonPowerOfTwoHeadDim, result);
}

// ── Fused-attention validation ──
//
// The two tests below are the validation harness Phase 2 v1 relies on
// before wiring `quantAttention` into transformer.zig SDPA call sites.
// They prove (a) `mlx_quantized_matmul` semantics match dequant+matmul
// in both transpose modes, and (b) the assembled `quantAttention`
// produces logits within the same loose tolerance (0.05 max-abs-diff)
// the existing affine round-trip tests use.

/// Build a `[B, H, T, D]` dense bf16 with smooth ramped values per
/// `(b, h, t, d)` so quantization is non-trivial. Caller frees.
pub fn buildSmoothBHTD(s: mlx.mlx_stream, B: c_int, H: c_int, T: c_int, D: c_int) !mlx.mlx_array {
    const total: usize = @intCast(B * H * T * D);
    const buf = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(buf);
    var i: usize = 0;
    for (0..@intCast(B)) |b| {
        for (0..@intCast(H)) |h| {
            for (0..@intCast(T)) |t| {
                for (0..@intCast(D)) |d| {
                    const f: f32 = @floatFromInt(b * 17 + h * 5 + t * 3 + d);
                    const denom: f32 = @floatFromInt(total);
                    buf[i] = (f / denom) * 2.0 - 1.0;
                    i += 1;
                }
            }
        }
    }
    const shape = [_]c_int{ B, H, T, D };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
    return bf16_arr;
}

test "mlx_quantized_matmul transpose=true matches dequant+matmul (4-bit, 4D)" {
    const s = mlx.gpuStream();
    // x: [1, 2, 3, 64]  (Q-shaped: B, H, T_q, D)
    // w: [1, 2, 5, 64]  (K-shaped: B, H_kv, T_k, D, will be quantized along D)
    // Expected: x @ w.T → [1, 2, 3, 5]
    const x = try buildSmoothBHTD(s, 1, 2, 3, 64);
    defer _ = mlx.mlx_array_free(x);
    const w = try buildSmoothBHTD(s, 1, 2, 5, 64);
    defer _ = mlx.mlx_array_free(w);

    var qw = try quantizeAffine(s, w, 64, 4);
    defer qw.deinit();

    // Reference: dequantize w, then dense matmul with explicit transpose.
    const w_deq = try dequantizeAffine(s, qw.q, qw.scales, qw.biases, 64, 4);
    defer _ = mlx.mlx_array_free(w_deq);
    // Transpose w_deq along last two dims: [1,2,5,64] → [1,2,64,5]
    var w_deq_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_deq_t);
    const axes_t = [_]c_int{ 0, 1, 3, 2 };
    try mlx.check(mlx.mlx_transpose_axes(&w_deq_t, w_deq, &axes_t, 4, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_matmul(&ref, x, w_deq_t, s));

    // Candidate: fused qmm with transpose=true.
    var cand = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand);
    try mlx.check(mlx.mlx_quantized_matmul(
        &cand,
        x,
        qw.q,
        qw.scales,
        qw.biases,
        true,
        mlx.mlx_optional_int.some(64),
        mlx.mlx_optional_int.some(4),
        "affine",
        s,
    ));

    // Shape sanity.
    const ref_shape = mlx.getShape(ref);
    const cand_shape = mlx.getShape(cand);
    try testing.expectEqual(ref_shape.len, cand_shape.len);
    for (ref_shape, cand_shape) |r, c| try testing.expectEqual(r, c);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    var max_err: f32 = 0;
    var max_ref: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
        if (@abs(r) > max_ref) max_ref = @abs(r);
    }
    // bf16 reductions inside qmm don't bit-match an explicit dequant +
    // mlx_matmul because the order of additions differs. Compare in
    // relative terms (the smooth-ramp test produces outputs in the
    // [-N*D*1.0, N*D*1.0] range so absolute thresholds are misleading).
    const rel_err = if (max_ref > 0) max_err / max_ref else max_err;
    if (rel_err >= 0.02) {
        std.debug.print("max_err={d} max_ref={d} rel_err={d}\n", .{ max_err, max_ref, rel_err });
    }
    try testing.expect(rel_err < 0.02);
}

test "mlx_quantized_matmul transpose=false matches dequant+matmul (4-bit, 4D)" {
    const s = mlx.gpuStream();
    // x: [1, 2, 3, 5]  (attn-shaped: B, H, T_q, T_k)
    // w: [1, 2, 5, 64] (V-shaped:    B, H_kv, T_k, D)
    // Expected: x @ w → [1, 2, 3, 64], contracting over T_k.
    // For V, the quantized last axis is D — so transpose=false should
    // dequantize each D-column on the fly while contracting over T_k.
    const x = try buildSmoothBHTD(s, 1, 2, 3, 5);
    defer _ = mlx.mlx_array_free(x);
    const w = try buildSmoothBHTD(s, 1, 2, 5, 64);
    defer _ = mlx.mlx_array_free(w);

    var qw = try quantizeAffine(s, w, 64, 4);
    defer qw.deinit();

    // Reference: dequant + plain matmul.
    const w_deq = try dequantizeAffine(s, qw.q, qw.scales, qw.biases, 64, 4);
    defer _ = mlx.mlx_array_free(w_deq);
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_matmul(&ref, x, w_deq, s));

    var cand = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand);
    try mlx.check(mlx.mlx_quantized_matmul(
        &cand,
        x,
        qw.q,
        qw.scales,
        qw.biases,
        false,
        mlx.mlx_optional_int.some(64),
        mlx.mlx_optional_int.some(4),
        "affine",
        s,
    ));

    const ref_shape = mlx.getShape(ref);
    const cand_shape = mlx.getShape(cand);
    try testing.expectEqual(ref_shape.len, cand_shape.len);
    for (ref_shape, cand_shape) |r, c| try testing.expectEqual(r, c);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

test "quantAttention matches dense SDPA at 4-bit (decode, T_q=1)" {
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const H: c_int = 2;
    const T_k: c_int = 8;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H, 1, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H, T_k, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H, T_k, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var qk = try quantizeAffine(s, k_dense, 64, 4);
    defer qk.deinit();
    var qv = try quantizeAffine(s, v_dense, 64, 4);
    defer qv.deinit();

    // Dense reference. We pass the dequantized K/V so any mismatch
    // attributable to quantization itself is folded into both paths.
    const k_ref = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeAffine(s, qv.q, qv.scales, qv.biases, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref,
        q,
        k_ref,
        v_ref,
        scale,
        "",
        none_mask,
        .{ .ctx = null },
        s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        4,
        64,
        .{ .ctx = null },
        .{ .ctx = null },
        scale,
        "",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    // Matches the bf16 reduction tolerance used elsewhere in this file.
    try testing.expect(max_err < 0.05);
}

test "quantAttention causal mask matches dense SDPA (prefill, T_q=T_k=4)" {
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const H: c_int = 2;
    const T: c_int = 4;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var qk = try quantizeAffine(s, k_dense, 64, 4);
    defer qk.deinit();
    var qv = try quantizeAffine(s, v_dense, 64, 4);
    defer qv.deinit();

    const k_ref = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeAffine(s, qv.q, qv.scales, qv.biases, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref,
        q,
        k_ref,
        v_ref,
        scale,
        "causal",
        none_mask,
        .{ .ctx = null },
        s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        4,
        64,
        .{ .ctx = null },
        .{ .ctx = null },
        scale,
        "causal",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

// ── Fused-turbo validation (Feature 1) ──
//
// These two tests are the correctness gate for fused-turbo. They store K/V
// exactly as the cache's turbo write path does (`quantize(K@Rk)` /
// `quantize(V@Rv)` with DISTINCT Rk, Rv) and assert that `quantAttention`
// fed the rotated triples + Rk/Rv reproduces the known-good dense-turbo path
// (dequantize+un-rotate → `mlx_fast_scaled_dot_product_attention`). The
// rotations are orthogonal/exact, so any error beyond the bf16-reduction
// tolerance means a transpose/orientation bug in the Q-in or output rotation —
// precisely the silent-garbage failure mode the handover warns about.

test "fused-turbo quantAttention matches dense-turbo SDPA (decode, T_q=1)" {
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const H: c_int = 2;
    const T_k: c_int = 8;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H, 1, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H, T_k, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H, T_k, D);
    defer _ = mlx.mlx_array_free(v_dense);

    // Distinct K and V rotations — the fused-turbo correctness hinge (Rk != Rv).
    var ts = try TurboState.initHadamard(testing.allocator, s, 1, @intCast(D));
    defer ts.deinit();
    const rk = ts.rk[0];
    const rv = ts.rv[0];

    var qk = try quantizeTurbo(s, k_dense, rk, 64, 4);
    defer qk.deinit();
    var qv = try quantizeTurbo(s, v_dense, rv, 64, 4);
    defer qv.deinit();

    const k_ref = try dequantizeTurbo(s, qk.q, qk.scales, qk.biases, rk, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeTurbo(s, qv.q, qv.scales, qv.biases, rv, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref, q, k_ref, v_ref, scale, "", none_mask, .{ .ctx = null }, s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        4,
        64,
        rk,
        rv,
        scale,
        "",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

test "fused-turbo quantAttention matches dense-turbo SDPA (causal, Rk != Rv)" {
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const H: c_int = 2;
    const T: c_int = 4;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var ts = try TurboState.initHadamard(testing.allocator, s, 1, @intCast(D));
    defer ts.deinit();
    const rk = ts.rk[0];
    const rv = ts.rv[0];

    var qk = try quantizeTurbo(s, k_dense, rk, 64, 4);
    defer qk.deinit();
    var qv = try quantizeTurbo(s, v_dense, rv, 64, 4);
    defer qv.deinit();

    const k_ref = try dequantizeTurbo(s, qk.q, qk.scales, qk.biases, rk, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeTurbo(s, qv.q, qv.scales, qv.biases, rv, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref, q, k_ref, v_ref, scale, "causal", none_mask, .{ .ctx = null }, s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        4,
        64,
        rk,
        rv,
        scale,
        "causal",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

// ── Asymmetric K/V validation (Feature 2) ──
//
// Exercises the per-side bits/group split AND per-side rotation in a single
// call: K stored as plain affine-8 (no rotation, rk == null), V stored as
// turbo-4 (Hadamard-rotated, rv set). This is the handover's quality-safe
// long-context combo (K q8 + V turbo4). Reference is dense SDPA over the
// matching per-side dequantizations.
test "asymmetric quantAttention: K affine-8 + V turbo-4 matches dense SDPA" {
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const H: c_int = 2;
    const T: c_int = 4;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(v_dense);

    // V rotation only — K is plain affine, so rk stays null.
    var ts = try TurboState.initHadamard(testing.allocator, s, 1, @intCast(D));
    defer ts.deinit();
    const rv = ts.rv[0];

    var qk = try quantizeAffine(s, k_dense, 64, 8);
    defer qk.deinit();
    var qv = try quantizeTurbo(s, v_dense, rv, 64, 4);
    defer qv.deinit();

    const k_ref = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, 8);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeTurbo(s, qv.q, qv.scales, qv.biases, rv, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref, q, k_ref, v_ref, scale, "causal", none_mask, .{ .ctx = null }, s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        8, // k_bits  — affine-8 K
        64, // k_group_size
        4, // v_bits  — turbo-4 V
        64, // v_group_size
        .{ .ctx = null }, // rk: K is affine, no rotation
        rv, // rv: V is turbo
        scale,
        "causal",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

// ── GQA validation (H_q > H_kv) ──
//
// The other quantAttention tests all use H_q == H_kv, so they never exercise
// the GQA head-folding path — the one that used to mlx_repeat the whole K/V
// cache and OOM at long context. These run H_q=4 / H_kv=2 (repeats=2) and
// assert the folded path still matches dense SDPA (which broadcasts GQA
// natively), proving q-head i correctly dots against kv-head i/repeats.

test "quantAttention GQA (H_q=4,H_kv=2) affine matches dense SDPA (causal)" {
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const H_q: c_int = 4;
    const H_kv: c_int = 2;
    const T: c_int = 4;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H_q, T, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H_kv, T, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H_kv, T, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var qk = try quantizeAffine(s, k_dense, 64, 4);
    defer qk.deinit();
    var qv = try quantizeAffine(s, v_dense, 64, 4);
    defer qv.deinit();

    const k_ref = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeAffine(s, qv.q, qv.scales, qv.biases, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    // fast SDPA broadcasts GQA (H_q/H_kv) natively → the reference grouping.
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref, q, k_ref, v_ref, scale, "causal", none_mask, .{ .ctx = null }, s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        4,
        64,
        .{ .ctx = null },
        .{ .ctx = null },
        scale,
        "causal",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

test "quantAttention GQA (H_q=4,H_kv=2) fused-turbo matches dense-turbo SDPA (causal)" {
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const H_q: c_int = 4;
    const H_kv: c_int = 2;
    const T: c_int = 4;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H_q, T, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H_kv, T, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H_kv, T, D);
    defer _ = mlx.mlx_array_free(v_dense);

    // Distinct K and V rotations (full Ornith path: turbo + GQA + causal).
    var ts = try TurboState.initHadamard(testing.allocator, s, 1, @intCast(D));
    defer ts.deinit();
    const rk = ts.rk[0];
    const rv = ts.rv[0];

    var qk = try quantizeTurbo(s, k_dense, rk, 64, 4);
    defer qk.deinit();
    var qv = try quantizeTurbo(s, v_dense, rv, 64, 4);
    defer qv.deinit();

    const k_ref = try dequantizeTurbo(s, qk.q, qk.scales, qk.biases, rk, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeTurbo(s, qv.q, qv.scales, qv.biases, rv, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref, q, k_ref, v_ref, scale, "causal", none_mask, .{ .ctx = null }, s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        4,
        64,
        rk,
        rv,
        scale,
        "causal",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

// ── K-tiled online-softmax validation (flash-quant) ──
//
// The other fused tests use T_k <= kv_attn_block, so they exercise the tiled
// path as a SINGLE block. These force a tiny `kv_attn_block` to walk MANY
// blocks over a small T_k, hitting the traps: a partial last block, a row that
// is fully masked in a straddling block (the NaN-guard path), the degenerate
// block==1, and the decode + GQA + turbo headline. All compare the tiled fused
// output to dense SDPA over the matching dequantization.

test "tiled fused: multi-block causal + partial tail + fully-masked row (affine)" {
    const s = mlx.gpuStream();
    const saved = kv_attn_block;
    kv_attn_block = 4; // [0,4)[4,8)[8,10): partial tail; [8,10) is future for q0
    defer kv_attn_block = saved;

    const B: c_int = 1;
    const H: c_int = 2;
    const T_q: c_int = 4;
    const T_k: c_int = 10;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H, T_q, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H, T_k, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H, T_k, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var qk = try quantizeAffine(s, k_dense, 64, 4);
    defer qk.deinit();
    var qv = try quantizeAffine(s, v_dense, 64, 4);
    defer qv.deinit();
    const k_ref = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeAffine(s, qv.q, qv.scales, qv.biases, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref, q, k_ref, v_ref, scale, "causal", none_mask, .{ .ctx = null }, s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        4,
        64,
        .{ .ctx = null },
        .{ .ctx = null },
        scale,
        "causal",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

test "tiled fused: block==1 (per-key) causal still matches dense SDPA (affine)" {
    const s = mlx.gpuStream();
    const saved = kv_attn_block;
    kv_attn_block = 1; // degenerate: one key per block
    defer kv_attn_block = saved;

    const B: c_int = 1;
    const H: c_int = 2;
    const T_q: c_int = 4;
    const T_k: c_int = 8;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H, T_q, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H, T_k, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H, T_k, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var qk = try quantizeAffine(s, k_dense, 64, 4);
    defer qk.deinit();
    var qv = try quantizeAffine(s, v_dense, 64, 4);
    defer qv.deinit();
    const k_ref = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeAffine(s, qv.q, qv.scales, qv.biases, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref, q, k_ref, v_ref, scale, "causal", none_mask, .{ .ctx = null }, s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        4,
        64,
        .{ .ctx = null },
        .{ .ctx = null },
        scale,
        "causal",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

test "tiled fused: decode (T_q=1) multi-block GQA + turbo matches dense SDPA" {
    const s = mlx.gpuStream();
    const saved = kv_attn_block;
    kv_attn_block = 4;
    defer kv_attn_block = saved;

    const B: c_int = 1;
    const H_q: c_int = 4;
    const H_kv: c_int = 2;
    const T_k: c_int = 10;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H_q, 1, D); // decode: T_q == 1
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H_kv, T_k, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H_kv, T_k, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var ts = try TurboState.initHadamard(testing.allocator, s, 1, @intCast(D));
    defer ts.deinit();
    const rk = ts.rk[0];
    const rv = ts.rv[0];
    var qk = try quantizeTurbo(s, k_dense, rk, 64, 4);
    defer qk.deinit();
    var qv = try quantizeTurbo(s, v_dense, rv, 64, 4);
    defer qv.deinit();
    const k_ref = try dequantizeTurbo(s, qk.q, qk.scales, qk.biases, rk, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeTurbo(s, qv.q, qv.scales, qv.biases, rv, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    // Decode: the single query sees all keys → no mask. SDPA broadcasts GQA.
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref, q, k_ref, v_ref, scale, "", none_mask, .{ .ctx = null }, s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        4,
        64,
        rk,
        rv,
        scale,
        "",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

test "tiled fused: warm continuation (T_q=3 << T_k=30) GQA causal matches dense SDPA" {
    // The case the !is_prefill gate wrongly forced to dense: a few new query
    // tokens (T_q=3) attending over a long resident context (T_k=30), aligned to
    // the tail (q0 = T_k - T_q = 27). With block=8 the first three blocks are
    // fully visible (no mask) and only the partial last block [24,30) straddles
    // — the warm-turn signature the tile-size gate routes back to tiled.
    const s = mlx.gpuStream();
    const saved = kv_attn_block;
    kv_attn_block = 8;
    defer kv_attn_block = saved;

    const B: c_int = 1;
    const H_q: c_int = 4;
    const H_kv: c_int = 2;
    const T_q: c_int = 3;
    const T_k: c_int = 30; // not a multiple of 8 → partial tail block too
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H_q, T_q, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H_kv, T_k, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H_kv, T_k, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var qk = try quantizeAffine(s, k_dense, 64, 4);
    defer qk.deinit();
    var qv = try quantizeAffine(s, v_dense, 64, 4);
    defer qv.deinit();
    const k_ref = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeAffine(s, qv.q, qv.scales, qv.biases, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    // Dense SDPA "causal" tail-aligns T_q<T_k queries (row i ↔ key q0+i) and
    // broadcasts GQA — the exact semantics tiledCausalAttention must match.
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref, q, k_ref, v_ref, scale, "causal", none_mask, .{ .ctx = null }, s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        4,
        64,
        .{ .ctx = null },
        .{ .ctx = null },
        scale,
        "causal",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

test "preferTiledPrefill routes warm small-T_q to tiled, cold full-chunk to dense" {
    // Ornith-ish dims: H_q=16, H_kv=4, hdim=128, 8 full-attn layers, 73k ctx.
    const h_q: u64 = 16;
    const h_kv: u64 = 4;
    const hdim: u64 = 128;
    const fa: u64 = 8;
    const t_k: u64 = 73_251;
    const block: u64 = 4096;
    const saved = kv_attn_tiled_budget_mb;
    kv_attn_tiled_budget_mb = 2048;
    defer kv_attn_tiled_budget_mb = saved;

    // Warm: 251 new tokens → tiled tile ≪ full-context dequant → tiled.
    try testing.expect(preferTiledPrefill(h_q, h_kv, 251, block, t_k, fa, hdim));
    // Cold: an 8192-token chunk → [H_q,8192,block] dwarfs the dequant AND blows
    // the budget ceiling → dense.
    try testing.expect(!preferTiledPrefill(h_q, h_kv, 8192, block, t_k, fa, hdim));
    // Budget=0 disables the ceiling, but cold still loses the size comparison.
    kv_attn_tiled_budget_mb = 0;
    try testing.expect(!preferTiledPrefill(h_q, h_kv, 8192, block, t_k, fa, hdim));
    try testing.expect(preferTiledPrefill(h_q, h_kv, 251, block, t_k, fa, hdim));
}

// ── TURBO × tiled × T_q>1 (the prod-config coverage gap) ──
//
// The multi-block tiled tests above are affine-only; the multi-block turbo test
// is decode (T_q=1, never masks). The turbo causal tests are T_q>1 but single
// block. So turbo + T_q>1 + MULTI-BLOCK (mask interacting with the per-block
// Q-rotation/V-rotation accumulation) was untested — exactly what warm prefill
// hits with the user's K affine-8 / V turbo-4 config. These force a tiny block
// and a warm shape (T_q ≪ T_k, q0>0) and compare to dense SDPA over the matching
// per-side dequantization.

test "tiled TURBO: K affine8 + V turbo4, T_q>1 multi-block causal matches dense SDPA" {
    const s = mlx.gpuStream();
    const saved = kv_attn_block;
    kv_attn_block = 8; // [0,8)[8,16)[16,24)[24,30): partial tail; straddle at end
    defer kv_attn_block = saved;

    const B: c_int = 1;
    const H_q: c_int = 4;
    const H_kv: c_int = 2;
    const T_q: c_int = 3;
    const T_k: c_int = 30; // q0 = 27
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H_q, T_q, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H_kv, T_k, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H_kv, T_k, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var ts = try TurboState.initHadamard(testing.allocator, s, 1, @intCast(D));
    defer ts.deinit();
    const rv = ts.rv[0];

    var qk = try quantizeAffine(s, k_dense, 64, 8);
    defer qk.deinit();
    var qv = try quantizeTurbo(s, v_dense, rv, 64, 4);
    defer qv.deinit();
    const k_ref = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, 8);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeTurbo(s, qv.q, qv.scales, qv.biases, rv, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, q, k_ref, v_ref, scale, "causal", none_mask, .{ .ctx = null }, s));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        8,
        64,
        4,
        64,
        .{ .ctx = null }, // K affine: no rotation
        rv, // V turbo: rotation
        scale,
        "causal",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

test "tiled TURBO: K turbo4 + V turbo4 (Rk != Rv), T_q>1 multi-block causal matches dense SDPA" {
    const s = mlx.gpuStream();
    const saved = kv_attn_block;
    kv_attn_block = 8;
    defer kv_attn_block = saved;

    const B: c_int = 1;
    const H_q: c_int = 4;
    const H_kv: c_int = 2;
    const T_q: c_int = 3;
    const T_k: c_int = 30;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H_q, T_q, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H_kv, T_k, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H_kv, T_k, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var ts = try TurboState.initHadamard(testing.allocator, s, 1, @intCast(D));
    defer ts.deinit();
    const rk = ts.rk[0];
    const rv = ts.rv[0];

    var qk = try quantizeTurbo(s, k_dense, rk, 64, 4);
    defer qk.deinit();
    var qv = try quantizeTurbo(s, v_dense, rv, 64, 4);
    defer qv.deinit();
    const k_ref = try dequantizeTurbo(s, qk.q, qk.scales, qk.biases, rk, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeTurbo(s, qv.q, qv.scales, qv.biases, rv, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, q, k_ref, v_ref, scale, "causal", none_mask, .{ .ctx = null }, s));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        4,
        64,
        rk,
        rv,
        scale,
        "causal",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}
