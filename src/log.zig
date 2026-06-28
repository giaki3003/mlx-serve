const std = @import("std");
const builtin = @import("builtin");

pub const Level = enum {
    off,
    err,
    warn,
    info,
    debug,

    pub fn fromString(s: []const u8) ?Level {
        if (std.mem.eql(u8, s, "off")) return .off;
        if (std.mem.eql(u8, s, "error")) return .err;
        if (std.mem.eql(u8, s, "warn")) return .warn;
        if (std.mem.eql(u8, s, "info")) return .info;
        if (std.mem.eql(u8, s, "debug")) return .debug;
        return null;
    }
};

// Default to silent under `zig build test`: passing tests exercise
// expected-path info/warn logs (tool-parse recovery, hot-cache eviction, image
// token insertion, quant-mode rejection, …) that otherwise flood the runner.
// Real test failures surface through the test runner (panics / error returns),
// never through log.zig, so silencing here only removes noise — a failing test
// still fails loudly. Non-test builds keep .info; both honor setLevel.
var current_level: Level = if (builtin.is_test) .off else .info;

pub fn setLevel(level: Level) void {
    current_level = level;
}

pub fn isDebug() bool {
    return @intFromEnum(current_level) >= @intFromEnum(Level.debug);
}

pub fn info(comptime fmt: []const u8, args: anytype) void {
    if (@intFromEnum(current_level) >= @intFromEnum(Level.info)) {
        std.debug.print(fmt, args);
    }
}

pub fn warn(comptime fmt: []const u8, args: anytype) void {
    if (@intFromEnum(current_level) >= @intFromEnum(Level.warn)) {
        std.debug.print(fmt, args);
    }
}

pub fn err(comptime fmt: []const u8, args: anytype) void {
    if (@intFromEnum(current_level) >= @intFromEnum(Level.err)) {
        std.debug.print(fmt, args);
    }
}

pub fn debug(comptime fmt: []const u8, args: anytype) void {
    if (@intFromEnum(current_level) >= @intFromEnum(Level.debug)) {
        std.debug.print(fmt, args);
    }
}

// ── Tests ──

const testing = std.testing;

test "Level.fromString valid levels" {
    try testing.expectEqual(Level.err, Level.fromString("error").?);
    try testing.expectEqual(Level.warn, Level.fromString("warn").?);
    try testing.expectEqual(Level.info, Level.fromString("info").?);
    try testing.expectEqual(Level.debug, Level.fromString("debug").?);
}

test "Level.fromString invalid returns null" {
    try testing.expect(Level.fromString("verbose") == null);
    try testing.expect(Level.fromString("") == null);
    try testing.expect(Level.fromString("INFO") == null);
}

test "Level ordering" {
    // err < warn < info < debug
    try testing.expect(@intFromEnum(Level.err) < @intFromEnum(Level.warn));
    try testing.expect(@intFromEnum(Level.warn) < @intFromEnum(Level.info));
    try testing.expect(@intFromEnum(Level.info) < @intFromEnum(Level.debug));
}

test "setLevel changes current level" {
    const original = current_level;
    defer setLevel(original); // restore

    setLevel(.debug);
    try testing.expectEqual(Level.debug, current_level);
    try testing.expect(isDebug());
    setLevel(.err);
    try testing.expectEqual(Level.err, current_level);
    try testing.expect(!isDebug());
}
