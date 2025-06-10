const DataAggregator = @import("data_aggregator/lib.zig").DataAggregator;
const SignalEngine = @import("signal_engine/lib.zig").SignalEngine;
const symbol_map = @import("symbol-map.zig");
const SymbolMap = symbol_map.SymbolMap;
const std = @import("std");
const types = @import("types.zig");
const Symbol = types.Symbol;

var should_stop: bool = false;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var enable_metrics = false;
    for (args) |arg| {
        if (std.mem.eql(u8, arg, "--metrics"[0..]) or std.mem.eql(u8, arg, "metrics"[0..])) {
            enable_metrics = true;
            break;
        }
    }

    const smp_allocator = std.heap.smp_allocator;

    var aggregator = try DataAggregator.init(enable_metrics, smp_allocator);
    defer aggregator.deinit();

    var signal_engine = try SignalEngine.init(smp_allocator, aggregator.symbol_map);
    defer signal_engine.deinit();

    try aggregator.connectToBinance();
    try aggregator.run();

    std.debug.print("WebSockets flowing, starting continuous Signal Engine and Trading...\n", .{});

    const sleep_ns = 50_000_000; // 50 ms
    const max_duration_ns = 180 * 60 * 1_000_000_000; // 180 min
    const warm_up_duration_ns = 5 * 60 * 1_000_000_000; // 5 min
    const start_time = std.time.nanoTimestamp();

    var signal_engine_started = false;

    while (!should_stop) {
        const now = std.time.nanoTimestamp();

        if (now - start_time >= max_duration_ns) {
            std.log.info("Reached 180-minute limit, stopping loop.", .{});
            // TODO: SELL ALL TRADES
            break;
        }

        if (now - start_time >= warm_up_duration_ns and !signal_engine_started) {
            std.log.info("Warm-up complete, starting signal engine...", .{});
            try signal_engine.run();
            signal_engine_started = true;
        }

        std.time.sleep(sleep_ns);
    }

    std.log.info("Stopping aggregator...", .{});
    try aggregator.stop();
}
