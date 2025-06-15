const DataAggregator = @import("data_aggregator/lib.zig").DataAggregator;
const SignalEngine = @import("signal_engine/lib.zig").SignalEngine;
const symbol_map = @import("symbol-map.zig");
const SymbolMap = symbol_map.SymbolMap;
const std = @import("std");
const types = @import("types.zig");
const Symbol = types.Symbol;

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

    const max_duration_ns = 180 * 60 * 1_000_000_000; // 180 min
    const warm_up_duration_ns = 10 * 60 * 1_000_000_000; // 10 min

    std.time.sleep(warm_up_duration_ns);
    std.log.info("Warm-up complete, starting signal engine...", .{});
    try signal_engine.run();

    const remaining_time = max_duration_ns - warm_up_duration_ns;
    std.time.sleep(remaining_time);

    std.log.info("Reached 180-minute limit, stopping...", .{});
    // TODO: SELL ALL TRADES
}
