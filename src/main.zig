const DataAggregator = @import("data_aggregator/lib.zig").DataAggregator;
const SignalEngine = @import("signal_engine/lib.zig").SignalEngine;
const symbol_map = @import("symbol-map.zig");
const SymbolMap = symbol_map.SymbolMap;
const std = @import("std");

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

    const aggregator_sym_map = aggregator.symbol_map;
    var signal_engine = try SignalEngine.init(smp_allocator, aggregator_sym_map);

    try aggregator.connectToBinance();
    try aggregator.run();

    std.debug.print("WebSockets flowing, starting continuous Signal Engine and Trading...\n", .{});
    std.log.info("=== Starting CUDA calculations with default parameters ===", .{});

    const sleep_ns = 500_000_000; //500 ms
    const max_duration_ns = 10 * 60 * 1_000_000_000; //2 min
    const warm_up_duration_ns = 5 * 60 * 1_000_000_000; //5 min
    const start_time = std.time.nanoTimestamp();
    //var mutex = std.Thread.Mutex{};
    while (!should_stop) {
        const now = std.time.nanoTimestamp();
        if (now - start_time >= max_duration_ns) {
            std.log.info("Reached 2-minute limit, stopping loop.", .{});
            break;
        }
        if (now - start_time >= warm_up_duration_ns) {
            try signal_engine.run();
        }
        // mutex.lock();
        // symbol_map.dump(&aggregator.symbol_map);
        // mutex.unlock();

        std.time.sleep(sleep_ns);
    }

    std.log.info("Stopping aggregator...", .{});
    try aggregator.stop();
}
