const DataAggregator = @import("data_aggregator/lib.zig").DataAggregator;
const stat_calc = @import("stat_calc/lib.zig");
const StatCalc = stat_calc.StatCalc;
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

    const device_id = try stat_calc.selectBestCUDADevice();
    const smp_allocator = std.heap.smp_allocator;
    std.log.info("Selected CUDA device: {}", .{device_id});
    var stat_cal = try StatCalc.init(smp_allocator, device_id);
    defer stat_cal.deinit();
    try stat_cal.getDeviceInfo();
    try stat_cal.warmUp();

    var aggregator = try DataAggregator.init(enable_metrics, smp_allocator);
    defer aggregator.deinit();
    try aggregator.connectToBinance();
    try aggregator.run();

    std.debug.print("WebSockets flowing, starting continuous CUDA calculations (Ctrl+C to stop)...\n", .{});
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
            std.log.info("Running batch calculation...", .{});
            stat_cal.calculateSymbolMapBatch(&aggregator.symbol_map, 6) catch |err| {
                std.log.err("Batch calculation failed: {}", .{err});
                continue;
            };
        }
        // mutex.lock();
        // symbol_map.dump(&aggregator.symbol_map);
        // mutex.unlock();

        std.time.sleep(sleep_ns);
    }

    std.log.info("Stopping aggregator...", .{});
    try aggregator.stop();
}
