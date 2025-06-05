const DataAggregator = @import("data_aggregator/lib.zig").DataAggregator;
const cuda = @import("cuda/lib.zig");
const StatCalc = cuda.StatCalc;
const SymbolMap = @import("symbol-map.zig").SymbolMap;
const std = @import("std");

var should_stop: bool = false;

// fn handleSigint(_: c_int) callconv(.C) void {
//     should_stop = true;
//     std.log.info("Received SIGINT, stopping gracefully...", .{});
// }

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

    const device_id = try cuda.selectBestCUDADevice();
    const smp_allocator = std.heap.smp_allocator;
    std.log.info("Selected CUDA device: {}", .{device_id});
    var stat_calc = try StatCalc.init(smp_allocator, device_id);
    defer stat_calc.deinit();
    try stat_calc.getDeviceInfo();
    try stat_calc.warmUp();

    var aggregator = try DataAggregator.init(enable_metrics, smp_allocator);
    defer aggregator.deinit();
    try aggregator.connectToBinance();
    try aggregator.run();

    // const act = std.posix.Sigaction{};
    // try std.posix.sigaction(std.posix.SIG.INT, &act, null);

    std.debug.print("WebSockets flowing, starting continuous CUDA calculations (Ctrl+C to stop)...\n", .{});
    std.log.info("=== Starting CUDA calculations with default parameters ===", .{});

    var mutex = std.Thread.Mutex{};
    while (!should_stop) {
        std.log.info("Running batch calculation...", .{});
        mutex.lock();
        stat_calc.calculateSymbolMapBatch(&aggregator.symbol_map, 14, 14) catch |err| {
            std.log.err("Batch calculation failed: {}", .{err});
            mutex.unlock();
            continue;
        };
        mutex.unlock();
    }

    std.log.info("Stopping aggregator...", .{});
    try aggregator.stop();
    //SymbolMap.dump(&aggregator.symbol_map);
}
