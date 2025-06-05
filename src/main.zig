const DataAggregator = @import("data_aggregator/lib.zig").DataAggregator;
const cuda = @import("cuda/lib.zig");
const StatCalc = cuda.StatCalc;
const SymbolMap = @import("symbol-map.zig").SymbolMap;
const std = @import("std");

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

    std.debug.print("WebSockets flowing warming up in 2 mins...\n", .{});
    std.log.info("=== Starting CUDA calculations with default parameters ===", .{});
    try stat_calc.calculateSymbolMapBatch(&aggregator.symbol_map, 14, 14);
    std.time.sleep(2 * std.time.ns_per_min);

    try aggregator.stop();
    //SymbolMap.dump(&aggregator.symbol_map);
}
