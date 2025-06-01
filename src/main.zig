const DataAggregator = @import("data_aggregator/lib.zig").DataAggregator;
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

    var aggregator = try DataAggregator.init(enable_metrics);
    defer aggregator.deinit();

    try aggregator.connectToBinance();
    try aggregator.run();

    std.debug.print("WebSockets flowing warming up in 2 mins...\n", .{});
    std.time.sleep(2 * std.time.ns_per_min);

    try aggregator.stop();
    //SymbolMap.dump(&aggregator.symbol_map);
}
