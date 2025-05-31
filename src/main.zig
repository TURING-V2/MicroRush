const data_aggregator = @import("data_aggregator/lib.zig");
const symbol_map = @import("symbol-map.zig");
const std = @import("std");

pub fn main() !void {
    var aggregator = try data_aggregator.DataAggregator.init();
    try aggregator.connectToBinance();
    try aggregator.run();
    std.log.info("Waiting for WebSocket data...", .{});

    std.time.sleep(2 * std.time.ns_per_min);

    try aggregator.stop();

    //symbol_map.dump(&aggregator.symbol_map);

    defer aggregator.deinit();
}
