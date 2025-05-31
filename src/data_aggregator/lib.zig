const binance = @import("binance.zig");
const symbol_map = @import("../symbol-map.zig");

const std = @import("std");

pub const DataAggregator = struct {
    symbol_map: symbol_map.SymbolMap,
    binance: binance.Client,
    allocator: std.mem.Allocator,

    pub fn init() !DataAggregator {
        // var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        // defer arena.deinit();
        // const allocator = arena.allocator();

        const allocator = std.heap.smp_allocator;

        const sym_map = try symbol_map.init(allocator);
        const binance_client = try binance.Client.init(allocator);
        return DataAggregator{ .symbol_map = sym_map, .binance = binance_client, .allocator = allocator };
    }
    pub fn deinit(self: *DataAggregator) void {
        symbol_map.deinit(&self.symbol_map);
        self.binance.deinit();
    }

    pub fn connectToBinance(self: *DataAggregator) !void {
        try self.binance.connect();
        try self.binance.loadSymbols(&self.symbol_map);
    }

    pub fn run(self: *DataAggregator) !void {
        try self.binance.ws_client.startListener(&self.symbol_map);
        std.log.info("WebSocket listener started", .{});
    }

    pub fn stop(self: *DataAggregator) !void {
        try self.binance.ws_client.stopListener();
        std.log.info("WebSocket listener stopped", .{});
    }
};
