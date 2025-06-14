const binance = @import("binance.zig");
const SymbolMap = @import("../symbol-map.zig").SymbolMap;
const metrics = @import("../metrics.zig");
const std = @import("std");

pub const DataAggregator = struct {
    symbol_map: *SymbolMap,
    binance: binance.Client,
    allocator: std.mem.Allocator,
    enable_metrics: bool,
    metrics_channel: ?*metrics.MetricsChannel,
    metrics_thread: ?*std.Thread,
    metrics_collector: ?metrics.MetricsCollector,

    pub fn init(enable_metrics: bool, allocator: std.mem.Allocator) !DataAggregator {
        var metrics_channel: ?*metrics.MetricsChannel = null;
        var metrics_thread: ?*std.Thread = null;
        var metrics_collector: ?metrics.MetricsCollector = null;

        if (enable_metrics) {
            metrics_channel = try metrics.MetricsChannel.init(allocator);
            metrics_collector = metrics.MetricsCollector.init(metrics_channel.?);
            const thread = try allocator.create(std.Thread);
            thread.* = try std.Thread.spawn(.{}, metrics.metricsThread, .{metrics_channel.?});
            metrics_thread = thread;
        }

        const sym_map = try allocator.create(SymbolMap);
        sym_map.* = SymbolMap.init(allocator);

        const binance_client = try binance.Client.init(allocator, if (enable_metrics) &metrics_collector.? else null);

        return DataAggregator{
            .allocator = allocator,
            .enable_metrics = enable_metrics,
            .metrics_channel = metrics_channel,
            .metrics_thread = metrics_thread,
            .metrics_collector = metrics_collector,
            .symbol_map = sym_map,
            .binance = binance_client,
        };
    }

    pub fn deinit(self: *DataAggregator) void {
        std.debug.print("WebSocket listener stopped\n", .{});
        self.binance.ws_client.stopListener() catch |err| {
            std.debug.print("Error stopping WebSocket listener: {}\n", .{err});
        };

        self.symbol_map.deinit();
        self.allocator.destroy(self.symbol_map);

        self.binance.deinit();

        if (self.enable_metrics) {
            if (self.metrics_channel) |channel| {
                channel.stop();
            }
            if (self.metrics_thread) |thread| {
                thread.join();
                self.allocator.destroy(thread);
            }
            if (self.metrics_channel) |channel| {
                channel.deinit();
            }
        }
    }

    pub fn connectToBinance(self: *DataAggregator) !void {
        try self.binance.connect();
        try self.binance.loadSymbols(self.symbol_map);
    }

    pub fn run(self: *DataAggregator) !void {
        try self.binance.ws_client.startListener(self.symbol_map);
        std.debug.print("WebSocket listener started\n", .{});
    }
};
