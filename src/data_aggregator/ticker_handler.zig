const std = @import("std");
const json = std.json;
const websocket = @import("websocket");

const SymbolMap = @import("../symbol-map.zig").SymbolMap;
const OHLC = @import("../types.zig").OHLC;
const metrics = @import("../metrics.zig");

pub const TickerHandler = struct {
    symbol_map: *SymbolMap,
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex = .{},
    message_count: u64,
    last_reset_time: i64,
    metrics_collector: ?*metrics.MetricsCollector,

    pub fn init(symbol_map: *SymbolMap, allocator: std.mem.Allocator, metrics_collector: ?*metrics.MetricsCollector) !TickerHandler {
        return TickerHandler{
            .symbol_map = symbol_map,
            .allocator = allocator,
            .message_count = 0,
            .last_reset_time = std.time.milliTimestamp(),
            .metrics_collector = metrics_collector,
        };
    }

    pub fn deinit(self: *TickerHandler) void {
        _ = self;
    }

    pub fn serverMessage(self: *TickerHandler, data: []u8, message_type: websocket.MessageType) !void {
        if (self.metrics_collector) |collector| {
            const start_time = std.time.nanoTimestamp();
            defer {
                const end_time = std.time.nanoTimestamp();
                const duration_ns = end_time - start_time;
                const duration_us = @as(f64, @floatFromInt(duration_ns)) / 1000.0;
                collector.recordTickerMessage(duration_us);
            }
        }

        if (message_type != .text) return;

        const parsed = json.parseFromSlice(json.Value, self.allocator, data, .{}) catch |err| {
            std.log.err("Failed to parse ticker JSON: {}", .{err});
            return;
        };
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return;

        try self.handleMiniTicker(root);
    }

    fn handleMiniTicker(self: *TickerHandler, root: json.Value) !void {
        const symbol_val = root.object.get("s") orelse return;
        if (symbol_val != .string) return;
        const symbol = symbol_val.string;

        const o_val = root.object.get("o") orelse return;
        const h_val = root.object.get("h") orelse return;
        const l_val = root.object.get("l") orelse return;
        const c_val = root.object.get("c") orelse return;
        const v_val = root.object.get("v") orelse return;

        if (o_val != .string or h_val != .string or l_val != .string or c_val != .string or v_val != .string) return;

        const open_price = std.fmt.parseFloat(f64, o_val.string) catch |err| {
            std.log.err("Failed to parse open price for {s}: {}", .{ symbol, err });
            return;
        };
        const high_price = std.fmt.parseFloat(f64, h_val.string) catch |err| {
            std.log.err("Failed to parse high price for {s}: {}", .{ symbol, err });
            return;
        };
        const low_price = std.fmt.parseFloat(f64, l_val.string) catch |err| {
            std.log.err("Failed to parse low price for {s}: {}", .{ symbol, err });
            return;
        };
        const close_price = std.fmt.parseFloat(f64, c_val.string) catch |err| {
            std.log.err("Failed to parse close price for {s}: {}", .{ symbol, err });
            return;
        };
        const volume = std.fmt.parseFloat(f64, v_val.string) catch |err| {
            std.log.err("Failed to parse volume for {s}: {}", .{ symbol, err });
            return;
        };

        if (self.symbol_map.getPtr(symbol)) |sym| {
            const ohlc = OHLC{
                .open_price = open_price,
                .high_price = high_price,
                .low_price = low_price,
                .close_price = close_price,
                .volume = volume,
            };
            self.mutex.lock();
            defer self.mutex.unlock();
            sym.addTicker(ohlc);
        }
    }
};
