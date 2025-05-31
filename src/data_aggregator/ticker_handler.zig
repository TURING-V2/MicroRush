const std = @import("std");
const json = std.json;
const websocket = @import("websocket");

const SymbolMap = @import("../symbol-map.zig").SymbolMap;
const OHLC = @import("../types.zig").OHLC;

pub const TickerHandler = struct {
    symbol_map: *SymbolMap,
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex = .{},
    message_count: u64,
    last_reset_time: i64,

    pub fn init(symbol_map: *SymbolMap, allocator: std.mem.Allocator) !TickerHandler {
        return TickerHandler{
            .symbol_map = symbol_map,
            .allocator = allocator,
            .message_count = 0,
            .last_reset_time = std.time.milliTimestamp(),
        };
    }

    pub fn deinit(self: *TickerHandler) void {
        _ = self;
    }

    pub fn serverMessage(self: *TickerHandler, data: []u8, message_type: websocket.MessageType) !void {
        if (message_type != .text) return;
        self.mutex.lock();
        self.message_count += 1;
        self.mutex.unlock();

        try self.printMessagesPerSecond();

        const parsed = json.parseFromSlice(json.Value, self.allocator, data, .{}) catch |err| {
            std.log.err("Failed to parse ticker JSON: {}", .{err});
            return;
        };
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return;
        try self.handleMiniTicker(root);
    }

    fn printMessagesPerSecond(self: *TickerHandler) !void {
        const current_time = std.time.milliTimestamp();
        const elapsed_ms = current_time - self.last_reset_time;

        // Print every ~1000ms (1 second)
        if (elapsed_ms >= 1000) {
            self.mutex.lock();
            const messages = self.message_count;
            self.message_count = 0; // Reset counter
            self.last_reset_time = current_time;
            self.mutex.unlock();

            const messages_per_sec = @as(f64, @floatFromInt(messages)) / (@as(f64, @floatFromInt(elapsed_ms)) / 1000.0);
            std.debug.print("TickerHandler messages per second: {d:.2}\n", .{messages_per_sec});
        }
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
