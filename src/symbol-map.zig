const std = @import("std");
const types = @import("types.zig");

pub const SymbolMap = types.SymbolMap;

pub fn init(self: *SymbolMap, allocator: std.mem.Allocator) !SymbolMap {
    return self.init(allocator);
}

pub fn deinit(self: *SymbolMap) void {
    self.deinit();
}

pub fn dump(self: *const SymbolMap) void {
    var it = self.iterator();
    while (it.next()) |entry| {
        const symbol = entry.value_ptr;
        std.log.info("Symbol: {s}", .{entry.key_ptr.*});
        std.log.info("  Ticker queue length: {}", .{symbol.count});

        var idx: usize = 0;
        while (idx < symbol.count) : (idx += 1) {
            const ohlc = symbol.ticker_queue[idx];
            std.log.info("    [{}] O:{d} H:{d} L:{d} C:{d} V:{d}", .{ idx, ohlc.open_price, ohlc.high_price, ohlc.low_price, ohlc.close_price, ohlc.volume });
        }
        std.log.info("  OrderBook Bids: {d} Asks: {d}", .{ symbol.orderbook.bids.len, symbol.orderbook.asks.len });

        symbol.orderbook.dump();
    }
}
