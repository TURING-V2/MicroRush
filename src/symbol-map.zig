const std = @import("std");
const types = @import("types.zig");
const Symbol = types.Symbol;

pub const SymbolMap = std.StringHashMap(Symbol);

pub fn dump(self: *const SymbolMap) void {
    var it = self.iterator();
    while (it.next()) |entry| {
        const symbol = entry.value_ptr;
        if (!symbol.orderbook.hasSignificantSpread(0.2)) {
            continue;
        }
        std.log.info("Symbol: {s}", .{entry.key_ptr.*});
        std.log.info(" Ticker queue length: {}", .{symbol.count});
        var idx: usize = 0;
        while (idx < symbol.count) : (idx += 1) {
            const ohlc = symbol.ticker_queue[idx];
            std.log.info(" [{}] O:{d} H:{d} L:{d} C:{d} V:{d}", .{ idx, ohlc.open_price, ohlc.high_price, ohlc.low_price, ohlc.close_price, ohlc.volume });
        }
        std.log.info(" OrderBook Bids: {d} Asks: {d}", .{ symbol.orderbook.bids.len, symbol.orderbook.asks.len });
        symbol.orderbook.dump();
    }
}
