const std = @import("std");

//////////////////////////////////////////////////////////
pub const SignalType = enum {
    BUY,
    SELL,
    HOLD,
};

pub const TradingSignal = struct {
    symbol_name: []const u8,
    signal_type: SignalType,
    rsi_value: f32,
    orderbook_percentage: f32,
    timestamp: i64,
};

pub const Position = struct {
    symbol_name: []const u8,
    is_open: bool,
    entry_rsi: f32,
    timestamp: i64,
};

//////////////////////////////////////////////////////////
pub const MAX_SYMBOLS_CUDA = 402;
pub const MAX_SYMBOLS = MAX_SYMBOLS_CUDA;
pub const MAX_RSI_VALUES_PER_SYMBOL = 15;

pub const DeviceInfo = extern struct {
    name: [256]u8,
    major: c_int,
    minor: c_int,
    totalGlobalMem: usize,
};

pub const GPUOHLCDataBatch = extern struct {
    close_prices: [MAX_SYMBOLS][15]f32,
    counts: [MAX_SYMBOLS]u32,
};

pub const GPURSIResultBatch = extern struct {
    rsi_values: [MAX_SYMBOLS][MAX_RSI_VALUES_PER_SYMBOL]f32,
    valid_rsi_count: [MAX_SYMBOLS]c_int,
};

pub const GPUOrderBookDataBatch = extern struct {
    bid_prices: [MAX_SYMBOLS][10]f32,
    bid_quantities: [MAX_SYMBOLS][10]f32,
    ask_prices: [MAX_SYMBOLS][10]f32,
    ask_quantities: [MAX_SYMBOLS][10]f32,
    bid_counts: [MAX_SYMBOLS]u32,
    ask_counts: [MAX_SYMBOLS]u32,
};

pub const GPUOrderBookResultBatch = extern struct {
    bid_percentage: [MAX_SYMBOLS]f32,
    ask_percentage: [MAX_SYMBOLS]f32,
    total_bid_volume: [MAX_SYMBOLS]f32,
    total_ask_volume: [MAX_SYMBOLS]f32,
};

pub const GPUBatchResult = struct {
    rsi: GPURSIResultBatch,
    orderbook: GPUOrderBookResultBatch,
};

/////////////////////////////////////////////////////////////
pub const OHLC = struct {
    open_price: f64,
    high_price: f64,
    low_price: f64,
    close_price: f64,
    volume: f64,
};

pub const PriceLevel = struct {
    price: f64,
    quantity: f64,
};

pub const OrderBook = struct {
    bids: [10]PriceLevel,
    asks: [10]PriceLevel,
    bid_head: usize,
    ask_head: usize,
    bid_count: usize,
    ask_count: usize,
    last_update_id: i64,

    pub fn init() OrderBook {
        return OrderBook{
            .bids = [_]PriceLevel{PriceLevel{ .price = 0.0, .quantity = 0.0 }} ** 10,
            .asks = [_]PriceLevel{PriceLevel{ .price = 0.0, .quantity = 0.0 }} ** 10,
            .bid_head = 0,
            .ask_head = 0,
            .bid_count = 0,
            .ask_count = 0,
            .last_update_id = 0,
        };
    }

    pub fn updateLevel(self: *OrderBook, price: f64, quantity: f64, is_bid: bool) void {
        if (is_bid) {
            self.updateBidLevel(price, quantity);
        } else {
            self.updateAskLevel(price, quantity);
        }
    }

    fn updateBidLevel(self: *OrderBook, price: f64, quantity: f64) void {
        var existing_idx: ?usize = null;
        var i: usize = 0;
        while (i < self.bid_count) : (i += 1) {
            const actual_idx = (self.bid_head + i) % 10;
            if (self.bids[actual_idx].price == price) {
                existing_idx = actual_idx;
                break;
            }
        }

        if (quantity == 0.0) {
            if (existing_idx) |idx| {
                self.removeBidLevel(idx);
            }
        } else {
            if (existing_idx) |idx| {
                self.bids[idx].quantity = quantity;
            } else {
                self.addBidLevel(price, quantity);
            }
        }
    }

    fn updateAskLevel(self: *OrderBook, price: f64, quantity: f64) void {
        var existing_idx: ?usize = null;
        var i: usize = 0;
        while (i < self.ask_count) : (i += 1) {
            const actual_idx = (self.ask_head + i) % 10;
            if (self.asks[actual_idx].price == price) {
                existing_idx = actual_idx;
                break;
            }
        }

        if (quantity == 0.0) {
            if (existing_idx) |idx| {
                self.removeAskLevel(idx);
            }
        } else {
            if (existing_idx) |idx| {
                self.asks[idx].quantity = quantity;
            } else {
                self.addAskLevel(price, quantity);
            }
        }
    }

    fn addBidLevel(self: *OrderBook, price: f64, quantity: f64) void {
        if (self.bid_count == 10) {
            const worst_idx = (self.bid_head + 9) % 10;
            if (price <= self.bids[worst_idx].price) {
                return;
            }
            self.bid_count -= 1;
        }

        var insert_pos: usize = 0;
        var i: usize = 0;
        while (i < self.bid_count) : (i += 1) {
            const actual_idx = (self.bid_head + i) % 10;
            if (self.bids[actual_idx].price < price) {
                insert_pos = i;
                break;
            }
            insert_pos = i + 1;
        }

        if (insert_pos < self.bid_count) {
            var shift_i = self.bid_count;
            while (shift_i > insert_pos) : (shift_i -= 1) {
                const from_idx = (self.bid_head + shift_i - 1) % 10;
                const to_idx = (self.bid_head + shift_i) % 10;
                self.bids[to_idx] = self.bids[from_idx];
            }
        }

        const new_idx = (self.bid_head + insert_pos) % 10;
        self.bids[new_idx] = PriceLevel{ .price = price, .quantity = quantity };
        self.bid_count += 1;
    }

    fn addAskLevel(self: *OrderBook, price: f64, quantity: f64) void {
        if (self.ask_count == 10) {
            const worst_idx = (self.ask_head + 9) % 10;
            if (price >= self.asks[worst_idx].price) {
                return;
            }
            self.ask_count -= 1;
        }

        var insert_pos: usize = 0;
        var i: usize = 0;
        while (i < self.ask_count) : (i += 1) {
            const actual_idx = (self.ask_head + i) % 10;
            if (self.asks[actual_idx].price > price) {
                insert_pos = i;
                break;
            }
            insert_pos = i + 1;
        }

        if (insert_pos < self.ask_count) {
            var shift_i = self.ask_count;
            while (shift_i > insert_pos) : (shift_i -= 1) {
                const from_idx = (self.ask_head + shift_i - 1) % 10;
                const to_idx = (self.ask_head + shift_i) % 10;
                self.asks[to_idx] = self.asks[from_idx];
            }
        }

        const new_idx = (self.ask_head + insert_pos) % 10;
        self.asks[new_idx] = PriceLevel{ .price = price, .quantity = quantity };
        self.ask_count += 1;
    }

    fn removeBidLevel(self: *OrderBook, target_idx: usize) void {
        var pos: usize = 0;
        var i: usize = 0;
        while (i < self.bid_count) : (i += 1) {
            const actual_idx = (self.bid_head + i) % 10;
            if (actual_idx == target_idx) {
                pos = i;
                break;
            }
        }

        while (pos < self.bid_count - 1) : (pos += 1) {
            const current_idx = (self.bid_head + pos) % 10;
            const next_idx = (self.bid_head + pos + 1) % 10;
            self.bids[current_idx] = self.bids[next_idx];
        }
        self.bid_count -= 1;
    }

    fn removeAskLevel(self: *OrderBook, target_idx: usize) void {
        var pos: usize = 0;
        var i: usize = 0;
        while (i < self.ask_count) : (i += 1) {
            const actual_idx = (self.ask_head + i) % 10;
            if (actual_idx == target_idx) {
                pos = i;
                break;
            }
        }

        while (pos < self.ask_count - 1) : (pos += 1) {
            const current_idx = (self.ask_head + pos) % 10;
            const next_idx = (self.ask_head + pos + 1) % 10;
            self.asks[current_idx] = self.asks[next_idx];
        }
        self.ask_count -= 1;
    }

    pub fn getBestBid(self: *const OrderBook) ?f64 {
        if (self.bid_count > 0) {
            return self.bids[self.bid_head].price;
        }
        return null;
    }

    pub fn getBestAsk(self: *const OrderBook) ?f64 {
        if (self.ask_count > 0) {
            return self.asks[self.ask_head].price;
        }
        return null;
    }

    pub fn getSpread(self: *const OrderBook) ?f64 {
        const best_bid = self.getBestBid();
        const best_ask = self.getBestAsk();
        if (best_bid != null and best_ask != null) {
            return best_ask.? - best_bid.?;
        }
        return null;
    }

    pub fn dump(self: *const OrderBook) void {
        std.log.info("=== Order Book (Update ID: {}) ===", .{self.last_update_id});
        std.log.info("ASKS (ascending - lowest first):", .{});
        var i: usize = 0;
        // Display asks in reverse order so highest ask is at top (traditional order book view)
        while (i < self.ask_count) : (i += 1) {
            const idx = (self.ask_head + self.ask_count - 1 - i) % 10;
            const level = self.asks[idx];
            std.log.info("  {d:.8} @ {d:.8}", .{ level.quantity, level.price });
        }
        if (self.getSpread()) |spread| {
            std.log.info("--- SPREAD: {d:.8} ---", .{spread});
        }
        std.log.info("BIDS (descending - highest first):", .{});
        i = 0;
        while (i < self.bid_count) : (i += 1) {
            const idx = (self.bid_head + i) % 10;
            const level = self.bids[idx];
            std.log.info("  {d:.8} @ {d:.8}", .{ level.quantity, level.price });
        }
        std.log.info("================================", .{});
        //_ = self.validate();
    }

    pub fn validate(self: *const OrderBook) bool {
        var i: usize = 1;
        while (i < self.bid_count) : (i += 1) {
            const prev_idx = (self.bid_head + i - 1) % 10;
            const curr_idx = (self.bid_head + i) % 10;
            if (self.bids[prev_idx].price < self.bids[curr_idx].price) {
                std.log.err("Bid order violation at index {}: {} < {}", .{ i, self.bids[prev_idx].price, self.bids[curr_idx].price });
                return false;
            }
        }

        i = 1;
        while (i < self.ask_count) : (i += 1) {
            const prev_idx = (self.ask_head + i - 1) % 10;
            const curr_idx = (self.ask_head + i) % 10;
            if (self.asks[prev_idx].price > self.asks[curr_idx].price) {
                std.log.err("Ask order violation at index {}: {} > {}", .{ i, self.asks[prev_idx].price, self.asks[curr_idx].price });
                return false;
            }
        }

        return true;
    }
};

pub const Symbol = struct {
    ticker_queue: [15]OHLC,
    head: usize,
    count: usize,
    orderbook: OrderBook,

    pub fn init() Symbol {
        return Symbol{
            .ticker_queue = [_]OHLC{OHLC{ .open_price = 0.0, .high_price = 0.0, .low_price = 0.0, .close_price = 0.0, .volume = 0.0 }} ** 15,
            .head = 0,
            .count = 0,
            .orderbook = OrderBook.init(),
        };
    }

    pub fn addTicker(self: *Symbol, ohlc: OHLC) void {
        self.ticker_queue[self.head] = ohlc;
        self.head = (self.head + 1) % 15;
        if (self.count < 15) {
            self.count += 1;
        }
    }
};
