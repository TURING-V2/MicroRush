const std = @import("std");
const types = @import("../types.zig");
const SymbolMap = @import("../symbol-map.zig").SymbolMap;
const PortfolioManager = @import("portfolio_manager.zig").PortfolioManager;
const TradingSignal = types.TradingSignal;
const Position = types.Position;
const SignalType = types.SignalType;

pub const TradeHandler = struct {
    allocator: std.mem.Allocator,
    signal_queue: std.ArrayList(TradingSignal),
    signal_thread: ?std.Thread,
    should_stop: std.atomic.Value(bool),
    mutex: std.Thread.Mutex,
    portfolio_manager: PortfolioManager,

    high_strength_sell: std.ArrayList(TradingSignal),
    high_strength_buy: std.ArrayList(TradingSignal),
    low_strength_sell: std.ArrayList(TradingSignal),
    low_strength_buy: std.ArrayList(TradingSignal),

    pub fn init(allocator: std.mem.Allocator, symbol_map: *const SymbolMap) TradeHandler {
        return TradeHandler{
            .allocator = allocator,
            .signal_queue = std.ArrayList(TradingSignal).init(allocator),
            .signal_thread = null,
            .should_stop = std.atomic.Value(bool).init(false),
            .mutex = std.Thread.Mutex{},
            .portfolio_manager = PortfolioManager.init(allocator, symbol_map),

            .high_strength_sell = std.ArrayList(TradingSignal).init(allocator),
            .high_strength_buy = std.ArrayList(TradingSignal).init(allocator),
            .low_strength_sell = std.ArrayList(TradingSignal).init(allocator),
            .low_strength_buy = std.ArrayList(TradingSignal).init(allocator),
        };
    }

    pub fn deinit(self: *TradeHandler) void {
        self.should_stop.store(true, .seq_cst);
        if (self.signal_thread) |thread| {
            thread.join();
        }
        self.signal_queue.deinit();
        self.high_strength_sell.deinit();
        self.high_strength_buy.deinit();
        self.low_strength_sell.deinit();
        self.low_strength_buy.deinit();
        self.portfolio_manager.deinit();
    }

    pub fn start(self: *TradeHandler) !void {
        self.signal_thread = try std.Thread.spawn(.{}, signalThreadFunction, .{self});
    }

    pub fn addSignal(self: *TradeHandler, signal: TradingSignal) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const is_high = signal.signal_strength > 0.9;

        switch (signal.signal_type) {
            .SELL => {
                if (self.hasOpenPosition(signal.symbol_name)) {
                    if (is_high) {
                        try self.high_strength_sell.append(signal);
                    } else {
                        try self.low_strength_sell.append(signal);
                    }
                }
            },
            .BUY => {
                if (!self.hasOpenPosition(signal.symbol_name)) {
                    if (is_high) {
                        try self.high_strength_buy.append(signal);
                    } else {
                        try self.low_strength_buy.append(signal);
                    }
                }
            },
            .HOLD => return,
        }
    }

    pub inline fn hasOpenPosition(self: *TradeHandler, symbol_name: []const u8) bool {
        if (self.portfolio_manager.positions.get(symbol_name)) |position| {
            return position.is_open;
        }
        return false;
    }

    pub fn getOpenPositionsCount(self: *TradeHandler) usize {
        var count: usize = 0;
        self.mutex.lock();
        defer self.mutex.unlock();
        var iterator = self.portfolio_manager.positions.iterator();
        while (iterator.next()) |entry| {
            if (entry.value_ptr.is_open) {
                count += 1;
            }
        }
        return count;
    }

    pub fn getPendingSignalsCount(self: *TradeHandler) usize {
        return self.signal_queue.items.len + self.sell_queue.items.len;
    }

    fn signalThreadFunction(self: *TradeHandler) !void {
        std.log.info("Trade handler thread started", .{});
        while (!self.should_stop.load(.seq_cst)) {
            self.mutex.lock();
            const queues = &[_]*std.ArrayList(TradingSignal){
                &self.high_strength_sell,
                &self.high_strength_buy,
                &self.low_strength_sell,
                &self.low_strength_buy,
            };

            var signals_processed = false;
            for (queues) |queue| {
                while (queue.items.len > 0) {
                    const signal = queue.orderedRemove(0);
                    self.mutex.unlock();
                    try self.executeSignalFast(signal);
                    signals_processed = true;
                    self.mutex.lock();
                }
            }
            self.mutex.unlock();

            self.portfolio_manager.checkStopLossConditions() catch |err| {
                std.log.warn("Failed to check stop loss conditions: {}", .{err});
            };

            if (!signals_processed) {
                std.time.sleep(100_000);
            }
        }
        std.log.info("Trade handler thread stopped", .{});
    }

    inline fn executeSignalFast(self: *TradeHandler, signal: TradingSignal) !void {
        try self.portfolio_manager.processSignal(signal);
    }

    pub fn getStats(self: *TradeHandler) struct {
        open_positions: usize,
        pending_buy_signals: usize,
        pending_sell_signals: usize,
        recent_signals: usize,
    } {
        self.mutex.lock();
        defer self.mutex.unlock();
        return .{
            .open_positions = self.getOpenPositionsCount(),
            .pending_buy_signals = self.signal_queue.items.len,
            .pending_sell_signals = self.sell_queue.items.len,
            .recent_signals = self.recent_signals.items.len,
        };
    }
};
