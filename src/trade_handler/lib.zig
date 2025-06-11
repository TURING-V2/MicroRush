const std = @import("std");
const types = @import("../types.zig");
const SymbolMap = @import("../symbol-map.zig").SymbolMap;
const PortfolioManager = @import("portfolio_manager.zig").PortfolioManager;
const TradingSignal = types.TradingSignal;
const Position = types.Position;
const SignalType = types.SignalType;

const RecentSignal = struct {
    symbol_name: []const u8,
    signal_type: SignalType,
    timestamp: i64,
};

pub const TradeHandler = struct {
    allocator: std.mem.Allocator,
    signal_queue: std.ArrayList(TradingSignal),
    signal_thread: ?std.Thread,
    should_stop: std.atomic.Value(bool),
    mutex: std.Thread.Mutex,
    portfolio_manager: PortfolioManager,
    recent_signals: std.ArrayList(RecentSignal),
    signal_cooldown_ns: i64,

    pub fn init(allocator: std.mem.Allocator, symbol_map: *const SymbolMap) TradeHandler {
        return TradeHandler{
            .allocator = allocator,
            .signal_queue = std.ArrayList(TradingSignal).init(allocator),
            .signal_thread = null,
            .should_stop = std.atomic.Value(bool).init(false),
            .mutex = std.Thread.Mutex{},
            .portfolio_manager = PortfolioManager.init(allocator, symbol_map),
            .recent_signals = std.ArrayList(RecentSignal).init(allocator),
            .signal_cooldown_ns = 50_000_000, // 50 ms
        };
    }

    pub fn deinit(self: *TradeHandler) void {
        self.should_stop.store(true, .seq_cst);
        if (self.signal_thread) |thread| {
            thread.join();
        }
        self.signal_queue.deinit();
        self.recent_signals.deinit();
        self.portfolio_manager.deinit();
    }

    fn isDuplicateSignal(self: *TradeHandler, signal: TradingSignal) bool {
        const current_time = std.time.nanoTimestamp();

        for (self.recent_signals.items) |recent| {
            if (std.mem.eql(u8, recent.symbol_name, signal.symbol_name) and
                recent.signal_type == signal.signal_type and
                (current_time - recent.timestamp) < self.signal_cooldown_ns)
            {
                return true;
            }
        }
        return false;
    }

    fn cleanupRecentSignals(self: *TradeHandler) void {
        const current_time = std.time.nanoTimestamp();
        var i: usize = 0;

        while (i < self.recent_signals.items.len) {
            if ((current_time - self.recent_signals.items[i].timestamp) > self.signal_cooldown_ns) {
                _ = self.recent_signals.orderedRemove(i);
            } else {
                i += 1;
            }
        }
    }

    pub fn start(self: *TradeHandler) !void {
        self.signal_thread = try std.Thread.spawn(.{}, signalThreadFunction, .{self});
    }

    pub fn addSignal(self: *TradeHandler, signal: TradingSignal) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.cleanupRecentSignals();

        if (self.isDuplicateSignal(signal)) {
            std.log.debug("DUPLICATE SIGNAL IGNORED: {s} {s}", .{ signal.symbol_name, @tagName(signal.signal_type) });
            return;
        }

        switch (signal.signal_type) {
            .BUY => {
                if (self.hasOpenPosition(signal.symbol_name)) {
                    std.log.debug("BUY SIGNAL IGNORED - Position exists: {s}", .{signal.symbol_name});
                    return;
                }
            },
            .SELL => {
                if (!self.hasOpenPosition(signal.symbol_name)) {
                    std.log.debug("SELL SIGNAL IGNORED - No position: {s}", .{signal.symbol_name});
                    return;
                }
            },
            .HOLD => return,
        }

        try self.signal_queue.append(signal);

        try self.recent_signals.append(RecentSignal{
            .symbol_name = signal.symbol_name,
            .signal_type = signal.signal_type,
            .timestamp = signal.timestamp,
        });

        std.log.debug("SIGNAL QUEUED: {s} {s} - RSI: {d:.2}, Strength: {d:.2}", .{ signal.symbol_name, @tagName(signal.signal_type), signal.rsi_value, signal.signal_strength });
    }

    pub fn hasOpenPosition(self: *TradeHandler, symbol_name: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
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
        return self.signal_queue.items.len;
    }

    fn signalThreadFunction(self: *TradeHandler) !void {
        std.log.info("Trade handler thread started", .{});
        while (!self.should_stop.load(.seq_cst)) {
            self.mutex.lock();
            // process all pending signals
            while (self.signal_queue.items.len > 0) {
                const signal = self.signal_queue.orderedRemove(0);
                self.mutex.unlock();
                try self.executeSignal(signal);
                self.mutex.lock();
            }
            self.mutex.unlock();
            std.time.sleep(1_000_000); // 1ms
        }
        std.log.info("Trade handler thread stopped", .{});
    }

    fn executeSignal(self: *TradeHandler, signal: TradingSignal) !void {
        switch (signal.signal_type) {
            .BUY => {
                if (self.hasOpenPosition(signal.symbol_name)) {
                    std.log.warn("EXECUTION BLOCKED: Already have position for {s}", .{signal.symbol_name});
                    return;
                }
            },
            .SELL => {
                if (!self.hasOpenPosition(signal.symbol_name)) {
                    std.log.warn("EXECUTION BLOCKED: No position to sell for {s}", .{signal.symbol_name});
                    return;
                }
            },
            .HOLD => return,
        }
        try self.portfolio_manager.processSignal(signal);
    }

    pub fn getStats(self: *TradeHandler) struct {
        open_positions: usize,
        pending_signals: usize,
        recent_signals: usize,
    } {
        self.mutex.lock();
        defer self.mutex.unlock();
        return .{
            .open_positions = self.getOpenPositionsCount(),
            .pending_signals = self.getPendingSignalsCount(),
            .recent_signals = self.recent_signals.items.len,
        };
    }
};
