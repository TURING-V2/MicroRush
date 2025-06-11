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
    positions: std.StringHashMap(Position),
    signal_thread: ?std.Thread,
    should_stop: std.atomic.Value(bool),
    mutex: std.Thread.Mutex,
    portfolio_manager: PortfolioManager,

    pub fn init(allocator: std.mem.Allocator, symbol_map: *const SymbolMap) TradeHandler {
        return TradeHandler{
            .allocator = allocator,
            .signal_queue = std.ArrayList(TradingSignal).init(allocator),
            .positions = std.StringHashMap(Position).init(allocator),
            .signal_thread = null,
            .should_stop = std.atomic.Value(bool).init(false),
            .mutex = std.Thread.Mutex{},
            .portfolio_manager = PortfolioManager.init(allocator, symbol_map),
        };
    }

    pub fn deinit(self: *TradeHandler) void {
        self.should_stop.store(true, .seq_cst);
        if (self.signal_thread) |thread| {
            thread.join();
        }
        self.signal_queue.deinit();
        self.positions.deinit();
    }

    pub fn start(self: *TradeHandler) !void {
        self.signal_thread = try std.Thread.spawn(.{}, signalThreadFunction, .{self});
    }

    pub fn addSignal(self: *TradeHandler, signal: TradingSignal) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.signal_queue.append(signal);
        if (signal.signal_type == .BUY) {
            try self.positions.put(signal.symbol_name, Position{
                .symbol_name = signal.symbol_name,
                .is_open = true,
                .entry_rsi = signal.rsi_value,
                .timestamp = signal.timestamp,
                .signal_strength = signal.signal_strength,
            });
            //std.log.info("SIGNAL QUEUED: {s} BUY - RSI: {d:.2}, Orderbook: {d:.2}%", .{ signal.symbol_name, signal.rsi_value, signal.orderbook_percentage });
        } else if (signal.signal_type == .SELL) {
            if (self.positions.getPtr(signal.symbol_name)) |position| {
                position.is_open = false;
            }
            //std.log.info("SIGNAL QUEUED: {s} SELL - RSI: {d:.2}, Orderbook: {d:.2}%", .{ signal.symbol_name, signal.rsi_value, signal.orderbook_percentage });
        }
    }

    pub fn hasOpenPosition(self: *TradeHandler, symbol_name: []const u8) bool {
        if (self.positions.get(symbol_name)) |position| {
            return position.is_open;
        }
        return false;
    }

    pub fn getOpenPositionsCount(self: *TradeHandler) usize {
        var count: usize = 0;
        var iterator = self.positions.iterator();
        while (iterator.next()) |entry| {
            if (entry.value_ptr.is_open) {
                count += 1;
            }
        }
        return count;
    }

    pub fn getPendingSignalsCount(self: *TradeHandler) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
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
            std.time.sleep(10_000_000); // 10ms
        }

        std.log.info("Trade handler thread stopped", .{});
    }

    fn executeSignal(self: *TradeHandler, signal: TradingSignal) !void {
        try self.portfolio_manager.processSignal(signal);
    }

    pub fn getStats(self: *TradeHandler) struct { open_positions: usize, pending_signals: usize } {
        return .{
            .open_positions = self.getOpenPositionsCount(),
            .pending_signals = self.getPendingSignalsCount(),
        };
    }
};
