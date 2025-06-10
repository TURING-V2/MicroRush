const std = @import("std");
const stat_calc_lib = @import("../stat_calc/lib.zig");
const StatCalc = stat_calc_lib.StatCalc;
const SymbolMap = @import("../symbol-map.zig").SymbolMap;
const types = @import("../types.zig");
const GPUBatchResult = types.GPUBatchResult;
const GPURSIResultBatch = types.GPURSIResultBatch;
const GPUOrderBookResultBatch = types.GPUOrderBookResultBatch;
const MAX_SYMBOLS = types.MAX_SYMBOLS;
const SignalType = types.SignalType;
const TradingSignal = types.TradingSignal;
const Position = types.Position;

extern fn analyze_trading_signals_simd(
    rsi_values: [*]f32,
    bid_percentages: [*]f32,
    ask_percentages: [*]f32,
    spread_percentages: [*]f32,
    has_positions: [*]bool,
    len: c_int,
    decisions: [*]TradingDecision,
) void;

extern fn check_bid_percentages_simd(bid_percentages: [*]f32, len: c_int, strong_bids: [*]bool) void;
extern fn check_spread_validity_simd(spread_percentages: [*]f32, len: c_int, valid_spreads: [*]bool) void;

const TradingDecision = extern struct {
    should_generate_buy: bool,
    should_generate_sell: bool,
    has_open_position: bool,
    spread_valid: bool,
};

pub const SignalEngine = struct {
    allocator: std.mem.Allocator,
    symbol_map: *const SymbolMap,
    stat_calc: ?*StatCalc = null,
    signal_queue: std.ArrayList(TradingSignal),
    positions: std.StringHashMap(Position),
    signal_thread: ?std.Thread,
    should_stop: std.atomic.Value(bool),
    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator, symbol_map: *const SymbolMap) !SignalEngine {
        const device_id = try stat_calc_lib.selectBestCUDADevice();
        var stat_calc = try allocator.create(StatCalc);
        stat_calc.* = try StatCalc.init(allocator, device_id);
        try stat_calc.getDeviceInfo();
        try stat_calc.warmUp();

        return SignalEngine{
            .allocator = allocator,
            .symbol_map = symbol_map,
            .stat_calc = stat_calc,
            .signal_queue = std.ArrayList(TradingSignal).init(allocator),
            .positions = std.StringHashMap(Position).init(allocator),
            .signal_thread = null,
            .should_stop = std.atomic.Value(bool).init(false),
            .mutex = std.Thread.Mutex{},
        };
    }

    pub fn deinit(self: *SignalEngine) void {
        self.should_stop.store(true, .seq_cst);
        if (self.signal_thread) |thread| {
            thread.join();
        }
        self.signal_queue.deinit();
        self.positions.deinit();
        self.stat_calc.?.deinit();
    }

    pub fn run(self: *SignalEngine) !void {
        var batch_results = try self.stat_calc.?.calculateSymbolMapBatch(self.symbol_map, 6);

        try self.processSignals(&batch_results.rsi, &batch_results.orderbook);
    }

    // ZIG SIMD bitwise ops is still in works
    fn processSignals(self: *SignalEngine, rsi_results: *GPURSIResultBatch, orderbook_results: *GPUOrderBookResultBatch) !void {
        const num_symbols = @min(self.symbol_map.count(), MAX_SYMBOLS);
        if (num_symbols == 0) return;

        var current_rsi_values = try self.allocator.alloc(f32, num_symbols);
        defer self.allocator.free(current_rsi_values);

        var bid_percentages = try self.allocator.alloc(f32, num_symbols);
        defer self.allocator.free(bid_percentages);

        var ask_percentages = try self.allocator.alloc(f32, num_symbols);
        defer self.allocator.free(ask_percentages);

        var spread_percentages = try self.allocator.alloc(f32, num_symbols);
        defer self.allocator.free(spread_percentages);

        var has_positions = try self.allocator.alloc(bool, num_symbols);
        defer self.allocator.free(has_positions);

        const decisions = try self.allocator.alloc(TradingDecision, num_symbols);
        defer self.allocator.free(decisions);

        var symbol_names = try self.allocator.alloc([]const u8, num_symbols);
        defer self.allocator.free(symbol_names);

        var symbol_idx: usize = 0;
        self.mutex.lock();
        var iterator = self.symbol_map.iterator();
        while (iterator.next()) |entry| {
            if (symbol_idx >= num_symbols) break;
            symbol_names[symbol_idx] = entry.key_ptr.*;

            const valid_count = @max(0, rsi_results.valid_rsi_count[symbol_idx]);
            if (valid_count > 0) {
                current_rsi_values[symbol_idx] = rsi_results.rsi_values[symbol_idx][@intCast(valid_count - 1)];
            } else {
                current_rsi_values[symbol_idx] = -1.0;
            }

            bid_percentages[symbol_idx] = orderbook_results.bid_percentage[symbol_idx];
            ask_percentages[symbol_idx] = orderbook_results.ask_percentage[symbol_idx];
            spread_percentages[symbol_idx] = orderbook_results.spread_percentage[symbol_idx];

            has_positions[symbol_idx] = self.hasOpenPosition(symbol_names[symbol_idx]);

            symbol_idx += 1;
        }
        self.mutex.unlock();

        analyze_trading_signals_simd(
            current_rsi_values.ptr,
            bid_percentages.ptr,
            ask_percentages.ptr,
            spread_percentages.ptr,
            has_positions.ptr,
            @intCast(num_symbols),
            decisions.ptr,
        );

        for (0..num_symbols) |i| {
            const decision = decisions[i];

            if (!decision.spread_valid) continue;

            const symbol_name = symbol_names[i];
            const rsi_value = current_rsi_values[i];

            if (decision.should_generate_buy) {
                try self.generateSignal(symbol_name, .BUY, rsi_value, bid_percentages[i]);
            }

            if (decision.should_generate_sell) {
                try self.generateSignal(symbol_name, .SELL, rsi_value, ask_percentages[i]);
            }
        }
    }

    fn generateSignal(self: *SignalEngine, symbol_name: []const u8, signal_type: SignalType, rsi_value: f32, orderbook_percentage: f32) !void {
        const signal = TradingSignal{
            .symbol_name = symbol_name,
            .signal_type = signal_type,
            .rsi_value = rsi_value,
            .orderbook_percentage = orderbook_percentage,
            .timestamp = @intCast(std.time.nanoTimestamp()),
        };

        try self.signal_queue.append(signal);

        // update position tracking
        if (signal_type == .BUY) {
            try self.positions.put(symbol_name, Position{
                .symbol_name = symbol_name,
                .is_open = true,
                .entry_rsi = rsi_value,
                .timestamp = signal.timestamp,
            });
            std.log.info("SIGNAL: {s} BUY - RSI: {d:.2}, Orderbook: {d:.2}%", .{ symbol_name, rsi_value, orderbook_percentage });
        } else if (signal_type == .SELL) {
            if (self.positions.getPtr(symbol_name)) |position| {
                position.is_open = false;
            }
            std.log.info("SIGNAL: {s} SELL - RSI: {d:.2}, Orderbook: {d:.2}%", .{ symbol_name, rsi_value, orderbook_percentage });
        }
    }

    fn hasOpenPosition(self: *SignalEngine, symbol_name: []const u8) bool {
        if (self.positions.get(symbol_name)) |position| {
            return position.is_open;
        }
        return false;
    }

    // MOCK CODE TESTING
    pub fn startSignalThread(self: *SignalEngine) !void {
        self.signal_thread = try std.Thread.spawn(.{}, signalThreadFunction, .{self});
    }

    fn signalThreadFunction(self: *SignalEngine) void {
        while (!self.should_stop.load(.seq_cst)) {
            self.mutex.lock();

            // process all pending signals
            while (self.signal_queue.items.len > 0) {
                const signal = self.signal_queue.orderedRemove(0);
                self.executeSignal(signal);
            }

            self.mutex.unlock();

            // allow other threads to work
            std.time.sleep(10_000_000); // 10ms
        }
    }

    fn executeSignal(_: *SignalEngine, signal: TradingSignal) void {
        switch (signal.signal_type) {
            .BUY => {
                std.debug.print("EXECUTING BUY: {s} at RSI {d:.2}\n", .{ signal.symbol_name, signal.rsi_value });
            },
            .SELL => {
                std.debug.print("EXECUTING SELL: {s} at RSI {d:.2}\n", .{ signal.symbol_name, signal.rsi_value });
            },
            .HOLD => {},
        }
    }

    pub fn getSignalStats(self: *SignalEngine) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var open_positions: usize = 0;
        var position_iterator = self.positions.iterator();
        while (position_iterator.next()) |entry| {
            if (entry.value_ptr.is_open) {
                open_positions += 1;
            }
        }
        std.log.info("Signal Engine Stats - Open Positions: {}, Pending Signals: {}", .{ open_positions, self.signal_queue.items.len });
    }
};
