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
const TradeHandler = @import("../trade_handler/lib.zig").TradeHandler;
const PortfolioManager = @import("../trade_handler/portfolio_manager.zig").PortfolioManager;

extern fn analyze_trading_signals_simd(
    rsi_values: [*]f32,
    bid_percentages: [*]f32,
    ask_percentages: [*]f32,
    spread_percentages: [*]f32,
    has_positions: [*]bool,
    len: c_int,
    decisions: [*]TradingDecision,
) void;

const TradingDecision = extern struct {
    should_generate_buy: bool,
    should_generate_sell: bool,
    has_open_position: bool,
    spread_valid: bool,
    signal_strength: f32,
};

pub const SignalEngine = struct {
    allocator: std.mem.Allocator,
    symbol_map: *const SymbolMap,
    stat_calc: ?*StatCalc = null,

    trade_handler: TradeHandler,

    // GPU batch processing
    processing_thread: ?std.Thread,
    batch_thread: ?std.Thread,
    should_stop: std.atomic.Value(bool),
    mutex: std.Thread.Mutex,

    // Batch result queue for processing
    batch_result_queue: std.ArrayList(GPUBatchResult),
    batch_queue_mutex: std.Thread.Mutex,
    batch_condition: std.Thread.Condition,

    pub fn init(allocator: std.mem.Allocator, symbol_map: *const SymbolMap) !SignalEngine {
        const device_id = try stat_calc_lib.selectBestCUDADevice();
        var stat_calc = try allocator.create(StatCalc);
        stat_calc.* = try StatCalc.init(allocator, device_id);
        try stat_calc.getDeviceInfo();
        try stat_calc.warmUp();

        const trade_handler = TradeHandler.init(allocator, symbol_map);

        return SignalEngine{
            .allocator = allocator,
            .symbol_map = symbol_map,
            .stat_calc = stat_calc,
            .trade_handler = trade_handler,
            .processing_thread = null,
            .batch_thread = null,
            .should_stop = std.atomic.Value(bool).init(false),
            .mutex = std.Thread.Mutex{},
            .batch_result_queue = std.ArrayList(GPUBatchResult).init(allocator),
            .batch_queue_mutex = std.Thread.Mutex{},
            .batch_condition = std.Thread.Condition{},
        };
    }

    pub fn deinit(self: *SignalEngine) void {
        self.should_stop.store(true, .seq_cst);
        self.batch_condition.signal();

        if (self.processing_thread) |thread| {
            thread.join();
        }
        if (self.batch_thread) |thread| {
            thread.join();
        }

        self.trade_handler.deinit();
        self.batch_result_queue.deinit();
        self.stat_calc.?.deinit();
    }

    pub fn run(self: *SignalEngine) !void {
        try self.startProcessingThread();
        try self.trade_handler.start();
        try self.startBatchThread();
    }

    pub fn startBatchThread(self: *SignalEngine) !void {
        self.batch_thread = try std.Thread.spawn(.{}, batchThreadFunction, .{self});
    }

    fn batchThreadFunction(self: *SignalEngine) void {
        std.log.info("Batch processing thread started", .{});

        while (!self.should_stop.load(.seq_cst)) {
            const batch_results = self.stat_calc.?.calculateSymbolMapBatch(self.symbol_map, 6) catch |err| {
                std.log.err("Error calculating batch: {}", .{err});
                std.time.sleep(1_000_000_000);
                continue;
            };

            self.batch_queue_mutex.lock();
            self.batch_result_queue.append(batch_results) catch |err| {
                std.log.err("Error queuing batch result: {}", .{err});
            };
            self.batch_queue_mutex.unlock();

            self.batch_condition.signal();
            std.time.sleep(1_000_000); // 1ms
        }

        std.log.info("Batch processing thread stopped", .{});
    }

    pub fn startProcessingThread(self: *SignalEngine) !void {
        self.processing_thread = try std.Thread.spawn(.{}, processingThreadFunction, .{self});
    }

    fn processingThreadFunction(self: *SignalEngine) void {
        std.log.info("Signal processing thread started", .{});

        while (!self.should_stop.load(.seq_cst)) {
            self.batch_queue_mutex.lock();
            while (self.batch_result_queue.items.len == 0 and !self.should_stop.load(.seq_cst)) {
                self.batch_condition.wait(&self.batch_queue_mutex);
            }
            if (self.should_stop.load(.seq_cst)) {
                self.batch_queue_mutex.unlock();
                break;
            }
            while (self.batch_result_queue.items.len > 0) {
                var batch_result = self.batch_result_queue.orderedRemove(0);
                self.batch_queue_mutex.unlock();
                self.processSignals(&batch_result.rsi, &batch_result.orderbook) catch |err| {
                    std.log.err("Error processing signals: {}", .{err});
                };
                self.batch_queue_mutex.lock();
            }
            self.batch_queue_mutex.unlock();
        }

        std.log.info("Signal processing thread stopped", .{});
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

            has_positions[symbol_idx] = self.trade_handler.hasOpenPosition(symbol_names[symbol_idx]);

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
                try self.generateSignal(symbol_name, .BUY, rsi_value, bid_percentages[i], decision.signal_strength);
            }
            if (decision.should_generate_sell) {
                try self.generateSignal(symbol_name, .SELL, rsi_value, ask_percentages[i], decision.signal_strength);
            }
        }
    }

    fn generateSignal(self: *SignalEngine, symbol_name: []const u8, signal_type: SignalType, rsi_value: f32, orderbook_percentage: f32, signal_strength: f32) !void {
        const signal = TradingSignal{
            .symbol_name = symbol_name,
            .signal_type = signal_type,
            .rsi_value = rsi_value,
            .orderbook_percentage = orderbook_percentage,
            .timestamp = @intCast(std.time.nanoTimestamp()),
            .signal_strength = signal_strength,
        };
        try self.trade_handler.addSignal(signal);
    }
};
