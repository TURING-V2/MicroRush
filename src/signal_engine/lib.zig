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

// work item for parallel processing
const ProcessingTask = struct {
    rsi_values: []f32,
    bid_percentages: []f32,
    ask_percentages: []f32,
    spread_percentages: []f32,
    has_positions: []bool,
    decisions: []TradingDecision,
    symbol_names: [][]const u8,
    start_idx: usize,
    end_idx: usize,
    task_id: u32,
};

// thread-safe signal queue with batched appends to reduce lock contention
const SignalQueue = struct {
    signals: std.ArrayList(TradingSignal),
    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator) SignalQueue {
        return SignalQueue{
            .signals = std.ArrayList(TradingSignal).init(allocator),
            .mutex = std.Thread.Mutex{},
        };
    }

    pub fn deinit(self: *SignalQueue) void {
        self.signals.deinit();
    }

    // add a whole slice of signals under a single lock
    pub fn addSlice(self: *SignalQueue, new_signals: []const TradingSignal) !void {
        if (new_signals.len == 0) return;
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.signals.appendSlice(new_signals);
    }

    pub fn drainAll(self: *SignalQueue, out_signals: *std.ArrayList(TradingSignal)) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try out_signals.appendSlice(self.signals.items);
        self.signals.clearRetainingCapacity();
    }
};

pub const SignalEngine = struct {
    allocator: std.mem.Allocator,
    symbol_map: *const SymbolMap,
    stat_calc: ?*StatCalc = null,
    trade_handler: TradeHandler,

    processing_thread: ?std.Thread,
    batch_thread: ?std.Thread,
    worker_threads: []std.Thread,
    num_worker_threads: u32,
    should_stop: std.atomic.Value(bool),
    mutex: std.Thread.Mutex,

    batch_result_queue: std.ArrayList(GPUBatchResult),
    batch_queue_mutex: std.Thread.Mutex,
    batch_condition: std.Thread.Condition,

    task_queue: std.ArrayList(ProcessingTask),
    task_queue_mutex: std.Thread.Mutex,
    task_condition: std.Thread.Condition,
    tasks_finished_sem: std.Thread.Semaphore,

    signal_queue: SignalQueue,

    tasks_completed: std.atomic.Value(u64),
    total_processing_time: std.atomic.Value(u64),

    pub fn init(allocator: std.mem.Allocator, symbol_map: *const SymbolMap) !SignalEngine {
        const device_id = try stat_calc_lib.selectBestCUDADevice();
        var stat_calc = try allocator.create(StatCalc);
        stat_calc.* = try StatCalc.init(allocator, device_id);
        try stat_calc.getDeviceInfo();
        try stat_calc.warmUp();

        const trade_handler = TradeHandler.init(allocator, symbol_map);

        const cpu_count = (std.Thread.getCpuCount() catch 8) / 2; // hyper threading cores wont count
        const num_workers = @max(2, cpu_count - 2);

        const worker_threads = try allocator.alloc(std.Thread, num_workers);

        return SignalEngine{
            .allocator = allocator,
            .symbol_map = symbol_map,
            .stat_calc = stat_calc,
            .trade_handler = trade_handler,
            .processing_thread = null,
            .batch_thread = null,
            .worker_threads = worker_threads,
            .num_worker_threads = @intCast(num_workers),
            .should_stop = std.atomic.Value(bool).init(false),
            .mutex = std.Thread.Mutex{},
            .batch_result_queue = std.ArrayList(GPUBatchResult).init(allocator),
            .batch_queue_mutex = std.Thread.Mutex{},
            .batch_condition = std.Thread.Condition{},
            .task_queue = std.ArrayList(ProcessingTask).init(allocator),
            .task_queue_mutex = std.Thread.Mutex{},
            .task_condition = std.Thread.Condition{},
            .tasks_finished_sem = std.Thread.Semaphore{},
            .signal_queue = SignalQueue.init(allocator),
            .tasks_completed = std.atomic.Value(u64).init(0),
            .total_processing_time = std.atomic.Value(u64).init(0),
        };
    }

    pub fn deinit(self: *SignalEngine) void {
        self.should_stop.store(true, .seq_cst);
        self.batch_condition.signal();
        self.task_condition.broadcast();

        if (self.processing_thread) |thread| {
            thread.join();
        }
        if (self.batch_thread) |thread| {
            thread.join();
        }

        for (self.worker_threads) |thread| {
            thread.join();
        }

        self.trade_handler.deinit();
        self.batch_result_queue.deinit();
        self.task_queue.deinit();
        self.signal_queue.deinit();
        self.allocator.free(self.worker_threads);

        if (self.stat_calc) |stat_calc| {
            stat_calc.deinit();
            self.allocator.destroy(stat_calc);
        }

        const completed = self.tasks_completed.load(.seq_cst);
        const total_time = self.total_processing_time.load(.seq_cst);
        if (completed > 0) {
            const avg_time_ns = total_time / completed;
            std.log.info("Performance: {} tasks completed, avg time: {d:.3}us", .{ completed, @as(f64, @floatFromInt(avg_time_ns)) / 1000.0 });
        }
    }

    pub fn run(self: *SignalEngine) !void {
        try self.startWorkerThreads();
        try self.startProcessingThread();
        try self.trade_handler.start();
        try self.startBatchThread();
    }

    fn startWorkerThreads(self: *SignalEngine) !void {
        for (0..self.num_worker_threads) |i| {
            self.worker_threads[i] = try std.Thread.spawn(.{ .allocator = self.allocator }, workerThreadFunction, .{ self, i });
        }
        std.log.info("Started {} worker threads", .{self.num_worker_threads});
    }

    fn workerThreadFunction(self: *SignalEngine, worker_id: usize) !void {
        std.log.info("Worker thread {} started", .{worker_id});

        var local_signals = std.ArrayList(TradingSignal).init(self.allocator);
        defer local_signals.deinit();

        while (!self.should_stop.load(.seq_cst)) {
            self.task_queue_mutex.lock();

            while (self.task_queue.items.len == 0 and !self.should_stop.load(.seq_cst)) {
                self.task_condition.wait(&self.task_queue_mutex);
            }

            if (self.should_stop.load(.seq_cst)) {
                self.task_queue_mutex.unlock();
                break;
            }

            const task = self.task_queue.orderedRemove(0);
            self.task_queue_mutex.unlock();

            const start_time = std.time.nanoTimestamp();
            local_signals.clearRetainingCapacity();

            self.processTaskChunk(task, &local_signals) catch |err| {
                std.log.err("Worker {} error processing task {}: {}", .{ worker_id, task.task_id, err });
            };

            // add all collected signals to the shared queue in one atomic operation
            self.signal_queue.addSlice(local_signals.items) catch |err| {
                std.log.err("Worker {} failed to add signals to queue: {}", .{ worker_id, err });
            };

            const end_time = std.time.nanoTimestamp();
            _ = self.tasks_completed.fetchAdd(1, .seq_cst);
            _ = self.total_processing_time.fetchAdd(@as(u64, @intCast(end_time - start_time)), .seq_cst);

            // signal to the main processing thread that this task is complete
            self.tasks_finished_sem.post();
        }

        std.log.info("Worker thread {} stopped", .{worker_id});
    }

    pub fn startBatchThread(self: *SignalEngine) !void {
        self.batch_thread = try std.Thread.spawn(.{ .allocator = self.allocator }, batchThreadFunction, .{self});
    }

    fn batchThreadFunction(self: *SignalEngine) void {
        std.log.info("Batch processing thread started", .{});

        while (!self.should_stop.load(.seq_cst)) {
            const batch_results = self.stat_calc.?.calculateSymbolMapBatch(self.symbol_map, 6) catch |err| {
                std.log.err("Error calculating batch: {}", .{err});
                std.time.sleep(1_000_000_000); // 1s
                continue;
            };

            self.batch_queue_mutex.lock();
            self.batch_result_queue.append(batch_results) catch |err| {
                std.log.err("Error queuing batch result: {}", .{err});
            };
            self.batch_queue_mutex.unlock();

            self.batch_condition.signal();
            std.time.sleep(50_000_000); // 50ms
        }

        std.log.info("Batch processing thread stopped", .{});
    }

    pub fn startProcessingThread(self: *SignalEngine) !void {
        self.processing_thread = try std.Thread.spawn(.{ .allocator = self.allocator }, processingThreadFunction, .{self});
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

            var batch_result = self.batch_result_queue.orderedRemove(0);
            self.batch_queue_mutex.unlock();

            self.processSignalsParallel(&batch_result.rsi, &batch_result.orderbook) catch |err| {
                std.log.err("Error processing signals: {}", .{err});
            };
        }

        std.log.info("Signal processing thread stopped", .{});
    }

    fn processSignalsParallel(self: *SignalEngine, rsi_results: *GPURSIResultBatch, orderbook_results: *GPUOrderBookResultBatch) !void {
        const num_symbols = @min(self.symbol_map.count(), MAX_SYMBOLS);
        if (num_symbols == 0) return;

        // --- 1. Prepare Data ---
        const current_rsi_values = try self.allocator.alloc(f32, num_symbols);
        defer self.allocator.free(current_rsi_values);

        const bid_percentages = try self.allocator.alloc(f32, num_symbols);
        defer self.allocator.free(bid_percentages);

        const ask_percentages = try self.allocator.alloc(f32, num_symbols);
        defer self.allocator.free(ask_percentages);

        const spread_percentages = try self.allocator.alloc(f32, num_symbols);
        defer self.allocator.free(spread_percentages);

        const has_positions = try self.allocator.alloc(bool, num_symbols);
        defer self.allocator.free(has_positions);

        const decisions = try self.allocator.alloc(TradingDecision, num_symbols);
        defer self.allocator.free(decisions);

        const symbol_names = try self.allocator.alloc([]const u8, num_symbols);
        defer self.allocator.free(symbol_names);
        self.mutex.lock();
        var symbol_idx: usize = 0;
        var iterator = self.symbol_map.iterator();
        while (iterator.next()) |entry| {
            if (symbol_idx >= num_symbols) break;
            symbol_names[symbol_idx] = entry.key_ptr.*;

            const valid_count = rsi_results.valid_rsi_count[symbol_idx];
            current_rsi_values[symbol_idx] = if (valid_count > 0)
                rsi_results.rsi_values[symbol_idx][@intCast(valid_count - 1)]
            else
                -1.0;

            bid_percentages[symbol_idx] = orderbook_results.bid_percentage[symbol_idx];
            ask_percentages[symbol_idx] = orderbook_results.ask_percentage[symbol_idx];
            spread_percentages[symbol_idx] = orderbook_results.spread_percentage[symbol_idx];
            has_positions[symbol_idx] = self.trade_handler.hasOpenPosition(symbol_names[symbol_idx]);
            symbol_idx += 1;
        }
        self.mutex.unlock();

        // --- 2. Create and Dispatch Tasks ---
        const min_chunk_size = 64;
        const simd_alignment = 8;
        var chunk_size = @max(min_chunk_size, (num_symbols + self.num_worker_threads - 1) / self.num_worker_threads);
        chunk_size = ((chunk_size + simd_alignment - 1) / simd_alignment) * simd_alignment;
        const num_chunks = (num_symbols + chunk_size - 1) / chunk_size;
        if (num_chunks == 0) return;

        {
            self.task_queue_mutex.lock();
            defer self.task_queue_mutex.unlock();
            for (0..num_chunks) |chunk_idx| {
                const start_idx = chunk_idx * chunk_size;
                const end_idx = @min(start_idx + chunk_size, num_symbols);
                if (start_idx >= end_idx) continue;

                try self.task_queue.append(ProcessingTask{
                    .rsi_values = current_rsi_values,
                    .bid_percentages = bid_percentages,
                    .ask_percentages = ask_percentages,
                    .spread_percentages = spread_percentages,
                    .has_positions = has_positions,
                    .decisions = decisions,
                    .symbol_names = symbol_names,
                    .start_idx = start_idx,
                    .end_idx = end_idx,
                    .task_id = @intCast(chunk_idx),
                });
            }
        }
        self.task_condition.broadcast();

        // --- 3. Wait for All Tasks to Complete ---
        // consuming no CPU.
        for (0..num_chunks) |_| {
            self.tasks_finished_sem.wait();
        }

        // --- 4. Collect and Process Results ---
        var collected_signals = std.ArrayList(TradingSignal).init(self.allocator);
        defer collected_signals.deinit();

        try self.signal_queue.drainAll(&collected_signals);

        for (collected_signals.items) |signal| {
            try self.trade_handler.addSignal(signal);
        }

        // if (collected_signals.items.len > 0) {
        //     std.log.debug("Processed {} symbols, generated {} signals across {} chunks", .{ num_symbols, collected_signals.items.len, num_chunks });
        // }
    }

    fn processTaskChunk(_: *SignalEngine, task: ProcessingTask, out_signals: *std.ArrayList(TradingSignal)) !void {
        const chunk_len = task.end_idx - task.start_idx;

        analyze_trading_signals_simd(
            task.rsi_values[task.start_idx..].ptr,
            task.bid_percentages[task.start_idx..].ptr,
            task.ask_percentages[task.start_idx..].ptr,
            task.spread_percentages[task.start_idx..].ptr,
            task.has_positions[task.start_idx..].ptr,
            @intCast(chunk_len),
            task.decisions[task.start_idx..].ptr,
        );

        for (task.start_idx..task.end_idx) |i| {
            const decision = task.decisions[i];
            if (!decision.spread_valid) continue;

            const symbol_name = task.symbol_names[i];
            const rsi_value = task.rsi_values[i];

            if (decision.should_generate_buy) {
                try out_signals.append(.{
                    .symbol_name = symbol_name,
                    .signal_type = .BUY,
                    .rsi_value = rsi_value,
                    .orderbook_percentage = task.bid_percentages[i],
                    .timestamp = @as(i64, @intCast(std.time.nanoTimestamp())),
                    .signal_strength = decision.signal_strength,
                });
            }

            if (decision.should_generate_sell) {
                try out_signals.append(.{
                    .symbol_name = symbol_name,
                    .signal_type = .SELL,
                    .rsi_value = rsi_value,
                    .orderbook_percentage = task.ask_percentages[i],
                    .timestamp = @as(i64, @intCast(std.time.nanoTimestamp())),
                    .signal_strength = decision.signal_strength,
                });
            }
        }
    }
};
