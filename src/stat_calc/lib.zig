const std = @import("std");
const SymbolMap = @import("../symbol-map.zig").SymbolMap;
const types = @import("../types.zig");
const Symbol = types.Symbol;
const OHLC = types.OHLC;
const ERR = @import("../errors.zig");
const StatCalcError = @import("../errors.zig").StatCalcError;
const DeviceInfo = types.DeviceInfo;
const GPUOHLCDataBatch = types.GPUOHLCDataBatch;
const GPUOrderBookDataBatch = types.GPUOrderBookDataBatch;
const GPURSIResultBatch = types.GPURSIResultBatch;
const GPUOrderBookResultBatch = types.GPUOrderBookResultBatch;
const MAX_SYMBOLS = types.MAX_SYMBOLS;
const GPUBatchResult = types.GPUBatchResult;

pub const KERNEL_SUCCESS = ERR.KernelError{ .code = 0, .message = "Success" };

extern "c" fn cuda_wrapper_init_device(device_id: c_int) ERR.KernelError;
extern "c" fn cuda_wrapper_reset_device() ERR.KernelError;
extern "c" fn cuda_wrapper_get_device_count(count: *c_int) ERR.KernelError;
extern "c" fn cuda_wrapper_get_device_info(device_id: c_int, info: *DeviceInfo) ERR.KernelError;
extern "c" fn cuda_wrapper_select_best_device(best_device_id: *c_int) ERR.KernelError;

extern "c" fn cuda_wrapper_allocate_memory(
    d_ohlc_batch: **GPUOHLCDataBatch,
    d_orderbook_batch: **GPUOrderBookDataBatch,
    d_rsi_result: **GPURSIResultBatch,
    d_orderbook_result: **GPUOrderBookResultBatch,
) ERR.KernelError;

extern "c" fn cuda_wrapper_free_memory(
    d_ohlc_batch: ?*GPUOHLCDataBatch,
    d_orderbook_batch: ?*GPUOrderBookDataBatch,
    d_rsi_result: ?*GPURSIResultBatch,
    d_orderbook_result: ?*GPUOrderBookResultBatch,
) ERR.KernelError;

extern "c" fn cuda_wrapper_run_rsi_batch(
    d_ohlc_batch_ptr: *GPUOHLCDataBatch,
    d_rsi_results_ptr: *GPURSIResultBatch,
    h_ohlc_batch: *const GPUOHLCDataBatch,
    h_rsi_results: *GPURSIResultBatch,
    num_symbols: c_int,
    rsi_period: c_int,
) ERR.KernelError;

extern "c" fn cuda_wrapper_run_orderbook_batch(
    d_orderbook_batch_ptr: *GPUOrderBookDataBatch,
    d_results_ptr: *GPUOrderBookResultBatch,
    h_orderbook_batch: *const GPUOrderBookDataBatch,
    h_results: *GPUOrderBookResultBatch,
    num_symbols: c_int,
) ERR.KernelError;

pub const StatCalc = struct {
    allocator: std.mem.Allocator,
    device_id: c_int,

    d_ohlc_batch: ?*GPUOHLCDataBatch,
    d_orderbook_batch: ?*GPUOrderBookDataBatch,
    d_rsi_result: ?*GPURSIResultBatch,
    d_orderbook_result: ?*GPUOrderBookResultBatch,

    h_rsi_result: GPURSIResultBatch,
    h_orderbook_result: GPUOrderBookResultBatch,

    pub fn init(allocator: std.mem.Allocator, device_id: c_int) !StatCalc {
        var calc = StatCalc{
            .allocator = allocator,
            .device_id = device_id,
            .d_ohlc_batch = null,
            .d_orderbook_batch = null,
            .d_rsi_result = null,
            .d_orderbook_result = null,
            .h_rsi_result = std.mem.zeroes(GPURSIResultBatch),
            .h_orderbook_result = std.mem.zeroes(GPUOrderBookResultBatch),
        };

        try calc.initCUDADevice();
        try calc.allocateDeviceMemory();

        return calc;
    }

    fn initCUDADevice(self: *StatCalc) !void {
        const kerr = cuda_wrapper_init_device(self.device_id);
        if (kerr.code != 0) {
            std.log.err("Failed to set CUDA device via wrapper: {} ({s})", .{ kerr.code, kerr.message });
            return StatCalcError.CUDAInitFailed;
        }

        var info: DeviceInfo = undefined;
        const kerr_info = cuda_wrapper_get_device_info(self.device_id, &info);
        if (kerr_info.code != 0) {
            std.log.err("Failed to get device info via wrapper: {} ({s})", .{ kerr_info.code, kerr_info.message });
            return StatCalcError.CUDAGetPropertiesFailed;
        }

        std.log.info("Using CUDA device: {s}", .{info.name});
        std.log.info("Compute capability: {}.{}", .{ info.major, info.minor });
        std.log.info("Global memory: {} MB", .{info.totalGlobalMem / (1024 * 1024)});
    }

    fn allocateDeviceMemory(self: *StatCalc) !void {
        var d_ohlc_batch_ptr: *GPUOHLCDataBatch = undefined;
        var d_orderbook_batch_ptr: *GPUOrderBookDataBatch = undefined;
        var d_rsi_result_ptr: *GPURSIResultBatch = undefined; // Added
        //var d_stoch_result_ptr: *GPUStochRSIResultBatch = undefined;
        var d_orderbook_result_ptr: *GPUOrderBookResultBatch = undefined;

        const kerr = cuda_wrapper_allocate_memory(
            &d_ohlc_batch_ptr,
            &d_orderbook_batch_ptr,
            &d_rsi_result_ptr,
            //&d_stoch_result_ptr,
            &d_orderbook_result_ptr,
        );

        if (kerr.code != 0) {
            std.log.err("CUDA memory allocation failed via wrapper: {} ({s})", .{ kerr.code, kerr.message });
            return StatCalcError.CUDAMemoryAllocationFailed;
        }

        self.d_ohlc_batch = d_ohlc_batch_ptr;
        self.d_orderbook_batch = d_orderbook_batch_ptr;
        self.d_rsi_result = d_rsi_result_ptr;
        self.d_orderbook_result = d_orderbook_result_ptr;
    }

    pub fn calculateSymbolMapBatch(self: *StatCalc, symbol_map: *SymbolMap, rsi_period: u32) !GPUBatchResult {
        const symbol_count = symbol_map.count();
        if (symbol_count == 0) {
            std.log.warn("SymbolMap is empty, nothing to calculate", .{});
            return ERR.Dump.MarketDataEmpty;
        }

        var valid_symbol_count: usize = 0;
        var mutex = std.Thread.Mutex{};
        mutex.lock();
        var iterator = symbol_map.iterator();
        while (iterator.next()) |entry| {
            if (entry.value_ptr.*.count == 15) {
                valid_symbol_count += 1;
            }
        }
        mutex.unlock();

        if (valid_symbol_count == 0) {
            std.log.warn("No symbols have count == 15 for OHLC, nothing to calculate for StochRSI", .{});
        }
        var rsi_valid_symbol_count = valid_symbol_count;

        if (rsi_valid_symbol_count > MAX_SYMBOLS) {
            std.log.warn("Valid symbol count {} for RSI exceeds MAX_SYMBOLS {}, truncating to {}", .{ rsi_valid_symbol_count, MAX_SYMBOLS, MAX_SYMBOLS });
            rsi_valid_symbol_count = MAX_SYMBOLS;
        }

        std.log.info("Processing {} valid symbols for RSI in batch mode...", .{rsi_valid_symbol_count});

        var symbols_slice = try self.allocator.alloc(Symbol, symbol_map.count());
        defer self.allocator.free(symbols_slice);
        var symbol_names = try self.allocator.alloc([]const u8, symbol_map.count());
        defer self.allocator.free(symbol_names);

        var rsi_symbols_slice_temp = try self.allocator.alloc(Symbol, rsi_valid_symbol_count);
        defer self.allocator.free(rsi_symbols_slice_temp);
        var rsi_symbol_names_temp = try self.allocator.alloc([]const u8, rsi_valid_symbol_count);
        defer self.allocator.free(rsi_symbol_names_temp);

        var orderbook_symbols_slice_temp = try self.allocator.alloc(Symbol, symbol_map.count());
        defer self.allocator.free(orderbook_symbols_slice_temp);
        var orderbook_symbol_names_temp = try self.allocator.alloc([]const u8, symbol_map.count());
        defer self.allocator.free(orderbook_symbol_names_temp);

        iterator = symbol_map.iterator();
        var rsi_idx: usize = 0;
        var orderbook_idx: usize = 0;
        var all_idx: usize = 0;

        mutex.lock();
        while (iterator.next()) |entry| {
            symbols_slice[all_idx] = entry.value_ptr.*;
            symbol_names[all_idx] = entry.key_ptr.*;

            if (rsi_idx < rsi_valid_symbol_count and entry.value_ptr.*.count == 15) {
                rsi_symbol_names_temp[rsi_idx] = entry.key_ptr.*;
                rsi_symbols_slice_temp[rsi_idx] = entry.value_ptr.*;
                rsi_idx += 1;
            }
            if (orderbook_idx < MAX_SYMBOLS) {
                orderbook_symbol_names_temp[orderbook_idx] = entry.key_ptr.*;
                orderbook_symbols_slice_temp[orderbook_idx] = entry.value_ptr.*;
                orderbook_idx += 1;
            }
            all_idx += 1;
        }
        mutex.unlock();

        const num_rsi_symbols_to_process = rsi_idx;
        const num_orderbook_symbols_to_process = @min(all_idx, MAX_SYMBOLS);

        const rsi_results = try self.calculateRSIBatch(rsi_symbols_slice_temp[0..num_rsi_symbols_to_process], rsi_period);

        const orderbook_results = try self.calculateOrderBookPercentageBatch(symbols_slice[0..num_orderbook_symbols_to_process]);

        // for (0..num_rsi_symbols_to_process) |i| {
        //     const name = rsi_symbol_names_temp[i];
        //     std.log.info("Symbol '{s}': RSI={d:.4}", .{ name, rsi_results.rsi_values });
        //     if (orderbook_results.bid_percentage[i] != 0.0 or orderbook_results.ask_percentage[i] != 0.0) {
        //         std.log.info("Symbol '{s}': Bid%={d:.2}, Ask%={d:.2}, BidVol={d:.2}, AskVol={d:.2}", .{ name, orderbook_results.bid_percentage[i], orderbook_results.ask_percentage[i], orderbook_results.total_bid_volume[i], orderbook_results.total_ask_volume[i] });
        //     }
        // }

        std.log.info("Batch processing completed. StochRSI: {} symbols, OrderBook: {} symbols", .{ num_rsi_symbols_to_process, num_orderbook_symbols_to_process });
        return GPUBatchResult{
            .rsi = rsi_results,
            .orderbook = orderbook_results,
        };
    }

    fn calculateRSIBatch(self: *StatCalc, symbols: []const Symbol, rsi_period_u32: u32) !GPURSIResultBatch {
        const num_symbols = @min(symbols.len, MAX_SYMBOLS);
        if (num_symbols == 0) return self.h_rsi_result;

        const rsi_period: c_int = @intCast(rsi_period_u32);
        //const stoch_period: c_int = @intCast(stoch_period_u32);
        const num_symbols_c: c_int = @intCast(num_symbols);

        // Step 1: Prepare OHLC data and calculate RSI
        var h_ohlc_batch_zig = GPUOHLCDataBatch{
            .close_prices = [_][15]f32{[_]f32{0.0} ** 15} ** MAX_SYMBOLS,
            .counts = [_]u32{0} ** MAX_SYMBOLS,
        };

        for (0..num_symbols) |i| {
            h_ohlc_batch_zig.counts[i] = @intCast(symbols[i].count);
            var data_idx: usize = 0;
            var circ_buffer_start_idx = symbols[i].head;
            if (symbols[i].count < 15) {
                circ_buffer_start_idx = 0;
            } else {
                circ_buffer_start_idx = symbols[i].head;
            }

            for (0..symbols[i].count) |j| {
                if (data_idx >= 15) break;
                const current_ohlc_idx = (circ_buffer_start_idx + j) % 15;
                h_ohlc_batch_zig.close_prices[i][data_idx] = @floatCast(symbols[i].ticker_queue[current_ohlc_idx].close_price);
                data_idx += 1;
            }
        }

        // initialize h_rsi_result before passing its address
        self.h_rsi_result = std.mem.zeroes(GPURSIResultBatch);

        const kerr = cuda_wrapper_run_rsi_batch(
            self.d_ohlc_batch.?,
            self.d_rsi_result.?,
            &h_ohlc_batch_zig,
            &self.h_rsi_result,
            num_symbols_c,
            rsi_period,
        );

        if (kerr.code != 0) {
            std.log.err("RSI kernel execution failed via wrapper: {} ({s})", .{ kerr.code, kerr.message });
            return StatCalcError.CUDAKernelExecutionFailed;
        }

        // Step 2: Calculate StochRSI using the RSI results from self.h_rsi_result
        // self.h_rsi_result is already populated by the previous call (D2H copy in cuda_wrapper_run_rsi_batch)
        // initialize h_stoch_result before passing its address
        // self.h_stoch_result = std.mem.zeroes(GPUStochRSIResultBatch);

        // kerr = cuda_wrapper_run_stoch_rsi_batch(
        //     self.d_rsi_result.?,
        //     self.d_stoch_result.?,
        //     &self.h_rsi_result,
        //     &self.h_stoch_result,
        //     num_symbols_c,
        //     stoch_period,
        // );

        // if (kerr.code != 0) {
        //     std.log.err("StochRSI kernel execution failed via wrapper: {} ({s})", .{ kerr.code, kerr.message });
        //     return StatCalcError.CUDAKernelExecutionFailed;
        // }

        return self.h_rsi_result;
    }

    fn calculateOrderBookPercentageBatch(self: *StatCalc, symbols: []const Symbol) !GPUOrderBookResultBatch {
        const num_symbols = @min(symbols.len, MAX_SYMBOLS);
        if (num_symbols == 0) return self.h_orderbook_result;

        var h_orderbook_batch_zig = GPUOrderBookDataBatch{
            .bid_prices = [_][10]f32{[_]f32{0.0} ** 10} ** MAX_SYMBOLS,
            .bid_quantities = [_][10]f32{[_]f32{0.0} ** 10} ** MAX_SYMBOLS,
            .ask_prices = [_][10]f32{[_]f32{0.0} ** 10} ** MAX_SYMBOLS,
            .ask_quantities = [_][10]f32{[_]f32{0.0} ** 10} ** MAX_SYMBOLS,
            .bid_counts = [_]u32{0} ** MAX_SYMBOLS,
            .ask_counts = [_]u32{0} ** MAX_SYMBOLS,
        };

        for (0..num_symbols) |i| {
            const orderbook = &symbols[i].orderbook;
            h_orderbook_batch_zig.bid_counts[i] = @intCast(orderbook.bid_count);
            h_orderbook_batch_zig.ask_counts[i] = @intCast(orderbook.ask_count);

            var data_idx_bid: usize = 0;
            var circ_idx_bid = orderbook.bid_head;
            if (orderbook.bid_count < 10) circ_idx_bid = 0;
            for (0..orderbook.bid_count) |j| {
                if (data_idx_bid >= 10) break;
                const current_bid_idx = (circ_idx_bid + j) % 10;
                h_orderbook_batch_zig.bid_prices[i][data_idx_bid] = @floatCast(orderbook.bids[current_bid_idx].price);
                h_orderbook_batch_zig.bid_quantities[i][data_idx_bid] = @floatCast(orderbook.bids[current_bid_idx].quantity);
                data_idx_bid += 1;
            }

            var data_idx_ask: usize = 0;
            var circ_idx_ask = orderbook.ask_head;
            if (orderbook.ask_count < 10) circ_idx_ask = 0;
            for (0..orderbook.ask_count) |j| {
                if (data_idx_ask >= 10) break;
                const current_ask_idx = (circ_idx_ask + j) % 10;
                h_orderbook_batch_zig.ask_prices[i][data_idx_ask] = @floatCast(orderbook.asks[current_ask_idx].price);
                h_orderbook_batch_zig.ask_quantities[i][data_idx_ask] = @floatCast(orderbook.asks[current_ask_idx].quantity);
                data_idx_ask += 1;
            }
        }

        // Initialize h_orderbook_result before passing its address
        self.h_orderbook_result = std.mem.zeroes(GPUOrderBookResultBatch);

        const kerr = cuda_wrapper_run_orderbook_batch(self.d_orderbook_batch.?, self.d_orderbook_result.?, &h_orderbook_batch_zig, &self.h_orderbook_result, @intCast(num_symbols));

        if (kerr.code != 0) {
            std.log.err("Orderbook kernel execution failed via wrapper: {} ({s})", .{ kerr.code, kerr.message });
            return StatCalcError.CUDAKernelExecutionFailed;
        }

        return self.h_orderbook_result;
    }

    pub fn getDeviceInfo(self: *StatCalc) !void {
        var info: DeviceInfo = undefined;
        const kerr = cuda_wrapper_get_device_info(self.device_id, &info);
        if (kerr.code != 0) {
            std.log.err("Failed to get device info via wrapper: {} ({s})", .{ kerr.code, kerr.message });
            return StatCalcError.CUDAGetPropertiesFailed;
        }

        std.log.info("=== CUDA Device Information ===", .{});
        std.log.info("Device Name: {s}", .{info.name});
        std.log.info("Compute Capability: {}.{}", .{ info.major, info.minor });
        std.log.info("Total Global Memory: {} MB", .{@divTrunc(info.totalGlobalMem, 1024 * 1024)});
        std.log.info("==============================", .{});
    }

    pub fn warmUp(self: *StatCalc) !void {
        var dummy_symbol = Symbol.init();
        const dummy_ohlc = OHLC{
            .open_price = 100.0,
            .high_price = 105.0,
            .low_price = 99.0,
            .close_price = 103.0,
            .volume = 1000.0,
        };

        for (0..15) |_| {
            dummy_symbol.addTicker(dummy_ohlc);
        }
        std.debug.assert(dummy_symbol.count == 15);

        for (0..5) |i| {
            dummy_symbol.orderbook.updateLevel(102.5 - @as(f64, @floatFromInt(i)) * 0.1, 100.0 + @as(f64, @floatFromInt(i)) * 10.0, true);
            dummy_symbol.orderbook.updateLevel(103.0 + @as(f64, @floatFromInt(i)) * 0.1, 120.0 + @as(f64, @floatFromInt(i)) * 10.0, false);
        }

        var symbols_slice = [_]Symbol{dummy_symbol};
        _ = try self.calculateRSIBatch(&symbols_slice, 14);
        _ = try self.calculateOrderBookPercentageBatch(&symbols_slice);

        std.log.info("CUDA warm-up completed", .{});
    }

    pub fn deinit(self: *StatCalc) void {
        const kerr = cuda_wrapper_free_memory(
            self.d_ohlc_batch,
            self.d_orderbook_batch,
            self.d_rsi_result,
            self.d_orderbook_result,
        );
        if (kerr.code != 0) {
            std.log.err("CUDA free memory failed via wrapper: {} ({s})", .{ kerr.code, kerr.message });
        }

        self.d_ohlc_batch = null;
        self.d_orderbook_batch = null;
        self.d_rsi_result = null;
        self.d_orderbook_result = null;

        const kerr_reset = cuda_wrapper_reset_device();
        if (kerr_reset.code != 0) {
            std.log.err("CUDA device reset failed via wrapper: {} ({s})", .{ kerr_reset.code, kerr_reset.message });
        }
    }
};

pub fn getCUDADeviceCount() !c_int {
    var device_count: c_int = 0;
    const kerr = cuda_wrapper_get_device_count(&device_count);
    if (kerr.code != 0) {
        std.log.err("Failed to get CUDA device count via wrapper: {} ({s})", .{ kerr.code, kerr.message });
        return StatCalcError.CUDAGetDeviceCountFailed;
    }
    return device_count;
}

pub fn selectBestCUDADevice() !c_int {
    var best_device: c_int = 0;
    const kerr = cuda_wrapper_select_best_device(&best_device);

    if (kerr.code == ERR.KERNEL_ERROR_NO_DEVICE.code) {
        return StatCalcError.NoCUDADevicesFound;
    }
    if (kerr.code != 0) {
        std.log.err("Failed to select best CUDA device via wrapper: {} ({s})", .{ kerr.code, kerr.message });
        return StatCalcError.CUDAGetDeviceCountFailed;
    }
    return best_device;
}
