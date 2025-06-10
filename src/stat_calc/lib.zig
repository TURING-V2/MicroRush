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
        var d_ohlc_batch_ptr: ?*GPUOHLCDataBatch = null;
        var d_orderbook_batch_ptr: ?*GPUOrderBookDataBatch = null;
        var d_rsi_result_ptr: ?*GPURSIResultBatch = null;
        var d_orderbook_result_ptr: ?*GPUOrderBookResultBatch = null;

        std.log.info("Attempting to allocate GPU memory...", .{});

        const kerr = cuda_wrapper_allocate_memory(
            @ptrCast(&d_ohlc_batch_ptr),
            @ptrCast(&d_orderbook_batch_ptr),
            @ptrCast(&d_rsi_result_ptr),
            @ptrCast(&d_orderbook_result_ptr),
        );

        if (kerr.code != 0) {
            std.log.err("CUDA memory allocation failed via wrapper: {} ({s})", .{ kerr.code, kerr.message });
            return StatCalcError.CUDAMemoryAllocationFailed;
        }

        if (d_ohlc_batch_ptr == null) {
            std.log.err("d_ohlc_batch_ptr is null after allocation", .{});
            return StatCalcError.CUDAMemoryAllocationFailed;
        }

        if (d_orderbook_batch_ptr == null) {
            std.log.err("d_orderbook_batch_ptr is null after allocation", .{});
            return StatCalcError.CUDAMemoryAllocationFailed;
        }

        if (d_rsi_result_ptr == null) {
            std.log.err("d_rsi_result_ptr is null after allocation", .{});
            return StatCalcError.CUDAMemoryAllocationFailed;
        }

        if (d_orderbook_result_ptr == null) {
            std.log.err("d_orderbook_result_ptr is null after allocation", .{});
            return StatCalcError.CUDAMemoryAllocationFailed;
        }

        self.d_ohlc_batch = d_ohlc_batch_ptr;
        self.d_orderbook_batch = d_orderbook_batch_ptr;
        self.d_rsi_result = d_rsi_result_ptr;
        self.d_orderbook_result = d_orderbook_result_ptr;

        std.log.info("GPU memory allocation successful", .{});
        std.log.info("  d_ohlc_batch: 0x{x}", .{@intFromPtr(self.d_ohlc_batch.?)});
        std.log.info("  d_orderbook_batch: 0x{x}", .{@intFromPtr(self.d_orderbook_batch.?)});
        std.log.info("  d_rsi_result: 0x{x}", .{@intFromPtr(self.d_rsi_result.?)});
        std.log.info("  d_orderbook_result: 0x{x}", .{@intFromPtr(self.d_orderbook_result.?)});
    }

    pub fn calculateSymbolMapBatch(self: *StatCalc, symbol_map: *const SymbolMap, rsi_period: u32) !GPUBatchResult {
        const symbol_count = symbol_map.count();
        if (symbol_count == 0) {
            std.log.warn("SymbolMap is empty, nothing to calculate", .{});
            return ERR.Dump.MarketDataEmpty;
        }

        var valid_symbol_count: usize = 0;
        var iterator = symbol_map.iterator();
        while (iterator.next()) |entry| {
            if (entry.value_ptr.*.count == 15) {
                valid_symbol_count += 1;
            }
        }

        if (valid_symbol_count == 0) {
            std.log.warn("No symbols have count == 15 for OHLC, RSI calculation will likely yield no/default results for many.", .{});
        }

        const max_symbols_to_process = @min(symbol_count, MAX_SYMBOLS);
        const max_rsi_symbols_to_process = @min(valid_symbol_count, MAX_SYMBOLS);

        //std.log.info("Processing up to {} symbols for RSI and up to {} for OrderBook in batch mode...", .{ max_rsi_symbols_to_process, max_symbols_to_process });

        var symbols_slice = try self.allocator.alloc(Symbol, max_symbols_to_process);
        defer self.allocator.free(symbols_slice);
        var symbol_names = try self.allocator.alloc([]const u8, max_symbols_to_process);
        defer self.allocator.free(symbol_names);

        var rsi_symbols_slice_temp = try self.allocator.alloc(Symbol, max_rsi_symbols_to_process);
        defer self.allocator.free(rsi_symbols_slice_temp);
        var rsi_symbol_names_temp = try self.allocator.alloc([]const u8, max_rsi_symbols_to_process);
        defer self.allocator.free(rsi_symbol_names_temp);

        iterator = symbol_map.iterator();
        var rsi_idx: usize = 0;
        var all_idx: usize = 0;

        while (iterator.next()) |entry| {
            if (all_idx >= max_symbols_to_process and rsi_idx >= max_rsi_symbols_to_process) {
                break;
            }
            if (all_idx < max_symbols_to_process) {
                symbols_slice[all_idx] = entry.value_ptr.*;
                symbol_names[all_idx] = entry.key_ptr.*;
                all_idx += 1;
            }
            if (rsi_idx < max_rsi_symbols_to_process and entry.value_ptr.*.count == 15) {
                rsi_symbol_names_temp[rsi_idx] = entry.key_ptr.*;
                rsi_symbols_slice_temp[rsi_idx] = entry.value_ptr.*;
                rsi_idx += 1;
            }
        }

        if (symbol_count > MAX_SYMBOLS) {
            std.log.warn("Total symbols ({}) exceeds MAX_SYMBOLS ({}), processing only first {} symbols", .{ symbol_count, MAX_SYMBOLS, max_symbols_to_process });
        }
        if (valid_symbol_count > MAX_SYMBOLS) {
            std.log.warn("Valid RSI symbols ({}) exceeds MAX_SYMBOLS ({}), processing only first {} symbols", .{ valid_symbol_count, MAX_SYMBOLS, max_rsi_symbols_to_process });
        }

        const num_rsi_symbols_to_process = rsi_idx;
        const num_orderbook_symbols_to_process = all_idx;

        const rsi_results = if (num_rsi_symbols_to_process > 0)
            try self.calculateRSIBatch(rsi_symbols_slice_temp[0..num_rsi_symbols_to_process], rsi_period)
        else
            std.mem.zeroes(GPURSIResultBatch);

        const orderbook_results = if (num_orderbook_symbols_to_process > 0)
            try self.calculateOrderBookPercentageBatch(symbols_slice[0..num_orderbook_symbols_to_process])
        else
            std.mem.zeroes(GPUOrderBookResultBatch);

        //std.log.info("Batch processing completed. RSI: {} symbols, OrderBook: {} symbols", .{ num_rsi_symbols_to_process, num_orderbook_symbols_to_process });
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

        self.h_rsi_result = std.mem.zeroes(GPURSIResultBatch);

        if (self.d_ohlc_batch == null) {
            std.log.err("Failed to allocate memory for d_ohlc_batch", .{});
            return StatCalcError.CUDAMemoryAllocationFailed;
        }

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
        // TODO: add enough symbols to run RSI this won't execute
        _ = try self.calculateRSIBatch(&symbols_slice, 6);
        _ = try self.calculateOrderBookPercentageBatch(&symbols_slice);

        std.log.info("CUDA warm-up completed", .{});
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
