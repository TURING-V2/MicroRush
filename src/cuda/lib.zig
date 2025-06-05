const std = @import("std");
const SymbolMap = @import("../types.zig").SymbolMap;
const Symbol = @import("../types.zig").Symbol;
const OHLC = @import("../types.zig").OHLC;
const StatCalcError = @import("../errors.zig").StatCalcError;

// Constants
const MAX_SYMBOLS_CUDA = 402;
const MAX_SYMBOLS = MAX_SYMBOLS_CUDA;

// Type definitions matching the C header
pub const KernelError = extern struct {
    code: c_int,
    message: [*:0]const u8,
};

// Predefined error constants (matching C header)
pub const KERNEL_SUCCESS = KernelError{ .code = 0, .message = "Success" };
pub const KERNEL_ERROR_INVALID_DEVICE = KernelError{ .code = 1, .message = "Invalid device ID" };
pub const KERNEL_ERROR_NO_DEVICE = KernelError{ .code = 2, .message = "No CUDA devices found" };
pub const KERNEL_ERROR_MEMORY_ALLOCATION = KernelError{ .code = 3, .message = "Memory allocation failed" };
pub const KERNEL_ERROR_MEMORY_SET = KernelError{ .code = 4, .message = "Memory set failed" };
pub const KERNEL_ERROR_MEMORY_FREE = KernelError{ .code = 5, .message = "Memory free failed" };
pub const KERNEL_ERROR_MEMCPY = KernelError{ .code = 6, .message = "Memory copy failed" };
pub const KERNEL_ERROR_KERNEL_LAUNCH = KernelError{ .code = 7, .message = "Kernel launch failed" };
pub const KERNEL_ERROR_KERNEL_EXECUTION = KernelError{ .code = 8, .message = "Kernel execution failed" };
pub const KERNEL_ERROR_DEVICE_RESET = KernelError{ .code = 9, .message = "Device reset failed" };
pub const KERNEL_ERROR_GET_PROPERTIES = KernelError{ .code = 10, .message = "Failed to get device properties" };
pub const KERNEL_ERROR_GET_DEVICE_COUNT = KernelError{ .code = 11, .message = "Failed to get device count" };

// CUDA device properties struct (simplified version)
pub const cudaDeviceProp = extern struct {
    name: [256]u8,
    major: c_int,
    minor: c_int,
    totalGlobalMem: usize,
    sharedMemPerBlock: usize,
    maxThreadsPerBlock: c_int,
    maxGridSize: [3]c_int,
    warpSize: c_int,
    memoryClockRate: c_int,
    memoryBusWidth: c_int,
    l2CacheSize: c_int,
};

pub const GPUOHLCDataBatch = extern struct {
    close_prices: [MAX_SYMBOLS][15]f32,
    counts: [MAX_SYMBOLS]u32,
};

pub const GPUOrderBookDataBatch = extern struct {
    bid_prices: [MAX_SYMBOLS][10]f32,
    bid_quantities: [MAX_SYMBOLS][10]f32,
    ask_prices: [MAX_SYMBOLS][10]f32,
    ask_quantities: [MAX_SYMBOLS][10]f32,
    bid_counts: [MAX_SYMBOLS]u32,
    ask_counts: [MAX_SYMBOLS]u32,
};

pub const GPUStochRSIResultBatch = extern struct {
    stoch_rsi_k: [MAX_SYMBOLS]f32,
    stoch_rsi_d: [MAX_SYMBOLS]f32,
    rsi: [MAX_SYMBOLS]f32,
};

pub const GPUOrderBookResultBatch = extern struct {
    bid_percentage: [MAX_SYMBOLS]f32,
    ask_percentage: [MAX_SYMBOLS]f32,
    total_bid_volume: [MAX_SYMBOLS]f32,
    total_ask_volume: [MAX_SYMBOLS]f32,
};

extern "c" fn cuda_wrapper_init_device(device_id: c_int) KernelError;
extern "c" fn cuda_wrapper_reset_device() KernelError;
extern "c" fn cuda_wrapper_get_device_count(count: *c_int) KernelError;
extern "c" fn cuda_wrapper_get_device_properties(device_id: c_int, props: *cudaDeviceProp) KernelError;
extern "c" fn cuda_wrapper_select_best_device(best_device_id: *c_int) KernelError;

extern "c" fn cuda_wrapper_allocate_memory(
    d_ohlc_batch: **GPUOHLCDataBatch,
    d_orderbook_batch: **GPUOrderBookDataBatch,
    d_stoch_result: **GPUStochRSIResultBatch,
    d_orderbook_result: **GPUOrderBookResultBatch,
) KernelError;

extern "c" fn cuda_wrapper_free_memory(
    d_ohlc_batch: ?*GPUOHLCDataBatch,
    d_orderbook_batch: ?*GPUOrderBookDataBatch,
    d_stoch_result: ?*GPUStochRSIResultBatch,
    d_orderbook_result: ?*GPUOrderBookResultBatch,
) KernelError;

extern "c" fn cuda_wrapper_run_stoch_rsi_batch(
    d_ohlc_batch_ptr: *GPUOHLCDataBatch,
    d_results_ptr: *GPUStochRSIResultBatch,
    h_ohlc_batch: *const GPUOHLCDataBatch,
    h_results: *GPUStochRSIResultBatch,
    num_symbols: c_int,
    rsi_period: c_int,
    stoch_period: c_int,
) KernelError;

extern "c" fn cuda_wrapper_run_orderbook_batch(
    d_orderbook_batch_ptr: *GPUOrderBookDataBatch,
    d_results_ptr: *GPUOrderBookResultBatch,
    h_orderbook_batch: *const GPUOrderBookDataBatch,
    h_results: *GPUOrderBookResultBatch,
    num_symbols: c_int,
) KernelError;

pub const StatCalc = struct {
    allocator: std.mem.Allocator,
    device_id: c_int,

    // Device pointers
    d_ohlc_batch: ?*GPUOHLCDataBatch,
    d_orderbook_batch: ?*GPUOrderBookDataBatch,
    d_stoch_result: ?*GPUStochRSIResultBatch,
    d_orderbook_result: ?*GPUOrderBookResultBatch,

    // Host copies of results
    h_stoch_result: GPUStochRSIResultBatch,
    h_orderbook_result: GPUOrderBookResultBatch,

    pub fn init(allocator: std.mem.Allocator, device_id: c_int) !StatCalc {
        var calc = StatCalc{
            .allocator = allocator,
            .device_id = device_id,
            .d_ohlc_batch = null,
            .d_orderbook_batch = null,
            .d_stoch_result = null,
            .d_orderbook_result = null,
            .h_stoch_result = std.mem.zeroes(GPUStochRSIResultBatch),
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

        var props: cudaDeviceProp = undefined;
        const kerr_props = cuda_wrapper_get_device_properties(self.device_id, &props);
        if (kerr_props.code != 0) {
            std.log.err("Failed to get device properties via wrapper: {} ({s})", .{ kerr_props.code, kerr_props.message });
            return StatCalcError.CUDAGetPropertiesFailed;
        }

        std.log.info("Using CUDA device: {s}", .{props.name});
        std.log.info("Compute capability: {}.{}", .{ props.major, props.minor });
        std.log.info("Global memory: {} MB", .{props.totalGlobalMem / (1024 * 1024)});
    }

    fn allocateDeviceMemory(self: *StatCalc) !void {
        // Ensure pointers are non-null or handle the optional case
        var d_ohlc_batch_ptr: *GPUOHLCDataBatch = undefined;
        var d_orderbook_batch_ptr: *GPUOrderBookDataBatch = undefined;
        var d_stoch_result_ptr: *GPUStochRSIResultBatch = undefined;
        var d_orderbook_result_ptr: *GPUOrderBookResultBatch = undefined;

        const kerr = cuda_wrapper_allocate_memory(
            &d_ohlc_batch_ptr,
            &d_orderbook_batch_ptr,
            &d_stoch_result_ptr,
            &d_orderbook_result_ptr,
        );

        if (kerr.code != 0) {
            std.log.err("CUDA memory allocation failed via wrapper: {} ({s})", .{ kerr.code, kerr.message });
            return StatCalcError.CUDAMemoryAllocationFailed;
        }

        // Assign the allocated pointers back to the struct fields
        self.d_ohlc_batch = d_ohlc_batch_ptr;
        self.d_orderbook_batch = d_orderbook_batch_ptr;
        self.d_stoch_result = d_stoch_result_ptr;
        self.d_orderbook_result = d_orderbook_result_ptr;
    }

    pub fn calculateSymbolMapBatch(self: *StatCalc, symbol_map: *const SymbolMap, rsi_period: u32, stoch_period: u32) !void {
        const symbol_count = symbol_map.count();
        if (symbol_count == 0) {
            std.log.warn("SymbolMap is empty, nothing to calculate", .{});
            return;
        }
        if (symbol_count > MAX_SYMBOLS) {
            std.log.warn("Symbol count {} exceeds MAX_SYMBOLS {}, truncating.", .{ symbol_count, MAX_SYMBOLS });
        }

        std.log.info("Processing {} symbols in batch mode...", .{symbol_count});

        var symbols_slice = try self.allocator.alloc(Symbol, symbol_count);
        defer self.allocator.free(symbols_slice);
        var symbol_names = try self.allocator.alloc([]const u8, symbol_count);
        defer self.allocator.free(symbol_names);

        var iterator = symbol_map.iterator();
        var index: usize = 0;
        while (iterator.next()) |entry| {
            if (index >= symbol_count) break;
            symbol_names[index] = entry.key_ptr.*;
            symbols_slice[index] = entry.value_ptr.*;
            index += 1;
        }

        const num_symbols_to_process = @min(symbol_count, MAX_SYMBOLS);

        const stoch_results = try self.calculateStochRSIBatch(symbols_slice[0..num_symbols_to_process], rsi_period, stoch_period);
        const orderbook_results = try self.calculateOrderBookPercentageBatch(symbols_slice[0..num_symbols_to_process]);

        for (0..num_symbols_to_process) |i| {
            const name = symbol_names[i];
            if (stoch_results.rsi[i] != 0.0 or stoch_results.stoch_rsi_k[i] != 0.0 or stoch_results.stoch_rsi_d[i] != 0.0) {
                std.log.info("Symbol '{s}': StochRSI K={d:.4}, D={d:.4}, RSI={d:.4}", .{ name, stoch_results.stoch_rsi_k[i], stoch_results.stoch_rsi_d[i], stoch_results.rsi[i] });
            }

            if (orderbook_results.bid_percentage[i] != 0.0 or orderbook_results.ask_percentage[i] != 0.0) {
                std.log.info("Symbol '{s}': Bid%={d:.2}, Ask%={d:.2}, BidVol={d:.2}, AskVol={d:.2}", .{ name, orderbook_results.bid_percentage[i], orderbook_results.ask_percentage[i], orderbook_results.total_bid_volume[i], orderbook_results.total_ask_volume[i] });
            }
        }

        std.log.info("Batch processing completed for {} symbols", .{num_symbols_to_process});
    }

    fn calculateStochRSIBatch(self: *StatCalc, symbols: []const Symbol, rsi_period: u32, stoch_period: u32) !GPUStochRSIResultBatch {
        const num_symbols = @min(symbols.len, MAX_SYMBOLS);
        if (num_symbols == 0) return self.h_stoch_result;

        var h_ohlc_batch_zig = GPUOHLCDataBatch{
            .close_prices = [_][15]f32{[_]f32{0.0} ** 15} ** MAX_SYMBOLS,
            .counts = [_]u32{0} ** MAX_SYMBOLS,
        };

        for (0..num_symbols) |i| {
            h_ohlc_batch_zig.counts[i] = @intCast(symbols[i].count);
            var data_idx: usize = 0;
            var circ_idx = symbols[i].head;
            if (symbols[i].count < 15) circ_idx = 0;

            for (0..symbols[i].count) |j| {
                if (data_idx >= 15) break;
                const current_ohlc_idx = (circ_idx + j) % 15;
                h_ohlc_batch_zig.close_prices[i][data_idx] = @floatCast(symbols[i].ticker_queue[current_ohlc_idx].close_price);
                data_idx += 1;
            }
        }

        const kerr = cuda_wrapper_run_stoch_rsi_batch(self.d_ohlc_batch.?, self.d_stoch_result.?, &h_ohlc_batch_zig, &self.h_stoch_result, @intCast(num_symbols), @intCast(rsi_period), @intCast(stoch_period));

        if (kerr.code != 0) {
            std.log.err("StochRSI kernel execution failed via wrapper: {} ({s})", .{ kerr.code, kerr.message });
            return StatCalcError.CUDAKernelExecutionFailed;
        }

        return self.h_stoch_result;
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

        const kerr = cuda_wrapper_run_orderbook_batch(self.d_orderbook_batch.?, self.d_orderbook_result.?, &h_orderbook_batch_zig, &self.h_orderbook_result, @intCast(num_symbols));

        if (kerr.code != 0) {
            std.log.err("Orderbook kernel execution failed via wrapper: {} ({s})", .{ kerr.code, kerr.message });
            return StatCalcError.CUDAKernelExecutionFailed;
        }

        return self.h_orderbook_result;
    }

    pub fn getDeviceInfo(self: *StatCalc) !void {
        var props: cudaDeviceProp = undefined;
        const kerr = cuda_wrapper_get_device_properties(self.device_id, &props);
        if (kerr.code != 0) {
            std.log.err("Failed to get device properties via wrapper: {} ({s})", .{ kerr.code, kerr.message });
            return StatCalcError.CUDAGetPropertiesFailed;
        }

        std.log.info("=== CUDA Device Information ===", .{});
        std.log.info("Device Name: {s}", .{props.name});
        std.log.info("Compute Capability: {}.{}", .{ props.major, props.minor });
        std.log.info("Total Global Memory: {} MB", .{@divTrunc(props.totalGlobalMem, 1024 * 1024)});
        std.log.info("Shared Memory per Block: {} KB", .{@divTrunc(props.sharedMemPerBlock, 1024)});
        std.log.info("Max Threads per Block: {}", .{props.maxThreadsPerBlock});
        std.log.info("Max Grid Size: [{}, {}, {}]", .{ props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2] });
        std.log.info("Warp Size: {}", .{props.warpSize});
        std.log.info("Memory Clock Rate: {} MHz", .{@divTrunc(props.memoryClockRate, 1000)});
        std.log.info("Memory Bus Width: {} bits", .{props.memoryBusWidth});
        std.log.info("L2 Cache Size: {} KB", .{@divTrunc(props.l2CacheSize, 1024)});
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

        for (0..14) |_| {
            dummy_symbol.addTicker(dummy_ohlc);
        }

        for (0..5) |i| {
            dummy_symbol.orderbook.updateLevel(102.5 - @as(f64, @floatFromInt(i)) * 0.1, 100.0 + @as(f64, @floatFromInt(i)) * 10.0, true);
            dummy_symbol.orderbook.updateLevel(103.0 + @as(f64, @floatFromInt(i)) * 0.1, 120.0 + @as(f64, @floatFromInt(i)) * 10.0, false);
        }

        var symbols_slice = [_]Symbol{dummy_symbol};
        _ = try self.calculateStochRSIBatch(&symbols_slice, 14, 3);
        _ = try self.calculateOrderBookPercentageBatch(&symbols_slice);

        std.log.info("CUDA warm-up completed", .{});
    }

    pub fn deinit(self: *StatCalc) void {
        const kerr = cuda_wrapper_free_memory(self.d_ohlc_batch, self.d_orderbook_batch, self.d_stoch_result, self.d_orderbook_result);
        if (kerr.code != 0) {
            std.log.err("CUDA free memory failed via wrapper: {} ({s})", .{ kerr.code, kerr.message });
        }

        self.d_ohlc_batch = null;
        self.d_orderbook_batch = null;
        self.d_stoch_result = null;
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

    if (kerr.code == KERNEL_ERROR_NO_DEVICE.code) {
        return StatCalcError.NoCUDADevicesFound;
    }
    if (kerr.code != 0) {
        std.log.err("Failed to select best CUDA device via wrapper: {} ({s})", .{ kerr.code, kerr.message });
        return StatCalcError.CUDAGetDeviceCountFailed;
    }
    return best_device;
}
