const std = @import("std");

pub const Dump = error{
    OutOfMemory,
    InvalidSymbol,
    InvalidOrder,
    OrderNotFound,
    OrderBookNotFound,
    OrderBookFull,
    OrderBookEmpty,
    InvalidTrade,
    TradeNotFound,
    TradeBookNotFound,
    TradeBookFull,
    TradeBookEmpty,
    InvalidMarketData,
    MarketDataNotFound,
    MarketDataFull,
    MarketDataEmpty,
    NetworkError,
    BadResponse,
    Timeout,
    AuthenticationError,
    RateLimitExceeded,

    InvalidBidFormat,
    InvalidAskFormat,
    DepthRequestFailed,
    ExchangeInfoRequestFailed,
    InvalidDepthEvent,
};

pub const DepthError = error{
    DepthRequestFailed,
    OutOfMemory,
} || std.json.Error || std.http.Client.RequestError || std.Uri.ParseError;

pub const StatCalcError = error{
    CUDAInitFailed,
    CUDAMemoryAllocationFailed,
    CUDAMemcpyFailed,
    CUDAKernelLaunchFailed,
    CUDAKernelExecutionFailed,
    CUDAGetPropertiesFailed,
    CUDAGetDeviceCountFailed,
    NoCUDADevicesFound,
    CUDADeviceResetFailed,
    CUDAFreeMemoryFailed,
};

pub const KernelError = extern struct {
    code: c_int,
    message: [*:0]const u8,
};

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
