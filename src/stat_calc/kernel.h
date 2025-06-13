#ifndef KERNEL_H
#define KERNEL_H

#define MAX_SYMBOLS_CUDA 404
#define MAX_ORDERBOOK_SIZE 5

typedef struct {
    int code;
    const char* message;
} KernelError;

static const KernelError KERNEL_SUCCESS = {0, "Success"};
static const KernelError KERNEL_ERROR_INVALID_DEVICE = {1, "Invalid device ID"};
static const KernelError KERNEL_ERROR_NO_DEVICE = {2, "No CUDA devices found"};
static const KernelError KERNEL_ERROR_MEMORY_ALLOCATION = {3, "Memory allocation failed"};
static const KernelError KERNEL_ERROR_MEMORY_SET = {4, "Memory set failed"};
static const KernelError KERNEL_ERROR_MEMORY_FREE = {5, "Memory free failed"};
static const KernelError KERNEL_ERROR_MEMCPY = {6, "Memory copy failed"};
static const KernelError KERNEL_ERROR_KERNEL_LAUNCH = {7, "Kernel launch failed"};
static const KernelError KERNEL_ERROR_KERNEL_EXECUTION = {8, "Kernel execution failed"};
static const KernelError KERNEL_ERROR_DEVICE_RESET = {9, "Device reset failed"};
static const KernelError KERNEL_ERROR_GET_PROPERTIES = {10, "Failed to get device properties"};
static const KernelError KERNEL_ERROR_GET_DEVICE_COUNT = {11, "Failed to get device count"};

typedef struct {
    char name[256];
    int major;
    int minor;
    size_t totalGlobalMem;
} DeviceInfo;

struct GPUOHLCDataBatch_C {
    float close_prices[MAX_SYMBOLS_CUDA][15];
    unsigned int counts[MAX_SYMBOLS_CUDA];
};

struct GPUOrderBookDataBatch_C {
    float bid_prices[MAX_SYMBOLS_CUDA][MAX_ORDERBOOK_SIZE];
    float bid_quantities[MAX_SYMBOLS_CUDA][MAX_ORDERBOOK_SIZE];
    float ask_prices[MAX_SYMBOLS_CUDA][MAX_ORDERBOOK_SIZE];
    float ask_quantities[MAX_SYMBOLS_CUDA][MAX_ORDERBOOK_SIZE];
    unsigned int bid_counts[MAX_SYMBOLS_CUDA];
    unsigned int ask_counts[MAX_SYMBOLS_CUDA];
};

struct GPURSIResultBatch_C {
    float rsi_values[MAX_SYMBOLS_CUDA][15];
    unsigned int valid_rsi_count[MAX_SYMBOLS_CUDA];
};

// struct GPUStochRSIResultBatch_C {
//     float stoch_rsi_k[MAX_SYMBOLS_CUDA];
//     float stoch_rsi_d[MAX_SYMBOLS_CUDA];
//     float rsi[MAX_SYMBOLS_CUDA];
// };

struct GPUOrderBookResultBatch_C {
    float bid_percentage[MAX_SYMBOLS_CUDA];
    float ask_percentage[MAX_SYMBOLS_CUDA]; 
    float total_bid_volume[MAX_SYMBOLS_CUDA];
    float total_ask_volume[MAX_SYMBOLS_CUDA];
    float spread_percentage[MAX_SYMBOLS_CUDA];
};

extern "C" {
    KernelError cuda_wrapper_init_device(int device_id);
    KernelError cuda_wrapper_reset_device();
    KernelError cuda_wrapper_get_device_count(int* count);
    KernelError cuda_wrapper_get_device_info(int device_id, DeviceInfo* info);
    KernelError cuda_wrapper_select_best_device(int* best_device_id);
    
    KernelError cuda_wrapper_allocate_memory(
        struct GPUOHLCDataBatch_C **d_ohlc_batch,
        struct GPUOrderBookDataBatch_C **d_orderbook_batch,
        struct GPURSIResultBatch_C **d_rsi_result,
        //struct GPUStochRSIResultBatch_C** d_stoch_result,
        struct GPUOrderBookResultBatch_C **d_orderbook_result
    );
    
    KernelError cuda_wrapper_free_memory(
        struct GPUOHLCDataBatch_C *d_ohlc_batch,
        struct GPUOrderBookDataBatch_C *d_orderbook_batch,
        struct GPURSIResultBatch_C *d_rsi_result,
        //struct GPUStochRSIResultBatch_C* d_stoch_result,
        struct GPUOrderBookResultBatch_C *d_orderbook_result
    );
    
    KernelError cuda_wrapper_run_rsi_batch(
        struct GPUOHLCDataBatch_C *d_ohlc_batch_ptr,
        struct GPURSIResultBatch_C *d_rsi_results_ptr,
        const struct GPUOHLCDataBatch_C *h_ohlc_batch,
        struct GPURSIResultBatch_C *h_rsi_results,
        int num_symbols,
        int rsi_period
    );
    
    // KernelError cuda_wrapper_run_stoch_rsi_batch(
    //     struct GPURSIResultBatch_C* d_rsi_results_ptr,
    //     struct GPUStochRSIResultBatch_C* d_stoch_results_ptr,
    //     const struct GPURSIResultBatch_C* h_rsi_results,
    //     struct GPUStochRSIResultBatch_C* h_stoch_results,
    //     int num_symbols,
    //     int stoch_period
    // );
    
    KernelError cuda_wrapper_run_orderbook_batch(
        struct GPUOrderBookDataBatch_C *d_orderbook_batch_ptr,
        struct GPUOrderBookResultBatch_C *d_results_ptr,
        const struct GPUOrderBookDataBatch_C *h_orderbook_batch,
        struct GPUOrderBookResultBatch_C *h_results,
        int num_symbols
    );
}

#endif
