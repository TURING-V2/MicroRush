#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>
#include <cstring>

__global__ void stoch_rsi_kernel_batch(const GPUOHLCDataBatch_C *ohlc_batch, int num_symbols, int rsi_period, int stoch_period, GPUStochRSIResultBatch_C *results) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
    int count = ohlc_batch->counts[symbol_idx];
    if (rsi_period < 1) rsi_period = 1;
    int number_of_rsi = count - rsi_period + 1;
    if (number_of_rsi <= 0) {
        if (threadIdx.x == 0) {
            results->rsi[symbol_idx] = 50.0f;
            results->stoch_rsi_k[symbol_idx] = 50.0f;
            results->stoch_rsi_d[symbol_idx] = 50.0f;
        }
        return;
    }

    __shared__ float rsi_values[15];
    __shared__ float stoch_k[15];

    int idx = threadIdx.x;

    if (idx < number_of_rsi) {
        int k = rsi_period - 1 + idx;
        int start = k - rsi_period + 1;
        int end = k;

        float sum_gain = 0.0f;
        float sum_loss = 0.0f;

        if (rsi_period > 1) {
            for (int i = start + 1; i <= end; i++) {
                if (i < 15 && (i-1) < 15 && i >= 0 && (i-1) >= 0) {
                    float delta = ohlc_batch->close_prices[symbol_idx][i] - ohlc_batch->close_prices[symbol_idx][i - 1];
                    if (delta > 0) sum_gain += delta;
                    else if (delta < 0) sum_loss += -delta;
                }
            }
            float avg_gain = sum_gain / (float)(rsi_period - 1);
            float avg_loss = sum_loss / (float)(rsi_period - 1);
            float rs = (avg_loss > 0.000001f) ? (avg_gain / avg_loss) : (avg_gain > 0.000001f ? 1e10f : 1.0f);
            if (avg_loss == 0.0f && avg_gain == 0.0f) rs = 1.0f;
            rsi_values[idx] = 100.0f - (100.0f / (1.0f + rs));
        } else {
            rsi_values[idx] = 50.0f;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (stoch_period < 1) stoch_period = 1;

        for (int j = 0; j < number_of_rsi; j++) {
            int start_idx = fmaxf(0, j - stoch_period + 1);
            int end_idx = j;

            float min_rsi = rsi_values[start_idx];
            float max_rsi = rsi_values[start_idx];
            for (int m = start_idx; m <= end_idx; m++) {
                if (rsi_values[m] < min_rsi) min_rsi = rsi_values[m];
                if (rsi_values[m] > max_rsi) max_rsi = rsi_values[m];
            }

            float current_rsi = rsi_values[j];
            if (max_rsi > min_rsi) {
                stoch_k[j] = (current_rsi - min_rsi) / (max_rsi - min_rsi) * 100.0f;
            } else {
                stoch_k[j] = 50.0f;
            }
        }

        int last_rsi_idx = number_of_rsi - 1;
        if (last_rsi_idx >= 0) {
            results->rsi[symbol_idx] = rsi_values[last_rsi_idx];
            results->stoch_rsi_k[symbol_idx] = stoch_k[last_rsi_idx];

            int D_period = 3;
            float sum_k_for_d = 0.0f;
            int actual_d_count = 0;
            for (int p = 0; p < D_period; p++) {
                int k_idx_for_d = last_rsi_idx - p;
                if (k_idx_for_d >= 0) {
                    sum_k_for_d += stoch_k[k_idx_for_d];
                    actual_d_count++;
                } else {
                    break;
                }
            }
            if (actual_d_count > 0) {
                results->stoch_rsi_d[symbol_idx] = sum_k_for_d / (float)actual_d_count;
            } else {
                results->stoch_rsi_d[symbol_idx] = 50.0f;
            }
        } else {
            results->rsi[symbol_idx] = 50.0f;
            results->stoch_rsi_k[symbol_idx] = 50.0f;
            results->stoch_rsi_d[symbol_idx] = 50.0f;
        }
    }
}

__global__ void orderbook_kernel_batch(const GPUOrderBookDataBatch_C *orderbook_batch, int num_symbols, GPUOrderBookResultBatch_C *results) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;

    __shared__ float partial_bid_sums[32];
    __shared__ float partial_ask_sums[32];

    int idx = threadIdx.x;
    int bid_count = orderbook_batch->bid_counts[symbol_idx];
    int ask_count = orderbook_batch->ask_counts[symbol_idx];

    if (idx < 10) {
        if (idx < bid_count) {
            partial_bid_sums[idx] = orderbook_batch->bid_quantities[symbol_idx][idx];
        } else {
            partial_bid_sums[idx] = 0.0f;
        }

        if (idx < ask_count) {
            partial_ask_sums[idx] = orderbook_batch->ask_quantities[symbol_idx][idx];
        } else {
            partial_ask_sums[idx] = 0.0f;
        }
    } else {
        if (idx < 32) {
            partial_bid_sums[idx] = 0.0f;
            partial_ask_sums[idx] = 0.0f;
        }
    }
    __syncthreads();

    for (int stride = 16; stride > 0; stride >>= 1) {
        if (idx < stride) {
            partial_bid_sums[idx] += partial_bid_sums[idx + stride];
            partial_ask_sums[idx] += partial_ask_sums[idx + stride];
        }
        __syncthreads();
    }

    if (idx == 0) {
        float total_bid = partial_bid_sums[0];
        float total_ask = partial_ask_sums[0];
        results->total_bid_volume[symbol_idx] = total_bid;
        results->total_ask_volume[symbol_idx] = total_ask;

        float total_volume = total_bid + total_ask;
        if (total_volume > 0.000001f) {
            results->bid_percentage[symbol_idx] = (total_bid / total_volume) * 100.0f;
            results->ask_percentage[symbol_idx] = (total_ask / total_volume) * 100.0f;
        } else {
            results->bid_percentage[symbol_idx] = 50.0f;
            results->ask_percentage[symbol_idx] = 50.0f;
        }
    }
}

static KernelError map_cuda_error(cudaError_t cuda_err, const char* context) {
    if (cuda_err == cudaSuccess) {
        return KERNEL_SUCCESS;
    }
    static char error_msg[256];
    snprintf(error_msg, sizeof(error_msg), "%s: %s", context, cudaGetErrorString(cuda_err));
    return { cuda_err, error_msg };
}

static KernelError launch_stoch_rsi_kernel_internal(
    const GPUOHLCDataBatch_C *d_ohlc_batch,
    GPUStochRSIResultBatch_C *d_results,
    int num_symbols,
    int rsi_period,
    int stoch_period)
{
    const int THREADS_PER_BLOCK = 32;
    if (num_symbols > 0) {
        stoch_rsi_kernel_batch<<<num_symbols, THREADS_PER_BLOCK>>>(d_ohlc_batch, num_symbols, rsi_period, stoch_period, d_results);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA StochRSI kernel launch failed");
        }
    }
    return KERNEL_SUCCESS;
}

static KernelError launch_orderbook_kernel_internal(
    const GPUOrderBookDataBatch_C *d_orderbook_batch,
    GPUOrderBookResultBatch_C *d_results,
    int num_symbols)
{
    const int THREADS_PER_BLOCK = 32;
    if (num_symbols > 0) {
        orderbook_kernel_batch<<<num_symbols, THREADS_PER_BLOCK>>>(d_orderbook_batch, num_symbols, d_results);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA Orderbook kernel launch failed");
        }
    }
    return KERNEL_SUCCESS;
}

extern "C" {
    KernelError cuda_wrapper_init_device(int device_id) {
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            return map_cuda_error(err, "Failed to set CUDA device");
        }
        return KERNEL_SUCCESS;
    }

    KernelError cuda_wrapper_reset_device() {
        cudaError_t err = cudaDeviceReset();
        if (err != cudaSuccess) {
            return map_cuda_error(err, "Device reset failed");
        }
        return KERNEL_SUCCESS;
    }

    KernelError cuda_wrapper_get_device_count(int* count) {
        cudaError_t err = cudaGetDeviceCount(count);
        if (err != cudaSuccess) {
            return map_cuda_error(err, "Failed to get device count");
        }
        return KERNEL_SUCCESS;
    }

    KernelError cuda_wrapper_get_device_properties(int device_id, struct cudaDeviceProp* props) {
        cudaError_t err = cudaGetDeviceProperties(props, device_id);
        if (err != cudaSuccess) {
            return map_cuda_error(err, "Failed to get device properties");
        }
        return KERNEL_SUCCESS;
    }

    KernelError cuda_wrapper_select_best_device(int* best_device_id_out) {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            return map_cuda_error(err, "Failed to get device count");
        }
        if (device_count == 0) {
            return KERNEL_ERROR_NO_DEVICE;
        }

        int best_device = 0;
        int max_compute_capability = 0;

        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp props;
            err = cudaGetDeviceProperties(&props, i);
            if (err == cudaSuccess) {
                int current_compute_capability = props.major * 100 + props.minor;
                if (current_compute_capability > max_compute_capability) {
                    max_compute_capability = current_compute_capability;
                    best_device = i;
                }
            } else {
                return map_cuda_error(err, "Failed to get properties for device");
            }
        }
        *best_device_id_out = best_device;
        return KERNEL_SUCCESS;
    }

    KernelError cuda_wrapper_allocate_memory(
        GPUOHLCDataBatch_C** d_ohlc_batch,
        GPUOrderBookDataBatch_C** d_orderbook_batch,
        GPUStochRSIResultBatch_C** d_stoch_result,
        GPUOrderBookResultBatch_C** d_orderbook_result
    ) {
        cudaError_t err;

        err = cudaMalloc((void**)d_ohlc_batch, sizeof(GPUOHLCDataBatch_C));
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA Malloc failed for d_ohlc_batch");
        }
        err = cudaMemset(*d_ohlc_batch, 0, sizeof(GPUOHLCDataBatch_C));
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA Memset failed for d_ohlc_batch");
        }

        err = cudaMalloc((void**)d_orderbook_batch, sizeof(GPUOrderBookDataBatch_C));
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA Malloc failed for d_orderbook_batch");
        }
        err = cudaMemset(*d_orderbook_batch, 0, sizeof(GPUOrderBookDataBatch_C));
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA Memset failed for d_orderbook_batch");
        }

        err = cudaMalloc((void**)d_stoch_result, sizeof(GPUStochRSIResultBatch_C));
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA Malloc failed for d_stoch_result");
        }
        err = cudaMemset(*d_stoch_result, 0, sizeof(GPUStochRSIResultBatch_C));
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA Memset failed for d_stoch_result");
        }

        err = cudaMalloc((void**)d_orderbook_result, sizeof(GPUOrderBookResultBatch_C));
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA Malloc failed for d_orderbook_result");
        }
        err = cudaMemset(*d_orderbook_result, 0, sizeof(GPUOrderBookResultBatch_C));
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA Memset failed for d_orderbook_result");
        }

        return KERNEL_SUCCESS;
    }

    KernelError cuda_wrapper_free_memory(
        GPUOHLCDataBatch_C* d_ohlc_batch,
        GPUOrderBookDataBatch_C* d_orderbook_batch,
        GPUStochRSIResultBatch_C* d_stoch_result,
        GPUOrderBookResultBatch_C* d_orderbook_result
    ) {
        KernelError last_err = KERNEL_SUCCESS;
        cudaError_t current_err;

        if (d_ohlc_batch) {
            current_err = cudaFree(d_ohlc_batch);
            if (current_err != cudaSuccess) {
                last_err = map_cuda_error(current_err, "CUDA Free failed for d_ohlc_batch");
            }
        }
        if (d_orderbook_batch) {
            current_err = cudaFree(d_orderbook_batch);
            if (current_err != cudaSuccess && last_err.code == 0) {
                last_err = map_cuda_error(current_err, "CUDA Free failed for d_orderbook_batch");
            }
        }
        if (d_stoch_result) {
            current_err = cudaFree(d_stoch_result);
            if (current_err != cudaSuccess && last_err.code == 0) {
                last_err = map_cuda_error(current_err, "CUDA Free failed for d_stoch_result");
            }
        }
        if (d_orderbook_result) {
            current_err = cudaFree(d_orderbook_result);
            if (current_err != cudaSuccess && last_err.code == 0) {
                last_err = map_cuda_error(current_err, "CUDA Free failed for d_orderbook_result");
            }
        }
        return last_err;
    }

    KernelError cuda_wrapper_run_stoch_rsi_batch(
        GPUOHLCDataBatch_C* d_ohlc_batch_ptr,
        GPUStochRSIResultBatch_C* d_results_ptr,
        const GPUOHLCDataBatch_C* h_ohlc_batch,
        GPUStochRSIResultBatch_C* h_results,
        int num_symbols,
        int rsi_period,
        int stoch_period
    ) {
        if (num_symbols == 0) return KERNEL_SUCCESS;
        cudaError_t err;

        err = cudaMemcpy(d_ohlc_batch_ptr, h_ohlc_batch, sizeof(GPUOHLCDataBatch_C), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA Memcpy H2D failed for StochRSI input");
        }

        KernelError kerr = launch_stoch_rsi_kernel_internal(d_ohlc_batch_ptr, d_results_ptr, num_symbols, rsi_period, stoch_period);
        if (kerr.code != 0) {
            return kerr;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA StochRSI kernel execution failed");
        }

        err = cudaMemcpy(h_results, d_results_ptr, sizeof(GPUStochRSIResultBatch_C), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA Memcpy D2H failed for StochRSI results");
        }

        return KERNEL_SUCCESS;
    }

    KernelError cuda_wrapper_run_orderbook_batch(
        GPUOrderBookDataBatch_C* d_orderbook_batch_ptr,
        GPUOrderBookResultBatch_C* d_results_ptr,
        const GPUOrderBookDataBatch_C* h_orderbook_batch,
        GPUOrderBookResultBatch_C* h_results,
        int num_symbols
    ) {
        if (num_symbols == 0) return KERNEL_SUCCESS;
        cudaError_t err;

        err = cudaMemcpy(d_orderbook_batch_ptr, h_orderbook_batch, sizeof(GPUOrderBookDataBatch_C), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA Memcpy H2D failed for Orderbook input");
        }

        KernelError kerr = launch_orderbook_kernel_internal(d_orderbook_batch_ptr, d_results_ptr, num_symbols);
        if (kerr.code != 0) {
            return kerr;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA Orderbook kernel execution failed");
        }

        err = cudaMemcpy(h_results, d_results_ptr, sizeof(GPUOrderBookResultBatch_C), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return map_cuda_error(err, "CUDA Memcpy D2H failed for Orderbook results");
        }

        return KERNEL_SUCCESS;
    }
}
