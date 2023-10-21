extern "C" {
#include "rmem_run_gpu.h"
}

__global__ void cuda_trigger_op_device_p2p(const size_t n_msg, int* data, const size_t len,
                                       ofi_drma_t* drma) {
    for (int i = 0; i < n_msg; ++i) {
        int* local_data = data + i * len;
        for (size_t j = 0; i < len; ++j) {
            local_data[j] = 1 + i * len + j;
        }
    }
}
__global__ void cuda_trigger_op_device_put(const size_t n_msg, int* data, const size_t len,
                                       ofi_drma_t* drma) {
    for (int i = 0; i < n_msg; ++i) {
        int* local_data = data + i * len;
        for (size_t j = 0; i < len; ++j) {
            local_data[j] = 1 + i * len + j;
        }
        m_rmem_trigger(drma[i]);
    }
}

extern "C" void cuda_trigger_op(const size_t n_msg, int* data, const size_t len, rmem_gpu_op_t op,
                     void* op_array, void* op_data, gpuStream_t stream) {
    switch (op) {
        case RMEM_GPU_PUT: {
            cuda_trigger_op_device_put<<<1, 1, 0, stream>>>(n_msg, data, len,(ofi_drma_t*)op_array);
        } break;
        case RMEM_GPU_P2P: {
            cuda_trigger_op_device_p2p<<<1, 1, 0, stream>>>(n_msg, data, len,(ofi_drma_t*)op_array);
        } break;
    }
}
