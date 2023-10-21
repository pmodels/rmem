#ifndef RMEM_RUN_GPU_H__
#define RMEM_RUN_GPU_H__

#include "gpu.h"

typedef enum {
    RMEM_GPU_P2P,
    RMEM_GPU_PUT,
} rmem_gpu_op_t;

//--------------------------------------------------------------------------------------------------
#if (HAVE_CUDA)
extern void cuda_trigger_op(const size_t n_msg, int* data, const size_t len, rmem_gpu_op_t op,
                     void* op_array, void* op_data, gpuStream_t stream);
#define gpu_trigger_op cuda_trigger_op
//--------------------------------------------------------------------------------------------------
#elif (HAVE_HIP)
extern void hip_trigger_op(const size_t n_msg, int* data, const size_t len, rmem_gpu_op_t op,
                    void* op_array, void* op_data, gpuStream_t stream);
#define gpu_trigger_op hip_trigger_op
//--------------------------------------------------------------------------------------------------
#else
static void host_trigger_op(const size_t n_msg, int* data, const size_t len, rmem_gpu_op_t op,
                            void* op_array, void* op_data, gpuStream_t stream) {
    for (int i = 0; i < n_msg; ++i) {
        int* local_data = data + i * len;
        for (size_t j = 0; j < len; ++j) {
            data[j + i * len] = 1 + i * len + j;
        }
        switch (op) {
            case RMEM_GPU_PUT: {
                ofi_rma_start((ofi_rmem_t*)op_data, ((ofi_rma_t*)op_array) + i, RMEM_TRIGGER);
            } break;
            case RMEM_GPU_P2P:
                break;
        }
    }
}
#define gpu_trigger_op host_trigger_op
#endif

#endif
