#ifndef RMEM_RUN_GPU_H__
#define RMEM_RUN_GPU_H__

#include <stddef.h>
#include "gpu.h"
#include "rmem_utils.h"

typedef enum {
    RMEM_GPU_P2P,
    RMEM_GPU_PUT,
} rmem_gpu_op_t;

#define RMEM_KERNEL_NOP 8

//--------------------------------------------------------------------------------------------------
#if (HAVE_CUDA)
extern void cuda_trigger_op(rmem_gpu_op_t op, const size_t n_msg, int* data, const size_t len,
                            rmem_trigr_ptr* trigr, gpuStream_t stream);
#define gpu_trigger_op cuda_trigger_op
//--------------------------------------------------------------------------------------------------
#elif (HAVE_HIP)
extern void hip_trigger_op(rmem_gpu_op_t op, const size_t n_msg, int* data, const size_t len,
                           rmem_trigr_ptr* trigr, gpuStream_t stream);
#define gpu_trigger_op hip_trigger_op
//--------------------------------------------------------------------------------------------------
#else
static void host_trigger_op(rmem_gpu_op_t op, const size_t n_msg, int* data, const size_t len,
                            rmem_trigr_ptr* trigr, gpuStream_t stream) {
    for (int i = 0; i < n_msg; ++i) {
        int* local_data = data + i * len;
        for (size_t j = 0; j < m_min(len, RMEM_KERNEL_NOP); ++j) {
            data[j + i * len] = 1 + i * len + j;
        }
        switch (op) {
            case RMEM_GPU_PUT: {
                m_verb("triggering %p",trigr[i]);
                m_rmem_trigger(trigr[i]);
            } break;
            case RMEM_GPU_P2P:
                break;
        }
    }
}
#define gpu_trigger_op host_trigger_op
#endif

#endif
