extern "C" {
#include "rmem_run_gpu.h"
}

__global__ void cuda_trigger_op_device(int* data, const size_t len, const size_t start_id,
                                       rmem_trigr_ptr* trigr, rmem_gpu_op_t op) {
    for (size_t j = 0; j < m_min(len, RMEM_KERNEL_NOP); ++j) {
        data[j] = start_id + j;
    }
    if (op == RMEM_GPU_PUT) {
        m_rmem_trigger(*trigr);
    }
}

extern "C" void cuda_trigger_op(rmem_gpu_op_t op, const size_t start, const size_t n_msg, int* data,
                                const size_t len, rmem_trigr_ptr* trigr, gpuStream_t stream) {
    for (int ii = 0; ii < n_msg; ++ii) {
        const int id = (start + ii) % n_msg;
        rmem_trigr_ptr* curr_trigr = (trigr) ? (trigr + id) : NULL;
        cuda_trigger_op_device<<<1, 1, 0, stream>>>(data + id * len, len, 1 + id * len, curr_trigr,
                                                    op);
    }
}
