#include <cuda.h>
#include <cuda_runtime.h>
extern "C" {
#include "dofi.h"
}
#include "rmem_utils.h"

__global__ void ofi_device_inc_ready(ofi_drma_t drma) {
    // printf("triggering from the GPU, with %p\n",drma);
    // printf("triggering from the GPU, address = %p\n",drma->ready);
    drma->ready[0]++;
}

extern "C" int ofi_rma_start_device(ofi_drma_t drma) {
#if (M_HAVE_CUDA)
    // m_log("launching kernel with request %p\n",drma);
    ofi_device_inc_ready<<<1, 1>>>(drma);
    cudaDeviceSynchronize();
    // m_log("cuda has started");
#endif
    return CUDA_SUCCESS;
}
