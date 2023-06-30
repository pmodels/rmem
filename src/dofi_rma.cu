#include <cuda.h>
#include <cuda_runtime.h>
#include "ofi.cuh"
#include "rmem_utils.h"

__global__ void ofi_device_inc_ready(ofi_drma_t* drma) {
    drma->ready[0]++;
}

extern "C" int ofi_rma_start(ofi_drma_t* drma) {
    m_log("launching kernel");
#if (M_HAVE_CUDA)
    ofi_device_inc_ready<<<1, 1>>>(drma);
#else
    ofi_device_inc_ready(drma);
#endif
    return CUDA_SUCCESS;
}
