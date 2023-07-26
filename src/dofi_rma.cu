extern "C" {
#include "dofi.h"
}
#include "rmem_utils.h"

#if (M_HAVE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void ofi_device_inc_ready(ofi_drma_t drma) { m_rmem_trigger(drma); }

extern "C" int ofi_rma_start_device(rmem_stream_t* stream, ofi_drma_t drma) {
    // m_log("launching kernel with request %p\n",drma);
    ofi_device_inc_ready<<<1, 1, 0, *stream>>>(drma);
    // m_log("cuda has started");
    return CUDA_SUCCESS;
}
#endif
