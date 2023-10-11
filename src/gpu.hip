#include "hip/hip_runtime.h"
extern "C" {
#include "gpu.h"
}
#include "rmem_utils.h"

__global__ void ofi_device_inc_ready(ofi_drma_t drma) { m_rmem_trigger(drma); }

extern "C" int ofi_rma_start_hip(hipStream_t* stream, ofi_drma_t drma) {
    hipLaunchKernelGGL(ofi_device_inc_ready, 1, 1, 0, *stream, drma);
    return hipSuccess;
}
