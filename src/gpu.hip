/*
 * Copyright (c) 2024, UChicago Argonne, LLC
 *	See COPYRIGHT in top-level directory
 */
#include "hip/hip_runtime.h"
extern "C" {
#include "gpu.h"
}
#include "rmem_utils.h"

__global__ void ofi_device_inc_ready(rmem_trigr_ptr trigr) { m_rmem_trigger(trigr); }

extern "C" int ofi_rma_start_hip(hipStream_t* stream, rmem_trigr_ptr trigr) {
    hipLaunchKernelGGL(ofi_device_inc_ready, 1, 1, 0, *stream, trigr);
    return hipSuccess;
}
