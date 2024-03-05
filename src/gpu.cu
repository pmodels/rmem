/*
 * Copyright (c) 2024, UChicago Argonne, LLC
 *	See COPYRIGHT in top-level directory
 */
extern "C" {
#include "gpu.h"
}
#include "rmem_utils.h"

__global__ void ofi_device_inc_ready(rmem_trigr_ptr trigr) { m_rmem_trigger(trigr); }

extern "C" int ofi_rma_start_cuda(cudaStream_t* stream, rmem_trigr_ptr trigr) {
    ofi_device_inc_ready<<<1, 1, 0, *stream>>>(trigr);
    return CUDA_SUCCESS;
}
