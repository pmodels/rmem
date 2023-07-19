#ifndef DOFI_H_
#define DOFI_H_

//--------------------------------------------------------------------------------------------------
#ifdef HAVE_CUDA
#define M_HAVE_CUDA 1
//--------------------------------------------------------------------------------------------------
#include <cuda.h>
#include "cuda_runtime_api.h"

typedef cudaStream_t rmem_stream_t;

#ifndef NDEBUG
#define m_cuda_call(func)                                                            \
    do {                                                                             \
        int m_cuda_call_res = func;                                                  \
        m_assert(m_cuda_call_res == cudaSuccess, "CUDA ERROR: %d", m_cuda_call_res); \
    } while (0)
#else
#define m_cuda_call(func) \
    do {                  \
        func;             \
    } while (0)
#endif

//--------------------------------------------------------------------------------------------------
#else
#define M_HAVE_CUDA 0
//--------------------------------------------------------------------------------------------------
typedef void* rmem_stream_t;
#endif

//--------------------------------------------------------------------------------------------------
// device version of the RMA request
struct ofi_device_rma_t {
    volatile int* ready;
};
typedef struct ofi_device_rma_t* ofi_drma_t;

extern int ofi_rma_start_device(rmem_stream_t* stream, ofi_drma_t drma);

#endif
