#ifndef OFI_CUH_
#define OFI_CUH_

#include <cuda.h>

#ifdef HAVE_LIBcuda
#define M_HAVE_CUDA 1
#else
#define M_HAVE_CUDA 1
#endif


// device version of the RMA request
typedef struct {
    volatile int* ready;
} ofi_drma_t;



//------------------------------------------------------------------------------
#ifndef NDEBUG
#define m_cuda_call(func)                                                          \
    do {                                                                          \
        int m_cuda_call_res = func;                                                \
        m_assert(m_cuda_call_res == cudaSuccess, "CUDA ERROR: %d", m_cuda_call_res); \
    } while (0)
#else
#define m_cuda_call(func)                                                          \
    do {                 \
        func;            \
    } while (0)
#endif

#endif
