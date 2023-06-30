#ifndef OFI_CUH_
#define OFI_CUH_


#ifdef HAVE_LIBcuda
#define M_HAVE_CUDA 1
#else
#define M_HAVE_CUDA 1
#endif


// device version of the RMA request
typedef struct {
    volatile int* ready;
} ofi_drma_t;
#endif
