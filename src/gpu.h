#ifndef GPU_H_
#define GPU_H_

//==================================================================================================
// define stuff to be overwritten bellow by cuda or hip
#define GPU_DEFAULT_STREAM 0

//--------------------------------------------------------------------------------------------------
#if defined(HAVE_CUDA) || defined(HAVE_HIP)
//--------------------------------------------------------------------------------------------------
#define M_HAVE_GPU 1
#define m_gpu(f)   f

//--------------------------------------------------
#ifdef HAVE_CUDA
//--------------------------------------------------
#include <cuda.h>
#include <cuda_runtime.h>

#define gpuSuccess  cudaSuccess
#define gpuStream_t cudaStream_t

// memory
#define gpuMalloc               cudaMalloc
#define gpuFree                 cudaFree
#define gpuHostRegister         cudaHostRegister
#define gpuHostUnregister       cudaHostUnregister
#define gpuHostGetDevicePointer cudaHostGetDevicePointer
#define gpuHostRegisterMapped   cudaHostRegisterMapped
#define gpuMemcpyHostToDevice   cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#define gpuMemcpyAsync          cudaMemcpyAsync

// stream
#define gpuStreamCreate      cudaStreamCreate
#define gpuStreamDestroy     cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
// other
#define gpuSetDevice      cudaSetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuDeviceProp     cudaDeviceProp

typedef enum {
    gpuMemoryTypeHost = cudaMemoryTypeHost,
    gpuMemoryTypeDevice = cudaMemoryTypeDevice,
    gpuMemoryTypeSystem = cudaMemoryTypeUnregistered,
    gpuMemoryTypeManaged = cudaMemoryTypeManaged,
} gpuMemoryType_t;
#define FI_HMEM_GPU FI_HMEM_CUDA

//--------------------------------------------------
#elif HAVE_HIP
//--------------------------------------------------
#include <hip/hip_runtime_api.h>

#define gpuSuccess  hipSuccess
#define gpuStream_t hipStream_t

// memory
#define gpuMalloc               hipMalloc
#define gpuFree                 hipFree
#define gpuHostRegister         hipHostRegister
#define gpuHostUnregister       hipHostUnregister
#define gpuHostGetDevicePointer hipHostGetDevicePointer
#define gpuHostRegisterMapped   hipHostRegisterMapped
#define gpuMemcpyHostToDevice   hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define gpuMemcpyAsync          hipMemcpyAsync

// stream
#define gpuStreamCreate      hipStreamCreate
#define gpuStreamDestroy     hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
// other
#define gpuSetDevice      hipSetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuDeviceProp     hipDeviceProp

typedef enum {
    gpuMemoryTypeHost = hipMemoryTypeHost,
    gpuMemoryTypeDevice = hipMemoryTypeDevice,
    gpuMemoryTypeSystem,
    gpuMemoryTypeManaged,
} gpuMemoryType_t;
#define FI_HMEM_GPU FI_HMEM_ROCR

#endif  // end CUDA-HIP

//--------------------------------------------------------------------------------------------------
#else  // no HAVE_GPU
//--------------------------------------------------------------------------------------------------
#define gpuStream_t void*

#define M_HAVE_GPU  0
#define gpuSuccess  0
#define m_gpu(f)    gpuSuccess

typedef enum {
    gpuMemoryTypeHost,
    gpuMemoryTypeDevice,
    gpuMemoryTypeSystem,
    gpuMemoryTypeManaged,
} gpuMemoryType_t;
#define FI_HMEM_GPU 0

#endif  // HAVE_GPU

//==================================================================================================
#define gpuMemcpySync(dest, src, size, type)                       \
    ({                                                             \
        gpuMemcpyAsync(dest, src, size, type, GPU_DEFAULT_STREAM); \
        gpuStreamSynchronize(GPU_DEFAULT_STREAM);                  \
    })

#ifndef NDEBUG
#define m_gpu_call(func)                                                         \
    do {                                                                         \
        int m_gpu_call_res = m_gpu(func);                                        \
        m_assert(m_gpu_call_res == gpuSuccess, "GPU ERROR: %d", m_gpu_call_res); \
    } while (0)
#else
#define m_gpu_call(func) \
    do {                 \
        m_gpu(func);     \
    } while (0)
#endif

//==================================================================================================

//--------------------------------------------------------------------------------------------------
typedef enum {
    RMEM_TRIGGER = 0,
    RMEM_AWARE,
} rmem_device_t;

// device version of the RMA request
struct ofi_gpu_rma_t {
    volatile int* ready;
};
typedef struct ofi_gpu_rma_t* ofi_drma_t;

#define m_rmem_trigger(drma) \
    do {                     \
        drma->ready[0]++;    \
    } while (0)

//==================================================================================================
// RMEM device functions
//==================================================================================================
#if (HAVE_CUDA)
extern int ofi_rma_start_cuda(cudaStream_t* stream, ofi_drma_t drma);
#elif (HAVE_HIP)
extern int ofi_rma_start_hip(gpuStream_t* stream, ofi_drma_t drma);
#endif

static int ofi_rma_start_gpu(gpuStream_t* stream, ofi_drma_t drma) {
#if (HAVE_CUDA)
    return ofi_rma_start_cuda(stream, drma);
#elif (HAVE_HIP)
    return ofi_rma_start_hip(stream, drma);
#else
    m_rmem_trigger(drma);
    return gpuSuccess;
#endif
};
//--------------------------------------------------------------------------------------------------
static gpuMemoryType_t gpuMemoryType(void* ptr) {
#if (HAVE_CUDA)
    unsigned int data;
    CUresult res = cuPointerGetAttribute(&data,CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr) ptr);
    return (res == CUDA_ERROR_INVALID_VALUE) ? gpuMemoryTypeSystem : (gpuMemoryType_t)data;
#elif (HAVE_HIP)
    hipPointerAttribute_t attr;
    hipError_t ret;
    ret = hipPointerGetAttributes(&attr, ptr);
    return (gpuMemoryType_t) attr.memoryType;
#else
    return gpuMemoryTypeSystem;
#endif
}

#endif
