#pragma once

#if defined(__CUDACC__)
#define PRISM_HOST_DEVICE __host__ __device__
#define PRISM_DEVICE_INLINE __device__ __forceinline__
#endif
