#pragma once

#if defined(__HIPCC__)
#define PRISM_HOST_DEVICE __host__ __device__
#define PRISM_DEVICE_INLINE __device__ __forceinline__
#endif
