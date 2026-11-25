#include "device_stream.h"

#include "wave_tile.h"

namespace prism {

DeviceScope::DeviceScope(int device) : device_(device) {
  PRISM_HIP(hipGetDevice(&previous_));
  if (previous_ != device_) {
    PRISM_HIP(hipSetDevice(device_));
  }
}

DeviceScope::~DeviceScope() {
  if (previous_ >= 0 && previous_ != device_) {
    (void)hipSetDevice(previous_);
  }
}

DeviceStream::DeviceStream(int device, hipStream_t stream) : stream_(stream) {
  if (stream == hipStreamPerThread)
    throw std::invalid_argument("A per-thread HIP stream cannot be shared by inference workers");
  DeviceScope scope(device);
  if (stream && stream != hipStreamLegacy) {
    hipDevice_t owner;
    PRISM_HIP(hipStreamGetDevice(stream, &owner));
    if (owner != device)
      throw std::invalid_argument("HIP stream belongs to device " + std::to_string(owner) + ", expected " +
                                  std::to_string(device));
  }
  hipDeviceProp_t properties;
  PRISM_HIP(hipGetDeviceProperties(&properties, device));
  const std::string arch = properties.gcnArchName;
  const std::string expected = PRISM_HIP_ARCH;
  if (arch != expected && arch.rfind(expected + ":", 0) != 0)
    throw std::invalid_argument("GPU " + std::to_string(device) + " is " + arch +
                                " but libprism_hip.so was built for " + expected);
  if (properties.warpSize != kWave)
    throw std::invalid_argument("HIP matrix kernels require wave32, device reports " +
                                std::to_string(properties.warpSize));
  sharedBytes_ = properties.sharedMemPerBlock;
}

}  // namespace prism
