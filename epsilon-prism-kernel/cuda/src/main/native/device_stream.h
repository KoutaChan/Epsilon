#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace prism::runtime {
inline void checkCuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
}

class DeviceScope {
 public:
  explicit DeviceScope(int device) : device_(device) {
    checkCuda(cudaGetDevice(&previous_), "get CUDA device");
    if (previous_ != device_) checkCuda(cudaSetDevice(device_), "select CUDA device");
  }
  ~DeviceScope() {
    if (device_ != previous_) (void)cudaSetDevice(previous_);
  }
  DeviceScope(const DeviceScope&) = delete;
  DeviceScope& operator=(const DeviceScope&) = delete;

 private:
  int device_;
  int previous_;
};

// 構築時にストリームの所有デバイスを照合する。転送時は呼び出し元がそのデバイスを選択する。
class DeviceStream {
 public:
  DeviceStream(int device, cudaStream_t stream) : stream_(stream) {
    if (!stream || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
      throw std::invalid_argument("CUDA inference requires an explicit non-default stream");
    DeviceScope scope(device);
    int owner;
    checkCuda(cudaStreamGetDevice(stream, &owner), "get CUDA stream device");
    if (device != owner) throw std::invalid_argument("CUDA stream belongs to a different device");
    cudaDeviceProp props;
    checkCuda(cudaGetDeviceProperties(&props, device), "get CUDA device properties");
    sharedBytes_ = props.sharedMemPerBlock;
    computeCapability_ = 10 * props.major + props.minor;
  }
  int computeCapability() const { return computeCapability_; }
  cudaStream_t handle() const { return stream_; }
  size_t sharedBytes() const { return sharedBytes_; }
  void upload(void* dst, const void* src, size_t bytes) const {
    checkCuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream_), "enqueue CUDA upload");
  }
  void download(void* dst, const void* src, size_t bytes) const {
    checkCuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream_), "enqueue CUDA download");
  }
  void copy(void* dst, const void* src, size_t bytes) const {
    checkCuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream_), "enqueue CUDA copy");
  }
  void synchronize() const { checkCuda(cudaStreamSynchronize(stream_), "synchronize CUDA stream"); }

 private:
  cudaStream_t stream_;
  size_t sharedBytes_;
  int computeCapability_;
};
}

namespace prism {
inline bool eventComplete(cudaEvent_t event) {
  auto status = cudaEventQuery(event);
  if (status == cudaErrorNotReady) {
    auto pending = cudaGetLastError();
    if (pending != cudaErrorNotReady) runtime::checkCuda(pending, "query CUDA event");
    return false;
  }
  runtime::checkCuda(status, "query CUDA event");
  return true;
}
}
