#pragma once

#include "hip_error.h"

namespace prism {

// このオブジェクトが生きている間だけ、今のスレッドの HIP のデバイスを替える。
class DeviceScope {
 public:
  explicit DeviceScope(int device);
  ~DeviceScope();

  DeviceScope(const DeviceScope&) = delete;
  DeviceScope& operator=(const DeviceScope&) = delete;

 private:
  int previous_ = -1;
  int device_;
};

// 記録済みのイベントを待たずに確認する。未完了を後続の起動エラーとして持ち越さない。
inline bool eventComplete(hipEvent_t event) {
  const auto status = hipEventQuery(event);
  if (status == hipErrorNotReady) {
    const auto pending = hipGetLastError();
    if (pending != hipErrorNotReady) PRISM_HIP(pending);
    return false;
  }
  PRISM_HIP(status);
  return true;
}

// 計画のデバイスを選んでから、標準ストリームまたは呼び出し側のストリームで提出する。
class DeviceStream {
 public:
  DeviceStream(int device, hipStream_t stream);
  hipStream_t handle() const { return stream_; }
  size_t sharedBytes() const { return sharedBytes_; }

 private:
  hipStream_t stream_;
  size_t sharedBytes_;
};

}  // namespace prism
