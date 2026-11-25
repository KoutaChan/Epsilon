#pragma once

#include <stdexcept>
#include <string>

#include "inference_device.h"

namespace prism {

// デバイスの確保の結果を確かめる。失敗なら、実行時のエラーの状態を消してから例外を投げる（後の起動の検査に持ち越さない）。空きが
// 足りなかった時は、要るバイト数と、その時の空きと総量を付ける。
inline void requireAllocated(gpu::Status status, size_t bytes, const char* what) {
  if (status == gpu::success) return;
  (void)gpu::lastError();
  if (status != gpu::outOfMemory) PRISM_GPU(status);
  size_t available = 0;
  size_t total = 0;
  const bool known = gpu::memoryInfo(&available, &total) == gpu::success;
  (void)gpu::lastError();
  throw std::runtime_error(
      std::string(what) + " needs " + std::to_string(bytes) + " bytes; " +
      (known ? std::to_string(available) + " of " + std::to_string(total) + " bytes of device memory are free"
             : "the free device memory is unknown"));
}

}  // namespace prism
