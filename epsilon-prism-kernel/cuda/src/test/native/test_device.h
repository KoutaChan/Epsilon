#pragma once

#include "device_stream.h"

namespace prism::test {
// デバイスのないホストでは CTest のスキップを返す。ほかの実行時エラーはテストの失敗にする。
inline bool hasDevice() {
  int count;
  const auto status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) return false;
  runtime::checkCuda(status, "count CUDA devices");
  return count > 0;
}
}  // namespace prism::test
