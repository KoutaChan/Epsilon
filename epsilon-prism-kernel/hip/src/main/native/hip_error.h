#pragma once

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace prism {

// 呼び出し名と HIP のエラー種別を保ち、任意の領域のメモリ不足を識別する。
class HipError : public std::runtime_error {
 public:
  HipError(hipError_t status, const char* call)
      : std::runtime_error(std::string(call) + " failed: " + hipGetErrorString(status)), status(status) {}
  const hipError_t status;
};

inline void hipCheck(hipError_t status, const char* call) {
  if (status != hipSuccess) {
    throw HipError(status, call);
  }
}

#define PRISM_HIP(call) ::prism::hipCheck((call), #call)

}  // namespace prism
