// DJL の配列のデバイスのアドレス。サーバに libtorch のヘッダがないので、libtorch が公開する
// at::TensorBase::const_data_ptr<T>() const を dlsym で引いて呼ぶ（DJL が先に読んだ libtorch_cpu.so を soname で使い回す）。
// const_data_ptr は記憶の先頭に storage_offset を足したアドレスを返し、型が合わなければ c10::Error（std::exception）を投げる。
#include <dlfcn.h>

#include <array>
#include <mutex>
#include <stdexcept>
#include <string>

#include "prism_hip_common.h"
#include "prism_hip_training.h"

namespace prism {
namespace {

// 非 virtual の const のメンバー関数（Itanium の ABI では this が最初の引数）。
using DataAccessor = const void* (*)(const void* tensor);

constexpr int kElements = 6;

// TensorElement の順の記号（c++filt: at::TensorBase::const_data_ptr<T, 0>() const）。
constexpr std::array<const char*, kElements> kSymbols = {
    "_ZNK2at10TensorBase14const_data_ptrIbLi0EEEPKT_v",
    "_ZNK2at10TensorBase14const_data_ptrIsLi0EEEPKT_v",
    "_ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v",
    "_ZNK2at10TensorBase14const_data_ptrIN3c108BFloat16ELi0EEEPKT_v",
    "_ZNK2at10TensorBase14const_data_ptrIfLi0EEEPKT_v",
    "_ZNK2at10TensorBase14const_data_ptrIhLi0EEEPKT_v",
};

std::array<DataAccessor, kElements> resolveAccessors() {
  void* torch = dlopen("libtorch_cpu.so", RTLD_LAZY | RTLD_NOLOAD);
  if (torch == nullptr) {
    throw std::runtime_error("libtorch_cpu.so is not loaded; open a PyTorch NDManager before the HIP training kernels");
  }
  std::array<DataAccessor, kElements> accessors{};
  for (int e = 0; e < kElements; ++e) {
    void* symbol = dlsym(torch, kSymbols[e]);
    if (symbol == nullptr) {
      throw std::runtime_error(std::string("libtorch_cpu.so does not export ") + kSymbols[e]);
    }
    accessors[e] = reinterpret_cast<DataAccessor>(symbol);
  }
  return accessors;
}

const std::array<DataAccessor, kElements>& accessors() {
  static std::once_flag once;
  static std::array<DataAccessor, kElements> resolved{};
  std::call_once(once, [] { resolved = resolveAccessors(); });
  return resolved;
}

}  // namespace

void* tensorData(int64_t handle, TensorElement element, int device) {
  const int index = static_cast<int>(element);
  if (handle == 0) {
    throw std::invalid_argument("the NDArray is closed");
  }
  if (index < 0 || index >= kElements) {
    throw std::invalid_argument("unknown tensor element " + std::to_string(index));
  }
  const void* data = accessors()[index](reinterpret_cast<const void*>(handle));
  if (data == nullptr) {
    throw std::invalid_argument("the NDArray has no storage");
  }
  hipPointerAttribute_t attributes{};
  PRISM_HIP(hipPointerGetAttributes(&attributes, data));
  if (attributes.type != hipMemoryTypeDevice || attributes.device != device) {
    throw std::invalid_argument("the NDArray is not in the memory of GPU " + std::to_string(device));
  }
  return const_cast<void*>(data);
}

}  // namespace prism
