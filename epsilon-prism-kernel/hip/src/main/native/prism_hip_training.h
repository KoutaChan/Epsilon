#pragma once
#include "training/training_launch.h"

namespace prism {
// DJL の配列の要素の型（Java の HipAttention.Element と同じ順）。
enum class TensorElement : int { kInt16 = 0, kInt32 = 1, kBfloat16 = 2, kFloat32 = 3, kUint8 = 4 };

// DJL の配列（PtNDArray.getHandle() の torch::Tensor*）のデータの先頭アドレスを返す。型が合わなければ libtorch が例外を投げる。
// アドレスが指定した GPU のメモリでなければ std::invalid_argument を投げる。
void* tensorData(int64_t handle, TensorElement element, int device);

hipStream_t trainingStream(hipStream_t expected);

}  // namespace prism
