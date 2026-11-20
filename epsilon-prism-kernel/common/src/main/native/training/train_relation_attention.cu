#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "training/training_launch.h"
#include "training_device.h"

#include "causal_mask.h"

#include "training/relation_attention.cuh"

namespace prism {
namespace {

int blockBucket(int blocks) {
  return blocks <= kSmallBlocks ? kSmallBlocks : blocks <= kMediumBlocks ? kMediumBlocks : kLargeBlocks;
}

const uint32_t* codeWords(const int16_t* codes) {
  if (reinterpret_cast<uintptr_t>(codes) % alignof(uint32_t) != 0) {
    throw std::invalid_argument("relation codes must be aligned to 4 bytes");
  }
  return reinterpret_cast<const uint32_t*>(codes);
}

}  // namespace

static void checkCausalFixed(const AttentionShape& shape, int fixed) {
  if (fixed <= 0 || fixed > shape.keys || shape.queries != shape.keys) {
    throw std::invalid_argument("invalid causal attention prefix");
  }
}

void checkAttentionShape(const AttentionShape& shape) {
  auto require = [](bool condition, const std::string& message) {
    if (!condition) {
      throw std::invalid_argument(message);
    }
  };
  require(shape.rows > 0, "attention needs rows, got " + std::to_string(shape.rows));
  require(shape.queries > 0 && shape.queries <= kAttentionMaxTokens, "attention queries must be in 1.." +
                                                                         std::to_string(kAttentionMaxTokens) +
                                                                         ", got " + std::to_string(shape.queries));
  require(
      shape.keys > 0 && shape.keys <= kAttentionMaxTokens,
      "attention keys must be in 1.." + std::to_string(kAttentionMaxTokens) + ", got " + std::to_string(shape.keys));
  require(shape.codeQueries >= shape.queries, "relation codes must cover the queries");
  require(shape.heads > 0 && shape.heads % 2 == 0,
          "attention needs an even number of heads, got " + std::to_string(shape.heads));
  require(shape.words == 2,
          "relation codes must be one 32-bit word per pair, got " + std::to_string(shape.words) + " words");
  require(shape.segments > 0 && shape.segments <= kAttentionMaxSegments,
          "relation segments must be in 1.." + std::to_string(kAttentionMaxSegments));
  require(shape.entries > 0 && shape.entries <= kAttentionMaxEntries,
          "relation entries must be in 1.." + std::to_string(kAttentionMaxEntries));
  require(shape.scale > 0.0f, "attention scale must be positive");
}

template <bool Causal>
static void launchForward(const AttentionShape& shape, int fixed, const AttentionForward& arrays, gpu::Stream stream) {
  if constexpr (Causal) checkCausalFixed(shape, fixed);
  checkAttentionShape(shape);
  const size_t rows = static_cast<size_t>(shape.rows);
  const size_t width = static_cast<size_t>(shape.heads) * kF;
  ForwardArgs args{};
  args.shape = shape;
  args.fixed = fixed;
  args.query = {arrays.query, rows * shape.queries * width};
  args.keyValue = {arrays.keyValue, rows * shape.keys * 2 * width};
  args.mask = {arrays.mask, rows * shape.keys};
  args.codes = {codeWords(arrays.codes), rows * shape.codeQueries * shape.keys};
  args.table = {arrays.table, static_cast<size_t>(shape.segments) * shape.entries * shape.heads};
  args.out = {arrays.out, rows * shape.queries * width};
  args.outFloat = {arrays.outFloat, rows * shape.queries * width};
  args.logSumExp = {arrays.logSumExp, rows * shape.heads * shape.queries};
  const int queryBlocks = (shape.queries + 15) / 16;
  const int keyBlocks = (shape.keys + 15) / 16;
  // 同じ行のヘッドの対が隣どうしで走る（行のコードと k・v を L2 で使い回す）。ウェーブの数はクエリのタイルの数。
  const dim3 grid(shape.heads / 2, shape.rows);
  const dim3 block(kWave * queryBlocks);
  switch (blockBucket(std::max(queryBlocks, keyBlocks))) {
    case kSmallBlocks:
      forwardKernel<kSmallBlocks, Causal><<<grid, block, 0, stream>>>(args);
      break;
    case kMediumBlocks:
      forwardKernel<kMediumBlocks, Causal><<<grid, block, 0, stream>>>(args);
      break;
    default:
      forwardKernel<kLargeBlocks, Causal><<<grid, block, 0, stream>>>(args);
      break;
  }
  PRISM_GPU(gpu::lastError());
}

template <bool Causal>
static void launchBackward(const AttentionShape& shape, int fixed, const AttentionBackward& arrays,
                           gpu::Stream stream) {
  if constexpr (Causal) checkCausalFixed(shape, fixed);
  checkAttentionShape(shape);
  const size_t rows = static_cast<size_t>(shape.rows);
  const size_t width = static_cast<size_t>(shape.heads) * kF;
  BackwardArgs args{};
  args.shape = shape;
  args.fixed = fixed;
  args.query = {arrays.query, rows * shape.queries * width};
  args.keyValue = {arrays.keyValue, rows * shape.keys * 2 * width};
  args.mask = {arrays.mask, rows * shape.keys};
  args.codes = {codeWords(arrays.codes), rows * shape.codeQueries * shape.keys};
  args.table = {arrays.table, static_cast<size_t>(shape.segments) * shape.entries * shape.heads};
  args.outFloat = {arrays.outFloat, rows * shape.queries * width};
  args.logSumExp = {arrays.logSumExp, rows * shape.heads * shape.queries};
  args.gradOut = {arrays.gradOut, rows * shape.queries * width};
  args.gradQuery = {arrays.gradQuery, rows * shape.queries * width};
  args.gradKeyValue = {arrays.gradKeyValue, rows * shape.keys * 2 * width};
  args.gradTable = {arrays.gradTable, rows * shape.heads * shape.segments * shape.entries};
  const int queryBlocks = (shape.queries + 15) / 16;
  const int keyBlocks = (shape.keys + 15) / 16;
  // 同じ行のヘッドが隣どうしで走る。ウェーブの数はキーのタイルの数（段 B に必要な 5 ウェーブより少なくしない）。
  const dim3 grid(shape.heads, shape.rows);
  const dim3 block(kWave * std::max(keyBlocks, kMinBackwardWaves));
  switch (blockBucket(std::max(queryBlocks, keyBlocks))) {
    case kSmallBlocks:
      backwardKernel<kSmallBlocks, Causal><<<grid, block, 0, stream>>>(args);
      break;
    case kMediumBlocks:
      backwardKernel<kMediumBlocks, Causal><<<grid, block, 0, stream>>>(args);
      break;
    default:
      backwardKernel<kLargeBlocks, Causal><<<grid, block, 0, stream>>>(args);
      break;
  }
  PRISM_GPU(gpu::lastError());
}

void attentionForward(const AttentionShape& shape, const AttentionForward& arrays, gpu::Stream stream) {
  launchForward<false>(shape, 0, arrays, stream);
}

void causalAttentionForward(const AttentionShape& shape, int fixed, const AttentionForward& arrays,
                            gpu::Stream stream) {
  launchForward<true>(shape, fixed, arrays, stream);
}

void attentionBackward(const AttentionShape& shape, const AttentionBackward& arrays, gpu::Stream stream) {
  launchBackward<false>(shape, 0, arrays, stream);
}

void causalAttentionBackward(const AttentionShape& shape, int fixed, const AttentionBackward& arrays,
                             gpu::Stream stream) {
  launchBackward<true>(shape, fixed, arrays, stream);
}

}  // namespace prism
