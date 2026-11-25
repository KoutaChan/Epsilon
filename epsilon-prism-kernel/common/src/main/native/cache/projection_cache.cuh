#pragma once

using cache::hashBytes;
using cache::InputKey;
using cache::RowBytes;
using cache::sameBytes;

// 種類ごとに表を区切り、列が全て一致したときだけ先に登録した代表を使う。
// 衝突が続く場合は自分を代表にするので、挿入順によらず同じ値になる。
__global__ void indexProjectionInputs(InputKey key, int kind, int first, int rows, int* owners, int* sources,
                                      int* table, int* count, ProjectionCache cache) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  // 探索の分岐に入る前に、代表の書き込み位置を一緒に予約するレーンを確定する。
  const unsigned members = device::ballot(0xffffffffu, row < rows);
  if (row >= rows) return;
  const int token = first + row;
  int owner = token;
  uint32_t hash = (2166136261u ^ kind) * 16777619u;
#pragma unroll
  for (const auto& column : key.columns) hash = hashBytes(column.row(row), column.width, hash);
  hash = (hash ^ (hash >> 16)) * 0x85ebca6bu;
  hash ^= hash >> 13;
  bool compute = true;
  const int page = prism::cache::cachedInput<kEmbeddingSlots>(key, kind, row, hash, cache, compute, ProjectionTags{});
  if (page >= 0) owner = ~page;
  int slot = hash % rows;
  for (int probe = 0; page < 0 && probe < 4; ++probe) {
    const int previous = atomicCAS(table + slot, 0, row + 1) - 1;
    if (previous < 0) break;
    bool equal = true;
#pragma unroll
    for (const auto& column : key.columns) equal &= sameBytes(column.row(row), column.row(previous), column.width);
    if (equal) {
      owner = first + previous;
      compute = false;
      break;
    }
    slot = slot + 1 == rows ? 0 : slot + 1;
  }
  owners[token] = owner;
  const int lane = threadIdx.x % kWave;
  const unsigned mask = device::ballot(members, compute);
  int begin = 0;
  if (lane == 0 && mask != 0) begin = atomicAdd(count, __popc(mask));
  begin = device::shuffle(members, begin, 0);
  if (compute) sources[begin + __popc(mask & ((1u << lane) - 1u))] = token;
}
