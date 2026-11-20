// Included only in the separate V4 candidate translation unit.
template <int A, bool kValue>
__device__ __forceinline__ void addSegmentOffsets(KeySlot& slot, const CandidateArgs& args,
                                                  const v4::SegmentWeights& weights, __amdgpu_buffer_rsrc_t table,
                                                  size_t codes, int tile, int head, Bounds& bounds) {
  // 固定位置から区分を読む。カーネル引数の配列を lane ごとの作業領域に写さずに使う。
#pragma unroll
  for (int segment = 0; segment < v4::kMaxSegmentSpans; ++segment) {
    if (segment >= weights.layout.count) break;
    const auto& span = weights.layout.spans[segment];
    const size_t at = codes + span.column + tile;
    const int id = args.code.data[bounds.index(at, 1, args.code.limit)];
    const size_t row = span.firstRow + bounds.index(id, 1, span.rows);
    const size_t offset = bounds.index(row * (2 * A) + (kValue ? A : 0) + 16 * head, 16, weights.offsets.limit);
    const uint32_t bytes = static_cast<uint32_t>(offset * sizeof(float));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      auto add = __builtin_amdgcn_raw_buffer_load_b128(table, bytes, 16 * i, 0);
      slot.code[i] = slot.code[i] + __builtin_bit_cast(f32x4, add);
    }
  }
}

#ifdef PRISM_V4_HOT
__device__ __forceinline__ int segmentCode(const CandidateArgs& args, v4::HotCandidateOffsets hot, int transition,
                                           int firstTile, int key, size_t codeAt, Bounds& bounds) {
  if (key >= firstTile) {
    const size_t at = static_cast<size_t>(transition) * args.layout.tiles + key - firstTile;
    return hot.codes.data[bounds.index(at, 1, hot.codes.limit)];
  }
  return args.code.data[bounds.index(codeAt + key, 1, args.code.limit)];
}
#endif

template <int H, int F, class Shared>
__device__ __forceinline__ void attendSegments(const CandidateArgs& args, const v4::SegmentWeights& weights,
                                               Shared& shared, int t0, int rows, int tid, Bounds& bounds
#ifdef PRISM_V4_HOT
                                               ,
                                               v4::HotCandidateOffsets hot
#endif
) {
  constexpr int A = H / 2;
  constexpr int kHeads = A / 16;
  constexpr int kItems = kHeads * kTile;
  constexpr int kKeyBytes = 2 * A * sizeof(uint16_t);
  constexpr int kRounds = (kItems + kThreads - 1) / kThreads;
  const int firstTile = F - args.layout.tiles;
  const int stride = weights.layout.codeWidth;
  const auto memory = bufferResource(args.memory.data);
#ifdef PRISM_V4_HOT
  const auto codeKey = bufferResource(hot.keys.data);
  const auto codeValue = bufferResource(hot.values.data);
#else
  const auto codeKey = bufferResource(args.codeKey.data);
  const auto codeValue = bufferResource(args.codeValue.data);
#endif
  const auto segments = bufferResource(weights.offsets.data);
  auto& inner = shared.mlp.inner;
  const float scale = args.scale * 1.4426950408889634f;
#pragma unroll 1
  for (int u = 0; u < kRounds; ++u) {
    const int item = tid + kThreads * u;
    // 同じ遷移のヘッドを隣の lane に置くと、各区分の同じ行の隣接範囲を読む。
    const int i = item / kHeads;
    if (item >= kItems || i >= rows) continue;
    const int h = item % kHeads;
    const size_t keyAt = bounds.index(static_cast<size_t>(shared.row[i]) * F * 2 * A, F * 2 * A, args.memory.limit);
    const uint32_t rowByte = static_cast<uint32_t>(keyAt * sizeof(uint16_t)) + h * 16 * sizeof(uint16_t);
    const size_t codeAt = static_cast<size_t>(t0 + i) * stride;
    u32x4 query[2] = {*reinterpret_cast<const u32x4*>(&inner[i][h * 16]),
                      *reinterpret_cast<const u32x4*>(&inner[i][h * 16 + 8])};
    float q[2][8];
    unpack8(query[0], q[0]);
    unpack8(query[1], q[1]);
#ifdef PRISM_V4_HOT
    float logits[F];
    float maximum = -INFINITY;
    const auto read = [&](KeySlot& slot, int j, bool value) {
      const int tagged = segmentCode(args, hot, t0 + i, firstTile, j, codeAt, bounds);
      const bool cached = (tagged & v4::kHotCodeTag) != 0;
      const int code = tagged & ~v4::kHotCodeTag;
      // キーと値の表は同じ行数・幅で作る。
      const size_t limit = cached ? hot.keys.limit : args.codeKey.limit;
      const size_t offset = bounds.index(static_cast<size_t>(code) * A + h * 16, 16, limit);
      slot.issue(memory, rowByte, j * kKeyBytes + (value ? A * sizeof(uint16_t) : 0), value ? codeValue : codeKey,
                 static_cast<uint32_t>(offset * sizeof(float)));
      return j >= firstTile && !cached;
    };
    const auto logit = [&](KeySlot& slot, int j, bool miss) {
      if (miss) addSegmentOffsets<A, false>(slot, args, weights, segments, codeAt, j - firstTile, h, bounds);
      logits[j] = (slot.keyDot(query) + slot.codeDot(q)) * scale;
      maximum = fmaxf(maximum, logits[j]);
    };
    // 次の表読みを発行してから、持っているキーを使う。内積と加算はキー順で進める。
    KeySlot slot0, slot1;
    bool miss0 = read(slot0, 0, false);
#pragma unroll 1
    for (int j = 0; j + 1 < F; j += 2) {
      const bool miss1 = read(slot1, j + 1, false);
      __builtin_amdgcn_sched_barrier(0);
      logit(slot0, j, miss0);
      miss0 = read(slot0, j + 2, false);
      __builtin_amdgcn_sched_barrier(0);
      logit(slot1, j + 1, miss1);
    }
    static_assert(F % 2 == 1);
    logit(slot0, F - 1, miss0);
    float total = 0.0f;
#pragma unroll 1
    for (int j = 0; j < F; ++j) {
      logits[j] = __builtin_amdgcn_exp2f(logits[j] - maximum);
      total += logits[j];
    }
    const float inverse = 1.0f / total;
    float value[2][8]{};
    const auto accumulate = [&](KeySlot& slot, int j, bool miss) {
      if (miss) addSegmentOffsets<A, true>(slot, args, weights, segments, codeAt, j - firstTile, h, bounds);
      slot.accumulate(logits[j] * inverse, value);
    };
    miss0 = read(slot0, 0, true);
#pragma unroll 1
    for (int j = 0; j + 1 < F; j += 2) {
      const bool miss1 = read(slot1, j + 1, true);
      __builtin_amdgcn_sched_barrier(0);
      accumulate(slot0, j, miss0);
      miss0 = read(slot0, j + 2, true);
      __builtin_amdgcn_sched_barrier(0);
      accumulate(slot1, j + 1, miss1);
    }
    accumulate(slot0, F - 1, miss0);
#else
    float logits[F];
    float maximum = -INFINITY;
#pragma unroll 1
    for (int j = 0; j < F; ++j) {
      const int code = args.code.data[bounds.index(codeAt + j, 1, args.code.limit)];
      const size_t offset = bounds.index(static_cast<size_t>(code) * A + h * 16, 16, args.codeKey.limit);
      KeySlot slot;
      slot.issue(memory, rowByte, j * kKeyBytes, codeKey, static_cast<uint32_t>(offset * sizeof(float)));
      if (j >= firstTile) addSegmentOffsets<A, false>(slot, args, weights, segments, codeAt, j - firstTile, h, bounds);
      logits[j] = (slot.keyDot(query) + slot.codeDot(q)) * scale;
      maximum = fmaxf(maximum, logits[j]);
    }
    float total = 0.0f;
#pragma unroll 1
    for (int j = 0; j < F; ++j) {
      logits[j] = __builtin_amdgcn_exp2f(logits[j] - maximum);
      total += logits[j];
    }
    const float inverse = 1.0f / total;
    float value[2][8]{};
#pragma unroll 1
    for (int j = 0; j < F; ++j) {
      const int code = args.code.data[bounds.index(codeAt + j, 1, args.code.limit)];
      const size_t offset = bounds.index(static_cast<size_t>(code) * A + h * 16, 16, args.codeValue.limit);
      KeySlot slot;
      slot.issue(memory, rowByte, j * kKeyBytes + A * sizeof(uint16_t), codeValue,
                 static_cast<uint32_t>(offset * sizeof(float)));
      if (j >= firstTile) addSegmentOffsets<A, true>(slot, args, weights, segments, codeAt, j - firstTile, h, bounds);
      slot.accumulate(logits[j] * inverse, value);
    }
#endif
    *reinterpret_cast<u32x4*>(&inner[i][h * 16]) = pack8Activation(value[0]);
    *reinterpret_cast<u32x4*>(&inner[i][h * 16 + 8]) = pack8Activation(value[1]);
  }
}
