// 実行の計画: 1 つの slot の作業域・重みの置き場・ストリーム・計時。網の名前は知らない（WeightPack の頭が置き場を説明する）。
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "prism_hip_common.h"
#include "prism_hip_format.h"
#include "prism_hip_kernels.h"

namespace prism {

// 作業域の区画（Java の HipRegion と同じ順）。
enum class Region : int {
  kInput,
  kResidual,
  kNormalized,
  kAttention,
  kRelationCode,
  kState,
  kMemory,
  kStatic,
  kTransition,
  kValue,
  kScores,
  kCount
};

// カーネル（Java の HipKernel と同じ順。K1〜K7 と、K1 の後の関係の符号 K0）。
enum class Kernel : int { kEmbed, kRelation, kAttention0, kMlp0, kAttention1, kMlp1Kv, kCandidate, kValueGraph, kCount };

constexpr int kRegions = static_cast<int>(Region::kCount);
constexpr int kKernels = static_cast<int>(Kernel::kCount);

// 計画の形（Java の HipShape）。
struct PlanShape {
  int hidden;
  int ffn;
  int gate;
  int depth;
  float epsilon;
  int fixedTokens;  // 行ごとの局・席・牌
  int rows;         // 容量
  int events;
  int transitions;
  int actions;
  size_t inputBytes;
  size_t weightBytes;

  int attention() const { return hidden / 2; }
  int heads() const { return attention() / 16; }
  size_t fixed() const { return static_cast<size_t>(fixedTokens) * rows; }
  size_t tokens() const { return fixed() + events; }
};

// 1 塊の数と区画の位置（Java の HipBatch）。
struct BatchShape {
  int rows;
  int events;
  int transitions;
  int actions;
  size_t bytes;
  size_t section[kSections];
};

// 計画。作業域・重みの置き場・重みから作るもの（偏りの表と QKV・MLP の重みの断片の写し）は作るときに一度だけ取り、仕事は全て slot の
// ストリームに積む。forward の中では同期もメモリの確保もしない。スレッド安全ではない（1 つの slot の submit のスレッドだけが使う）。
class Plan {
 public:
  // stream は DJL の PtStream.getId()（HIP のストリーム）。計画より長く生きていること。
  Plan(int device, hipStream_t stream, const PlanShape& shape);
  ~Plan();

  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;

  int device() const { return device_; }
  const PlanShape& shape() const { return shape_; }

  // 区画の容量（バイト）。
  size_t regionBytes(Region region) const;

  // WeightPack（頭つき）を読んで照らし（BF16・FP32 の部分に NaN があれば拒む）、デバイスへ写して偏りの表と QKV・MLP の重みの断片の
  // 写しを作り直し、版を覚える。公開のときだけ呼び、ストリームを同期する。重みを替える道はこれだけで、途中で失敗したら版を忘れる
  // （forward・run は次に写し終えるまで止まる）。
  void uploadWeights(const void* host, size_t bytes, int64_t version);

  // 区画の先頭 bytes バイトとホストの間の写し。ストリームに積むだけで、host は写しが終わるまで生かす。
  void copyIn(Region region, const void* host, size_t bytes);
  void copyOut(Region region, void* host, size_t bytes);

  // 作業域の今の中身のまま 1 つのカーネルを回す（試験と測り）。
  void run(Kernel kernel, int layer, const BatchShape& batch);

  // H2D 1 回 → K1〜K7 → s の D2H 1 回（同期しない）。
  void forward(const void* input, const BatchShape& batch, void* scores, int64_t version);

  void setProfile(bool enabled) { profile_ = enabled; }

  // 最後の forward・run のカーネルごとの時間（ms、層ごとの起動は和）。計時していないカーネルは NaN。終わりの印を待つ。
  std::array<float, kKernels> kernelMillis();

  void synchronize();

 private:
  static constexpr int kMaxLaunches = kMaxDepth;

  void launch(Kernel kernel, int layer, const BatchShape& batch);
  void validate(const BatchShape& batch) const;
  void validateHeader(const PackHeader& header, size_t bytes) const;
  void resetTiming();
  template <class Launch>
  void timed(Kernel kernel, Launch&& launch);

  // デバイスのメモリと印を返す（今のデバイスが device_ であること）。
  void release() noexcept;

  template <class T>
  Span<T> regionSpan(Region region, size_t elementOffset) const;
  template <class T>
  Span<const T> weight(Weight weight, int layer) const;
  template <class T>
  Span<const T> section(const BatchShape& batch, Section section) const;
  template <class T>
  Span<T> relationBias(int layer) const;
  template <class T>
  Span<T> qkvFragments(int layer) const;
  template <class T>
  Span<T> mlpFragments(Weight weight, int layer) const;
  MlpWeights mlpWeights(int layer) const;
  MlpWeights candidateWeights() const;
  TokenLayout layout(const BatchShape& batch) const;
  size_t weightBytes(Weight weight) const;

  int device_;
  hipStream_t stream_;
  PlanShape shape_;
  std::array<size_t, kRegions> regionOffset_{};
  std::array<size_t, kRegions> regionCapacity_{};
  char* workspace_ = nullptr;
  char* weights_ = nullptr;
  float* relationBias_ = nullptr;     // 層ごとの偏りの表（prism_hip_relation.h の置き方。uploadWeights が ρ から作る）
  uint16_t* qkvFragments_ = nullptr;  // 層ごとの QKV の断片の写し（k2_attention.hip の置き方。uploadWeights が W_qkv から作る）
  uint16_t* mlpFragments_ = nullptr;  // MLP の重みの断片の順の写し（prism_hip_mlp.h の置き方。uploadWeights が層ごとの W_o・W1・
                                      // W2、W_kv と候補の W_o・W1・W2 から作る）
  size_t weightsUsed_ = 0;
  PackHeader header_{};
  int64_t version_ = -1;
  bool profile_ = false;
  std::array<std::array<hipEvent_t, kMaxLaunches>, kKernels> start_{};
  std::array<std::array<hipEvent_t, kMaxLaunches>, kKernels> stop_{};
  std::array<int, kKernels> launches_{};
};

// 呼び出しの間だけ、今のスレッドの HIP のデバイスを替える。
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

}  // namespace prism
