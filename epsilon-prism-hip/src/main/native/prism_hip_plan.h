// 実行の計画: 1 つのスロットが持つ作業領域・重みのバッファ・ストリーム・計時。ネットワークの名前は知らない（層とバッファの中身は、
// 形の記述と WeightPack のヘッダーが説明する）。
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "prism_hip_common.h"
#include "prism_hip_format.h"
#include "prism_hip_kernels.h"

namespace prism {

// 作業領域を分けた各領域（Java の HipRegion と同じ順）。
enum class Region : int {
  kInput,
  kResidual,
  kNormalized,
  kQkv0,
  kAttention,
  kRelationCode,
  kState,
  kMemory,
  kStatic,
  kTransition,
  kValue,
  kScores,
  kEventKeyValue,
  kRereadMemory,
  kCount
};

// カーネル（Java の HipKernel と同じ順）。
enum class Kernel : int {
  kEmbed,
  kRelation,
  kAttentionAll,
  kMlpAll,
  kAttentionFixed,
  kMlpFixed,
  kMlpLast,
  kRereadKeyValue,
  kCandidate,
  kReread,
  kValueGraph,
  kCount
};

constexpr int kRegions = static_cast<int>(Region::kCount);
constexpr int kKernels = static_cast<int>(Kernel::kCount);

// 計画の形（Java の HipShape）。
struct PlanShape {
  Architecture architecture;
  int rows;  // 容量
  int events;
  int transitions;
  int actions;
  size_t inputBytes;
  size_t weightBytes;

  int hidden() const { return architecture.hidden; }
  int attention() const { return architecture.attention(); }
  int heads() const { return architecture.heads(); }
  size_t fixed() const { return static_cast<size_t>(architecture.fixed()) * rows; }
  size_t tokens() const { return fixed() + events; }
};

// 1 つのバッチの大きさ（行・イベント・遷移・手の数とバイト数）と、領域の位置（Java の HipBatch）。
struct BatchShape {
  int rows;
  int events;
  int transitions;
  int actions;
  size_t bytes;
  size_t section[kSections];
};

// 形の記述が決まりを守っているかを確かめる（Java の HipArchitecture と同じ決まりと、カーネルの上限。外れたら std::invalid_argument）。
void validateArchitecture(const Architecture& architecture);

// この形で、このカーネルがこの層を回すか（Java の HipKernel.acceptsLayer）。
bool acceptsLayer(Kernel kernel, int layer, const Architecture& architecture);

// 計画。作業領域・重みのバッファ・重みから作るもの（バイアスの表と、QKV・MLP の重みをフラグメントの順に並べた写し）は作るときに一度だけ
// 確保し、処理は全てスロットのストリームに積む。forward の中では同期もメモリの確保もしない。スレッドセーフではない（使うのは 1 つの
// スロットの submit のスレッドだけ）。
class Plan {
 public:
  // stream は DJL の PtStream.getId()（HIP のストリーム）。計画より長く生きていること。
  Plan(int device, hipStream_t stream, const PlanShape& shape);
  ~Plan();

  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;

  int device() const { return device_; }
  const PlanShape& shape() const { return shape_; }

  // 領域の容量（バイト）。
  size_t regionBytes(Region region) const;

  // WeightPack（ヘッダーつき）を読んで照合する（ヘッダーの形の記述が計画のものとビットで同じこと、BF16・FP32 の部分に NaN がないこと）。
  // そのあとデバイスへ写し、バイアスの表と、QKV（層 1 から）・MLP の重みをフラグメントの順に並べた写しを作り直して、バージョンを覚える。
  // 呼ぶのは公開のときだけで、ストリームを同期する。重みを替える手段はこれだけ。途中で失敗したらバージョンを忘れ、次に写し終えるまで
  // forward・run は例外を投げる。
  void uploadWeights(const void* host, size_t bytes, int64_t version);

  // 領域の先頭 bytes バイトとホストの間で写す。ストリームに積むだけなので、写し終えるまで host を生かしておくこと。
  void copyIn(Region region, const void* host, size_t bytes);
  void copyOut(Region region, void* host, size_t bytes);

  // 作業領域の今の中身のまま、1 つのカーネルを回す（テストと計測用）。K0・K2・K4 は、行の組を数えるために作業領域の行のインデックスを
  // ホストへ写す（同期する）。K2・K4 は、前の K0 が書いた組の順の行の番号を読む。
  void run(Kernel kernel, int layer, const BatchShape& batch);

  // H2D を 1 回 → 形の記述の順にカーネル → s の D2H を 1 回（同期しない）。K2・K4 の行の組は、input の行のインデックスからホストで数える。
  void forward(const void* input, const BatchShape& batch, void* scores, int64_t version);

  void setProfile(bool enabled) { profile_ = enabled; }

  // 最後の forward・run のカーネルごとの時間（ms。層ごとに起動したものは和）。計時していないカーネルは NaN。終了のイベントを待つ。
  std::array<float, kKernels> kernelMillis();

  void synchronize();

 private:
  static constexpr int kMaxLaunches = kMaxDepth;

  // MLP の重みをフラグメントの順に並べた写し 1 つ: 重みのスロットの項目にある行列の、行 [firstRow, firstRow + rows) の [rows, columns]。
  struct FragmentCopy {
    Weight weight;
    int entry;
    int firstRow;
    int rows;
    int columns;
    size_t offset;  // 写しの領域の中での BF16 の位置
  };

  void launch(Kernel kernel, int layer, const BatchShape& batch);
  // ホストの行のインデックス [行, 幅]（int32、アラインメントは問わない）から、K2・K4 の組ごとの行の数 groupRows_ を数える。
  void countGroupRows(const char* rowColumn, int rows);
  void validate(const BatchShape& batch) const;
  void validateHeader(const PackHeader& header, size_t bytes) const;
  // 形の記述から、MLP の重みをフラグメントの順に並べた写しの一覧を作る（層ごとの O 射影、FFN の項目ごとの up・down、候補の層のキーと
  // バリューと O 射影、局・席・牌の層ごとの QKV の k・v の行、2 段目のキーとバリューと出力の射影）。
  void planFragmentCopies();
  void resetTiming();
  template <class Launch>
  void timed(Kernel kernel, Launch&& launch);

  // デバイスのメモリとイベントを解放する（今のデバイスが device_ であること）。
  void release() noexcept;

  template <class T>
  Span<T> regionSpan(Region region, size_t elementOffset) const;
  template <class T>
  Span<const T> weight(Weight weight, int entry) const;
  template <class T>
  Span<const T> section(const BatchShape& batch, Section section) const;
  template <class T>
  Span<T> relationBias(int layer) const;
  template <class T>
  Span<T> qkvFragments(int layer) const;
  Span<const uint16_t> mlpFragments(Weight weight, int entry) const;
  MlpWeights mlpWeights(int layer, TokenGroup group) const;
  MlpWeights candidateWeights() const;
  DiscardValueWeights discardWeights() const;
  // 局・席・牌の層の FFN の区間を segment に入れ、区間の数を返す（型を分けなければ局・席・牌の 1 つ、型ごとなら局・席と牌の 2 つ）。
  // last なら最後の正規化のゲインも入れる。
  int segments(int layer, const BatchShape& batch, bool last, MlpSegment (&segment)[2]) const;
  // 最後の全トークンの層の K3 が作る、イベントのキーとバリューの組（局・席・牌の層ごとの分と、2 段目があれば 2 段目のイベントの記憶の分）
  // を args に入れる。
  void eventKeyValues(const BatchShape& batch, MlpArgs& args) const;
  TokenLayout layout(const BatchShape& batch) const;
  size_t weightBytes(Weight weight, int entry) const;

  int device_;
  hipStream_t stream_;
  PlanShape shape_;
  std::array<size_t, kRegions> regionOffset_{};
  std::array<size_t, kRegions> regionCapacity_{};
  std::vector<FragmentCopy> fragmentCopies_;
  std::array<int, kWeightEntries> fragmentCopy_{};  // 重みのスロットの項目ごとの写しの番号（なければ −1）
  size_t mlpFragmentElements_ = 0;
  char* workspace_ = nullptr;
  char* weights_ = nullptr;
  float* relationBias_ = nullptr;     // 層ごとのバイアスの表（置き方は prism_hip_relation.h。uploadWeights が ρ から作る）
  uint16_t* qkvFragments_ = nullptr;  // 層ごとの、QKV をフラグメントの順に並べた写し（置き方は k2_attention.hip。uploadWeights が
                                      // 層 1 からの W_qkv から作る。層 0 は K1 の表に折り込んであり、K2 の層 0 は qkv0 を読む）
  uint16_t* mlpFragments_ = nullptr;  // MLP の重みをフラグメントの順に並べた写し（置き方は prism_hip_mlp.h。中身は
                                      // fragmentCopies_ の一覧）
  int32_t* rowOrder_ = nullptr;  // K2・K4 が読む、組の順に並べた行の番号（K0 の起動で rowOrderKernel がバッチごとに書く。RowOrderArgs）
  int groupRows_[kAttentionGroups] = {};  // 今のバッチの組ごとの行の数（forward・run がホストで数える）
  size_t weightsUsed_ = 0;
  size_t embedShared_ = 0;  // K1 の動的な LDS（uploadWeights がヘッダーの特徴の数から決める。embedSharedBytes）
  PackHeader header_{};
  int64_t version_ = -1;
  bool profile_ = false;
  std::array<std::array<hipEvent_t, kMaxLaunches>, kKernels> start_{};
  std::array<std::array<hipEvent_t, kMaxLaunches>, kKernels> stop_{};
  std::array<int, kKernels> launches_{};  // カーネルごとの、最後の forward・run で前後の印を記録した起動の数
};

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

}  // namespace prism
