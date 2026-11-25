// 計画を開く時の確保、イベントのページの数、提出の途中の失敗、頻出辞書の更新の順の契約。記録用の実装で回す。
#include "collection_fixture.h"
#include "batch_executor.h"
#include "recorded_commands.h"

#include <cassert>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace prism;
using test::Command;

constexpr size_t kDeviceBytes = size_t{4} << 30;

int identity;
gpu::Stream stream() { return reinterpret_cast<gpu::Stream>(&identity); }

PlanShape projectedShape(bool causal) {
  return test::compileModel(test::collectionModel(kWidthProjected, kRuleCapabilities.back(), causal, false));
}

bool mentions(const std::exception& e, const char* text) {
  return std::string(e.what()).find(text) != std::string::npos;
}

// 計画を開く時のどの確保が足りなくても、要るバイト数と空きを付けた例外になる。頻出辞書と候補の分割を黙って外さない。
void allocationFailures() {
  const auto shape = projectedShape(false);
  int failures = 0;
  for (int k = 0;; ++k) {
    test::failAllocationAfter(k);
    try {
      Plan plan(0, stream(), shape);
      test::failAllocationAfter(-1);
      const auto capacity = plan.configureEventPages(1, -1, 0);
      assert(capacity.dictionary && capacity.splitCandidate && capacity.pageSlots == 0);
      break;
    } catch (const std::runtime_error& e) {
      assert(mentions(e, " needs ") && mentions(e, " bytes of device memory are free"));
      ++failures;
    }
  }
  test::failAllocationAfter(-1);
  assert(failures > 0);
}

// 数を指定したページは、足りなければ要るバイト数と空きを付けた例外になる。自動のページは 1 バッチのイベントの数に満たなければ
// 確保せず、確保に失敗したら 1 バッチの数まで縮める。射影表の行は 2 の冪で、1 バッチのトークンの数を超えない。
void eventPageCapacity() {
  const auto shape = projectedShape(true);
  const auto data = test::weights(shape);
  auto opened = [&] {
    auto plan = std::make_unique<Plan>(0, stream(), shape);
    plan->uploadWeights(data.data(), data.size(), 1);
    return plan;
  };
  const auto none = opened()->configureEventPages(1, -1, 0);
  assert(none.pageSlots == 0 && none.freeBefore == none.freeAfter && none.total == int64_t(kDeviceBytes));
  const auto batch = opened()->configureEventPages(1, shape.events, 0);
  assert(batch.pageSlots == shape.events);
  const int64_t rows = batch.projectionRows;
  assert(rows > 0 && (rows & (rows - 1)) == 0 && size_t(rows) <= shape.tokens());
  const int64_t batchBytes = batch.stagingBytes + batch.pageBytes + batch.projectionBytes;

  test::setDeviceMemory(size_t(batchBytes) - 1, kDeviceBytes);
  auto small = opened();
  test::resetCommands();
  assert(small->configureEventPages(1, 0, 0).pageSlots == 0 && test::commands().empty());
  bool refused = false;
  try {
    opened()->configureEventPages(1, shape.events, 0);
  } catch (const std::runtime_error& e) {
    refused = mentions(e, " needs ") && mentions(e, std::to_string(batchBytes - 1).c_str());
  }
  assert(refused);

  // 予算に余りがあれば、確保に失敗するたびに小さく確保し直し、1 バッチの数でも失敗したら確保しない。
  test::setDeviceMemory(2 * size_t(batchBytes), kDeviceBytes);
  assert(opened()->configureEventPages(1, 0, 0).pageSlots > shape.events);
  auto shrinking = opened();
  test::resetCommands();
  test::failAllocationAfter(0);
  assert(shrinking->configureEventPages(1, 0, 0).pageSlots == 0);
  test::failAllocationAfter(-1);
  std::vector<size_t> attempts;
  for (const auto& command : test::commands())
    if (command.kind == Command::kAllocate) attempts.push_back(command.bytes);
  assert(attempts.size() >= 2 && attempts.back() == size_t(batchBytes));
  for (size_t i = 1; i < attempts.size(); ++i) assert(attempts[i] < attempts[i - 1]);
  test::setDeviceMemory(kDeviceBytes, kDeviceBytes);
}

// 提出の途中で転送が失敗したら、ページの中身を使わない。次のページつきの提出は何も積まずに拒み、重みを公開し直すと戻る。
void failedSubmissionInvalidatesPages() {
  const auto shape = projectedShape(true);
  const auto data = test::weights(shape);
  Plan plan(0, stream(), shape);
  plan.uploadWeights(data.data(), data.size(), 1);
  plan.configureEventPages(1, shape.events, 0);
  auto b = test::rowBatch(shape, 4, {20, 52, 84, 116});
  const int rows = b.batch.rows, events = b.batch.events;
  std::vector<int> map(rows + 2 * events);
  std::iota(map.begin() + rows, map.begin() + rows + events, 0);
  std::iota(map.begin() + rows + events, map.end(), 0);
  std::vector<float> scores(b.batch.actions);
  auto paged = [&] { plan.forward(b.input.data(), b.batch, scores.data(), 1, map.data(), events, shape.events); };
  paged();
  test::failNextUpload();
  bool failed = false;
  try {
    paged();
  } catch (const std::runtime_error&) {
    failed = true;
  }
  assert(failed);
  test::resetCommands();
  bool refused = false;
  try {
    paged();
  } catch (const std::invalid_argument&) {
    refused = test::commands().empty();
  }
  assert(refused);
  plan.forward(b.input.data(), b.batch, scores.data(), 1);
  plan.uploadWeights(data.data(), data.size(), 1);
  paged();
}

// 頻出辞書を作り直す提出は、組と表をデバイスへ写して表を作ってから、同じストリームで入力を写す。前の辞書の転送が終わっていない
// 組の領域は書き換えない。
void dictionaryRefreshOrder() {
  const auto shape = projectedShape(false);
  const auto data = test::weights(shape);
  Plan plan(0, stream(), shape);
  plan.uploadWeights(data.data(), data.size(), 1);
  auto b = test::rowBatch(shape, shape.rows, {});
  const auto& a = shape.architecture;
  const int codeWidth = shape.declaration->inputs[int(Section::kTransitionCode)].width;
  const int segmentColumn = shape.options.segments.spans[0].column;
  auto setCodes = [&](int base, int segment) {
    uint8_t* codes = b.input.data() + b.batch.section[int(Section::kTransitionCode)];
    for (int t = 0; t < b.batch.transitions; ++t) {
      for (int tile = 0; tile < a.tiles; ++tile) {
        codes[size_t(t) * codeWidth + a.fixed() - a.tiles + tile] = uint8_t(base);
        codes[size_t(t) * codeWidth + segmentColumn + tile] = uint8_t(segment);
      }
    }
  };
  std::vector<float> scores(b.batch.actions);
  // 入力より前に写したもの（辞書の組と表）の数を返す。
  auto submit = [&] {
    test::resetCommands();
    plan.forward(b.input.data(), b.batch, scores.data(), 1);
    const auto& fifo = test::commands();
    size_t first = 0;
    while (fifo.at(first).host != b.input.data()) ++first;
    int uploads = 0;
    bool recorded = false;
    bool built = false;
    for (size_t i = 0; i < first; ++i) {
      assert(fifo[i].stream == stream());
      uploads += fifo[i].kind == Command::kUpload;
      recorded |= fifo[i].kind == Command::kEvent;
      built |= fifo[i].kind == Command::kLaunch;
    }
    assert(recorded == (uploads > 0) && built == (uploads > 0));
    return uploads;
  };
  setCodes(0, 0);
  assert(submit() == 2);
  for (int i = 0; i < 7; ++i) assert(submit() == 0);
  setCodes(1, 1);
  assert(submit() == 2);
  for (int i = 0; i < 7; ++i) assert(submit() == 0);
  setCodes(2, 0);
  test::delayNextEvent();
  assert(submit() == 0);
  assert(submit() == 2);
}

}  // namespace

int main() {
  allocationFailures();
  eventPageCapacity();
  failedSubmissionInvalidatesPages();
  dictionaryRefreshOrder();
}
