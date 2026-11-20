#include "collection_fixture.h"
#include "batch_executor.h"
#include "input_dimensions.h"
#include "recorded_commands.h"
#include <array>
#include <cassert>
#include <cstring>
#include <memory>
#include <numeric>

namespace {
using namespace prism;
using test::kFirstTransitionColumn;
using test::kTransitionsColumn;

// 1 回の提出の起動の上限。行・イベント・遷移の数に依らない（1 行と 1024 行のバッチで同じ上限に収まる）。
constexpr int kMaxSubmissionLaunches = 64;

template <class F>
void rejectedBeforeSubmission(F submit) {
  test::resetCommands();
  try {
    submit();
    assert(false);
  } catch (const std::invalid_argument&) {
    assert(test::commands().empty());
  }
}

void checkSubmission(const void* stream, const std::vector<uint8_t>& input, const BatchShape& batch, const int* map,
                     size_t mapBytes, bool sparse, bool restoreCandidates, size_t fixedRelationBytes,
                     const void* scores, int limit) {
  using test::Command;
  const auto& fifo = test::commands();
  assert(!fifo.empty());
  size_t cursor = 0;
  auto upload = [&](const void* host, size_t bytes) {
    if (bytes == 0) return;
    assert(cursor < fifo.size());
    const auto& op = fifo[cursor++];
    assert(op.kind == Command::kUpload && op.host == host && op.bytes == bytes);
  };
  // 辞書を更新した提出では、入力を転送する前に辞書を書き込む。
  while (cursor < fifo.size() && fifo[cursor].host != input.data()) ++cursor;
  if (sparse) {
    upload(input.data(), batch.section[int(Section::kEventCategory)]);
    upload(input.data() + batch.section[int(Section::kRelation)], fixedRelationBytes);
    const size_t start = batch.section[int(Section::kTransitionCategory)];
    if (restoreCandidates) {
      upload(input.data() + start, batch.section[int(Section::kTransitionCode)] - start);
      const size_t end = batch.section[int(Section::kDiscardExact)];
      upload(input.data() + end, batch.bytes - end);
    } else {
      upload(input.data() + start, batch.bytes - start);
    }
  } else {
    upload(input.data(), batch.bytes);
  }
  if (map) upload(map, mapBytes);
  const auto& last = fifo.back();
  assert(last.kind == Command::kDownload && last.host == scores && last.bytes == batch.actions * sizeof(float));
  assert(fifo.size() >= 2 && fifo[fifo.size() - 2].kind == Command::kLaunch);
  int launches = 0, downloads = 0;
  for (size_t i = 0; i < fifo.size(); ++i) {
    const auto& op = fifo[i];
    assert(op.kind != Command::kHostWait && op.kind != Command::kAllocate);
    assert(op.stream == stream);
    if (i >= cursor) assert(op.kind != Command::kUpload);
    launches += op.kind == Command::kLaunch;
    downloads += op.kind == Command::kDownload;
  }
  assert(launches > 0 && launches <= limit && downloads == 1);
}

void verify(const WidthCapability& width, RuleCapability rule, bool paged, bool reread, bool segmentAttention = false,
            int depth = 2) {
  auto shape = test::compileModel(test::collectionModel(width, rule, paged, reread, segmentAttention, depth));
  auto data = test::weights(shape);
  int identity;
  Plan plan(0, reinterpret_cast<gpu::Stream>(&identity), shape);
  plan.uploadWeights(data.data(), data.size(), 1);
  const bool restoreCandidates = paged && segmentAttention;
  if (restoreCandidates) {
    CandidateInputLayout descriptor{};
    descriptor.ukeire = descriptor.wait = descriptor.discarded = {1, 1};
    const auto bytes =
        offsetof(CandidateInputLayout, slots) + shape.options.segments.count * sizeof(descriptor.slots[0]);
    plan.configureCandidateInput(&descriptor, bytes / sizeof(int));
  }
  const int slots = shape.events;
  if (paged) plan.configureEventPages(1, slots, 0);
  for (int rows : {1, 1024}) {
    std::vector<int> counts(rows);
    for (int i = 0; i < std::min(rows, 4); ++i) counts[i] = std::min(shape.architecture.maxEvents, 32 * i + 20);
    auto rowBatch = test::rowBatch(shape, rows, counts);
    const BatchShape batch = rowBatch.batch;
    const std::vector<uint8_t>& input = rowBatch.input;
    std::vector<float> scores(rows);
    std::vector<int> map(rows + 2 * batch.events);
    std::iota(map.begin() + rows, map.begin() + rows + batch.events, 0);
    std::iota(map.begin() + rows + batch.events, map.end(), 0);
    auto invalid = batch;
    invalid.section[int(Section::kActionExact)]++;
    rejectedBeforeSubmission([&] { plan.forward(input.data(), invalid, scores.data(), 1); });
    invalid = batch;
    invalid.rows = shape.rows + 1;
    rejectedBeforeSubmission([&] { plan.forward(input.data(), invalid, scores.data(), 1); });
    // 行の遷移が前の行に続かない、負の数、総数がバッチと違う入力は、何も積まずに拒む。
    for (auto [r, column, value] :
         {std::array<int, 3>{0, kFirstTransitionColumn, 1}, std::array<int, 3>{rows - 1, kTransitionsColumn, -1},
          std::array<int, 3>{rows - 1, kTransitionsColumn, 2}}) {
      auto broken = rowBatch;
      broken.row(r)[column] = value;
      rejectedBeforeSubmission([&] { plan.forward(broken.input.data(), batch, scores.data(), 1); });
    }
    for (int misses : {batch.events, batch.events / 2, 0}) {
      const size_t mapBytes = (size_t(rows) + batch.events + misses) * sizeof(int);
      const size_t eventBytes = shape.declaration->inputs[int(Section::kEventCategory)].width +
                                2 * shape.declaration->inputs[int(Section::kEventScalar)].width +
                                shape.declaration->inputs[int(Section::kRelation)].width;
      for (bool sparse : {false, true}) {
        if (sparse && !paged) continue;
        const size_t basicCodes = restoreCandidates ? size_t(batch.transitions) * rule.fixed() : 0;
        const size_t transfer = sparse ? mapBytes + misses * eventBytes + basicCodes : 0;
        if (transfer) map.resize((transfer + sizeof(int) - 1) / sizeof(int));
        test::resetCommands();
        plan.forward(input.data(), batch, scores.data(), 1, paged ? map.data() : nullptr, paged ? misses : 0,
                     paged ? slots : 0, transfer);
        checkSubmission(&identity, input, batch, paged ? map.data() : nullptr, transfer ? transfer : mapBytes, sparse,
                        restoreCandidates,
                        size_t(rows) * rule.fixed() * shape.declaration->inputs[int(Section::kRelation)].width,
                        scores.data(), kMaxSubmissionLaunches);
      }
    }
    test::failNextCompletion();
    bool failed = false;
    try {
      plan.uploadWeights(data.data(), data.size(), 2);
    } catch (const std::runtime_error&) {
      failed = true;
    }
    assert(failed);
    for (int version : {1, 2}) {
      rejectedBeforeSubmission([&] { plan.forward(input.data(), batch, scores.data(), version); });
    }
    plan.uploadWeights(data.data(), data.size(), 1);
  }
}
}

// 別々のデバイスの計画を GPU 1 → 0 → 1 の順に使っても、各計画の確保・転送・起動・待ちはその計画のデバイスで記録され、呼び出し元が
// 選んでいたデバイスは戻る。
namespace {
void verifyDevices(const WidthCapability& width, RuleCapability rule) {
  auto shape = test::compileModel(test::collectionModel(width, rule, false, false));
  auto data = test::weights(shape);
  int streams[2];
  auto onDevice = [&](int device, auto&& work) {
    test::selectDevice(2);
    test::resetCommands();
    work();
    assert(test::selectedDevice() == 2);
    assert(!test::commands().empty());
    for (const auto& op : test::commands()) assert(op.device == device);
  };
  std::unique_ptr<Plan> plans[2];
  for (int device : {1, 0}) {
    onDevice(device, [&] {
      plans[device] = std::make_unique<Plan>(device, reinterpret_cast<gpu::Stream>(&streams[device]), shape);
    });
    onDevice(device, [&] { plans[device]->uploadWeights(data.data(), data.size(), 1); });
  }
  std::vector<int> counts{20, 52};
  auto rowBatch = test::rowBatch(shape, 2, counts);
  std::vector<float> scores(rowBatch.batch.actions);
  for (int device : {1, 0, 1}) {
    onDevice(device, [&] { plans[device]->forward(rowBatch.input.data(), rowBatch.batch, scores.data(), 1); });
    for (const auto& op : test::commands()) assert(op.stream == &streams[device]);
  }
}
}

int main() {
  verifyDevices(prism::kWidthCapabilities[0], prism::kRuleCapabilities[0]);
  for (auto rule : prism::kRuleCapabilities)
    for (auto width : prism::kWidthCapabilities) {
      verify(width, rule, false, false);
      verify(width, rule, true, false);
      verify(width, rule, true, false, true, 3);
      if (width.reread) verify(width, rule, false, true);
    }
}
