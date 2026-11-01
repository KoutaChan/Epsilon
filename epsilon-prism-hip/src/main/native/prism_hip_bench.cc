// prism_hip_bench: JVM の外で Plan を直に回す台。rocprofv3 は surefire の JVM の下では落ちるので、カーネルの計数と kernel-trace は
// この実行ファイルで取る。材料は epsilon-prism の HipCaseBench が書く場所（case.txt・weights.bin・input.bin、GPU で書けば
// scores.bin も）。jar の .so には入らず、既定の組み立てにも入らない（CMakeLists.txt の prism_hip_bench）。
//
// 1 つの GPU に計画を --lanes 個作り（本番の実行枠と同じく計画ごとに自分のストリーム）、どの計画にも同じ重みと塊を置く。
//   既定      : 各レーンは前の塊の終わりを待って次を積む（本番の slot と同じく 1 レーンに塊 1 つ）。レーンどうしは同時に走る。
//   --serial  : レーンを 1 つずつ回す（積んで待つ）。他の計画は同じ GPU に置かれているだけ。
//   --kernel K: 最初に forward を 1 回回して作業域を満たし、その後は K だけを Plan::run で繰り返す（K が自分の書いた区画を読み直す
//               ので値は回ごとに変わるが、測るのは時間だけ）。
// どの forward の s も最初の s とビットで同じでなければならない（HIP の推論はビットで再現し、塊の組に依らない）。
// --dry-run は引数と材料を読んで照らすだけで、HIP の関数を 1 つも呼ばない。
//
// 標準出力の行の頭は "PRISM-HIP-BENCH"。speed の行は HipForwardTest#measureSpeed の PRISM-HIP-SPEED と同じ書き方（ms と µs/判断、
// 計時した回の中央値。層ごとに回すカーネルは層の和）で、壁の時間は計時した回の全体 ÷（レーン × 回）。
// 終了コード: 0 成功、1 HIP の失敗、2 引数か材料の誤り、3 s の不一致。
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "prism_hip_plan.h"

namespace {

using prism::BatchShape;
using prism::Kernel;
using prism::kKernels;
using prism::Plan;
using prism::PlanShape;

// Java の HipKernel の名前（prism::Kernel の順）。
constexpr std::array<const char*, kKernels> kKernelNames = {
    "EMBED", "RELATION", "ATTENTION0", "MLP0", "ATTENTION1", "MLP1_KV", "CANDIDATE", "VALUE_GRAPH"};

constexpr int kMaxLanes = 8;
constexpr int kMaxIdleStreams = 16;

// s の不一致（終了コード 3）。
struct ScoreMismatch : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Options {
  std::string directory;
  int device = 0;
  int lanes = 1;
  int idleStreams = 0;
  int warmup = 5;
  int iterations = 20;
  int kernel = -1;  // −1 は forward
  size_t padBefore = 0;
  size_t padAfter = 0;
  bool profile = false;
  bool serial = false;
  bool dryRun = false;
  std::string scoresOut;
  std::string compareWith;
};

// HipCaseBench が書いた 1 塊。
struct Case {
  PlanShape plan{};
  BatchShape batch{};
  int64_t version = -1;
  std::vector<char> weights;
  std::vector<char> input;
  std::vector<char> expected;  // --compare の s（無ければ空）
};

void usage() {
  std::fprintf(stderr,
               "usage: prism_hip_bench --case DIR [--device N] [--lanes N] [--idle-streams N] [--warmup N]\n"
               "                       [--iterations N] [--kernel NAME] [--serial] [--profile] [--pad-before BYTES]\n"
               "                       [--pad-after BYTES] [--scores FILE] [--compare FILE] [--dry-run]\n");
}

long long number(const std::string& name, const std::string& text, long long low, long long high) {
  size_t used = 0;
  long long value = 0;
  try {
    value = std::stoll(text, &used, 10);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used == 0 || used != text.size() || value < low || value > high) {
    throw std::invalid_argument(name + " must be an integer in [" + std::to_string(low) + ", " +
                                std::to_string(high) + "], got '" + text + "'");
  }
  return value;
}

int kernelOf(const std::string& name) {
  for (int k = 0; k < kKernels; ++k) {
    if (name == kKernelNames[k]) {
      return k;
    }
  }
  throw std::invalid_argument("unknown kernel '" + name + "' (expected a HipKernel name such as MLP0)");
}

Options parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(flag + " needs a value");
      }
      return argv[++i];
    };
    if (flag == "--case") {
      options.directory = value();
    } else if (flag == "--device") {
      options.device = static_cast<int>(number(flag, value(), 0, 63));
    } else if (flag == "--lanes") {
      options.lanes = static_cast<int>(number(flag, value(), 1, kMaxLanes));
    } else if (flag == "--idle-streams") {
      options.idleStreams = static_cast<int>(number(flag, value(), 0, kMaxIdleStreams));
    } else if (flag == "--warmup") {
      options.warmup = static_cast<int>(number(flag, value(), 0, 100000));
    } else if (flag == "--iterations") {
      options.iterations = static_cast<int>(number(flag, value(), 1, 1000000));
    } else if (flag == "--kernel") {
      options.kernel = kernelOf(value());
    } else if (flag == "--pad-before") {
      options.padBefore = static_cast<size_t>(number(flag, value(), 0, 1LL << 34));
    } else if (flag == "--pad-after") {
      options.padAfter = static_cast<size_t>(number(flag, value(), 0, 1LL << 34));
    } else if (flag == "--profile") {
      options.profile = true;
    } else if (flag == "--serial") {
      options.serial = true;
    } else if (flag == "--dry-run") {
      options.dryRun = true;
    } else if (flag == "--scores") {
      options.scoresOut = value();
    } else if (flag == "--compare") {
      options.compareWith = value();
    } else {
      throw std::invalid_argument("unknown argument '" + flag + "'");
    }
  }
  if (options.directory.empty()) {
    throw std::invalid_argument("--case DIR is required");
  }
  return options;
}

std::vector<char> readFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::invalid_argument("cannot open " + path);
  }
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (in.bad()) {
    throw std::invalid_argument("cannot read " + path);
  }
  return bytes;
}

void writeFile(const std::string& path, const void* data, size_t bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  out.close();
  if (!out) {
    throw std::invalid_argument("cannot write " + path);
  }
}

// case.txt の "名前=値" の行（空行と # で始まる行は飛ばす）。
std::map<std::string, std::string> readDescription(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::invalid_argument("cannot open " + path);
  }
  std::map<std::string, std::string> fields;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const size_t equals = line.find('=');
    if (equals == std::string::npos || equals == 0) {
      throw std::invalid_argument(path + ": malformed line '" + line + "'");
    }
    if (!fields.emplace(line.substr(0, equals), line.substr(equals + 1)).second) {
      throw std::invalid_argument(path + ": duplicate field '" + line.substr(0, equals) + "'");
    }
  }
  return fields;
}

Case readCase(const Options& options) {
  const std::string dir = options.directory + "/";
  const std::map<std::string, std::string> fields = readDescription(dir + "case.txt");
  auto field = [&](const std::string& name, long long low, long long high) {
    const auto it = fields.find(name);
    if (it == fields.end()) {
      throw std::invalid_argument("case.txt has no field '" + name + "'");
    }
    return number("case.txt " + name, it->second, low, high);
  };
  constexpr long long kInt = 0x7fffffffLL;
  Case result;
  PlanShape& plan = result.plan;
  plan.hidden = static_cast<int>(field("hidden", 1, kInt));
  plan.ffn = static_cast<int>(field("ffn", 1, kInt));
  plan.gate = static_cast<int>(field("gate", 1, kInt));
  plan.depth = static_cast<int>(field("depth", 1, kInt));
  const uint32_t epsilonBits = static_cast<uint32_t>(field("epsilonBits", 0, 0xffffffffLL));
  std::memcpy(&plan.epsilon, &epsilonBits, sizeof(float));
  plan.fixedTokens = static_cast<int>(field("fixedTokens", 1, kInt));
  plan.rows = static_cast<int>(field("capacityRows", 1, kInt));
  plan.events = static_cast<int>(field("capacityEvents", 0, kInt));
  plan.transitions = static_cast<int>(field("capacityTransitions", 1, kInt));
  plan.actions = static_cast<int>(field("capacityActions", 1, kInt));
  plan.inputBytes = static_cast<size_t>(field("capacityInputBytes", 1, kInt));
  plan.weightBytes = static_cast<size_t>(field("weightBytes", 1, kInt));
  result.version = field("version", 0, kInt);

  BatchShape& batch = result.batch;
  batch.rows = static_cast<int>(field("rows", 1, plan.rows));
  batch.events = static_cast<int>(field("events", 0, plan.events));
  batch.transitions = static_cast<int>(field("transitions", batch.rows, plan.transitions));
  batch.actions = static_cast<int>(field("actions", batch.rows, plan.actions));
  batch.bytes = static_cast<size_t>(field("bytes", 1, static_cast<long long>(plan.inputBytes)));
  const auto sections = fields.find("sections");
  if (sections == fields.end()) {
    throw std::invalid_argument("case.txt has no field 'sections'");
  }
  std::stringstream list(sections->second);
  std::string item;
  int count = 0;
  while (std::getline(list, item, ',')) {
    if (count == prism::kSections) {
      throw std::invalid_argument("case.txt lists more than " + std::to_string(prism::kSections) + " sections");
    }
    // 区画の先頭は blob の内で、K1 の 16 バイトの読み込みにそろう（Java の HipBatch と同じ約束）。
    const long long offset = number("case.txt sections", item, 0, static_cast<long long>(batch.bytes));
    if (offset % 16 != 0) {
      throw std::invalid_argument("section " + std::to_string(count) + " offset " + item + " is not 16-byte aligned");
    }
    batch.section[count++] = static_cast<size_t>(offset);
  }
  if (count != prism::kSections) {
    throw std::invalid_argument("case.txt lists " + std::to_string(count) + " sections, expected " +
                                std::to_string(prism::kSections));
  }

  result.weights = readFile(dir + "weights.bin");
  if (result.weights.size() < sizeof(prism::PackHeader) || result.weights.size() > plan.weightBytes) {
    throw std::invalid_argument("weights.bin holds " + std::to_string(result.weights.size()) +
                                " bytes; the plan takes a header and at most " + std::to_string(plan.weightBytes));
  }
  int32_t magic = 0;
  std::memcpy(&magic, result.weights.data(), sizeof(magic));
  if (magic != prism::kPackMagic) {
    throw std::invalid_argument("weights.bin does not start with the weight pack magic");
  }
  result.input = readFile(dir + "input.bin");
  if (result.input.size() != batch.bytes) {
    throw std::invalid_argument("input.bin holds " + std::to_string(result.input.size()) + " bytes, case.txt says " +
                                std::to_string(batch.bytes));
  }
  if (!options.compareWith.empty()) {
    result.expected = readFile(options.compareWith);
    if (result.expected.size() != static_cast<size_t>(batch.actions) * sizeof(float)) {
      throw std::invalid_argument(options.compareWith + " holds " + std::to_string(result.expected.size()) +
                                  " bytes, the batch has " + std::to_string(batch.actions) + " FP32 scores");
    }
  }
  return result;
}

// --kernel で回す層（HipKernel#acceptsLayer と同じ: K2・K3 は最初の層、K4・K5 は最後の層、ほかは 0）。
int layerOf(int kernel, int depth) {
  switch (static_cast<Kernel>(kernel)) {
    case Kernel::kAttention1:
    case Kernel::kMlp1Kv:
      return depth - 1;
    default:
      return 0;
  }
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

// 1 つの計画と、そのストリーム・ページ固定の入力と s。
struct Lane {
  hipStream_t stream = nullptr;
  std::unique_ptr<Plan> plan;
  void* input = nullptr;
  void* scores = nullptr;
  std::vector<std::array<float, kKernels>> millis;  // 計時した回ごとのカーネルの時間
};

// GPU の上の全ての資源。壊すときは計画 → ストリーム → ホストのメモリの順に返す（計画はストリームより先に死ぬ）。
class Bench {
 public:
  Bench(const Options& options, const Case& material) : options_(options), case_(material) {}

  ~Bench() {
    for (Lane& lane : lanes_) {
      if (lane.stream != nullptr) {
        (void)hipStreamSynchronize(lane.stream);
      }
      lane.plan.reset();
      if (lane.stream != nullptr) {
        (void)hipStreamDestroy(lane.stream);
      }
      if (lane.input != nullptr) {
        (void)hipHostFree(lane.input);
      }
      if (lane.scores != nullptr) {
        (void)hipHostFree(lane.scores);
      }
    }
    for (hipStream_t stream : idle_) {
      (void)hipStreamDestroy(stream);
    }
    for (void* block : blocks_) {
      (void)hipFree(block);
    }
  }

  Bench(const Bench&) = delete;
  Bench& operator=(const Bench&) = delete;

  void open() {
    PRISM_HIP(hipSetDevice(options_.device));
    hipDeviceProp_t properties;
    PRISM_HIP(hipGetDeviceProperties(&properties, options_.device));
    const char* queues = std::getenv("GPU_MAX_HW_QUEUES");
    // PCI の番地は、時計と電力の記録（sysfs の hwmon）をこの GPU に結び付けるため。
    std::printf("PRISM-HIP-BENCH device %d %s %s pci %04x:%02x:%02x.0 GPU_MAX_HW_QUEUES=%s\n", options_.device,
                properties.name, properties.gcnArchName, static_cast<unsigned>(properties.pciDomainID),
                static_cast<unsigned>(properties.pciBusID), static_cast<unsigned>(properties.pciDeviceID),
                queues == nullptr ? "(unset)" : queues);
    pad(options_.padBefore);
    // 本番の 1 装置には、収集中は空いているストリームもある。一度使って HW の queue に結び付けてから、レーンのストリームを作る。
    if (options_.idleStreams > 0) {
      void* touch = nullptr;
      PRISM_HIP(hipMalloc(&touch, sizeof(uint32_t)));
      blocks_.push_back(touch);
      for (int s = 0; s < options_.idleStreams; ++s) {
        hipStream_t stream = nullptr;
        PRISM_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        idle_.push_back(stream);
        PRISM_HIP(hipMemsetAsync(touch, 0, sizeof(uint32_t), stream));
        PRISM_HIP(hipStreamSynchronize(stream));
      }
    }
    const size_t scoreBytes = static_cast<size_t>(case_.batch.actions) * sizeof(float);
    lanes_.resize(options_.lanes);
    for (Lane& lane : lanes_) {
      PRISM_HIP(hipStreamCreateWithFlags(&lane.stream, hipStreamNonBlocking));
      lane.plan = std::make_unique<Plan>(options_.device, lane.stream, case_.plan);
      lane.plan->uploadWeights(case_.weights.data(), case_.weights.size(), case_.version);
      lane.plan->setProfile(options_.profile);
      PRISM_HIP(hipHostMalloc(&lane.input, case_.batch.bytes, hipHostMallocDefault));
      PRISM_HIP(hipHostMalloc(&lane.scores, scoreBytes, hipHostMallocDefault));
      std::memcpy(lane.input, case_.input.data(), case_.batch.bytes);
    }
    pad(options_.padAfter);
  }

  // 全ての回を回し、計時した回の壁の時間（秒）を返す。
  double run() {
    const int total = options_.warmup + options_.iterations;
    const bool kernelOnly = options_.kernel >= 0;
    // --kernel は forward 1 回で作業域を満たしてから。同時に走らせる forward は回 0 を先に全てのレーンに積む。
    for (Lane& lane : lanes_) {
      if (kernelOnly) {
        submitForward(lane);
        finish(lane, false);
      } else if (!options_.serial) {
        submitForward(lane);
      }
    }
    auto start = std::chrono::steady_clock::now();
    for (int it = 0; it < total; ++it) {
      if (it == options_.warmup) {
        start = std::chrono::steady_clock::now();
      }
      const bool timed = it >= options_.warmup;
      for (Lane& lane : lanes_) {
        if (kernelOnly) {
          lane.plan->run(static_cast<Kernel>(options_.kernel), layerOf(options_.kernel, case_.plan.depth),
                         case_.batch);
          if (options_.serial) {
            finishKernel(lane, timed);
          }
        } else if (options_.serial) {
          submitForward(lane);
          finish(lane, timed);
        } else {
          // forward の回 it はもう積んである。終わりを待って照らし、次の回を積む。
          finish(lane, timed);
          if (it + 1 < total) {
            submitForward(lane);
          }
        }
      }
      if (kernelOnly && !options_.serial) {
        for (Lane& lane : lanes_) {
          finishKernel(lane, timed);
        }
      }
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
  }

  const std::vector<Lane>& lanes() const { return lanes_; }
  const std::vector<char>& reference() const { return reference_; }
  long long batches() const { return batches_; }

 private:
  void pad(size_t bytes) {
    if (bytes > 0) {
      void* block = nullptr;
      PRISM_HIP(hipMalloc(&block, bytes));
      blocks_.push_back(block);
    }
  }

  void submitForward(Lane& lane) {
    lane.plan->forward(lane.input, case_.batch, lane.scores, case_.version);
  }

  // forward の終わりを待ち、s を最初の s と照らす。
  void finish(Lane& lane, bool timed) {
    lane.plan->synchronize();
    const size_t bytes = static_cast<size_t>(case_.batch.actions) * sizeof(float);
    if (reference_.empty()) {
      reference_.assign(static_cast<const char*>(lane.scores), static_cast<const char*>(lane.scores) + bytes);
    } else if (std::memcmp(reference_.data(), lane.scores, bytes) != 0) {
      throw ScoreMismatch("scores of batch " + std::to_string(batches_) + " differ from the first batch at score " +
                          std::to_string(firstDifference(reference_.data(), lane.scores, bytes)));
    }
    ++batches_;
    if (timed && options_.profile) {
      lane.millis.push_back(lane.plan->kernelMillis());
    }
  }

  void finishKernel(Lane& lane, bool timed) {
    lane.plan->synchronize();
    if (timed && options_.profile) {
      lane.millis.push_back(lane.plan->kernelMillis());
    }
  }

  static size_t firstDifference(const void* a, const void* b, size_t bytes) {
    const auto* x = static_cast<const uint32_t*>(a);
    const auto* y = static_cast<const uint32_t*>(b);
    size_t i = 0;
    while (i < bytes / sizeof(uint32_t) && x[i] == y[i]) {
      ++i;
    }
    return i;
  }

  const Options& options_;
  const Case& case_;
  std::vector<Lane> lanes_;
  std::vector<hipStream_t> idle_;
  std::vector<void*> blocks_;  // --pad-before・--pad-after の確保と、空のストリームに一度使わせる 4 バイト
  std::vector<char> reference_;
  long long batches_ = 0;
};

// カーネルごとの中央値（ms）の行。samples は計時した回ごとの時間。
std::string kernelLine(const std::vector<std::array<float, kKernels>>& samples, int rows, double* sum) {
  std::string line;
  *sum = 0.0;
  for (int k = 0; k < kKernels; ++k) {
    std::vector<double> values;
    for (const auto& sample : samples) {
      if (!std::isnan(sample[k])) {
        values.push_back(sample[k]);
      }
    }
    if (values.empty()) {
      continue;
    }
    const double ms = median(values);
    *sum += ms;
    char item[96];
    std::snprintf(item, sizeof(item), " %s=%.3fms(%.3fus)", kKernelNames[k], ms, ms * 1e3 / rows);
    line += item;
  }
  return line;
}

void report(const Options& options, const Case& material, const Bench& bench, double seconds) {
  const BatchShape& batch = material.batch;
  const double rows = batch.rows;
  const long long timed = static_cast<long long>(options.iterations) * options.lanes;
  const double wallMs = seconds * 1e3 / static_cast<double>(timed);
  if (options.profile) {
    std::vector<std::array<float, kKernels>> all;
    for (const Lane& lane : bench.lanes()) {
      all.insert(all.end(), lane.millis.begin(), lane.millis.end());
    }
    double sum = 0.0;
    const std::string line = kernelLine(all, batch.rows, &sum);
    std::printf(
        "PRISM-HIP-BENCH speed depth %d lanes %d%s rows %d tokens/row %.1f transitions/row %.1f actions/row %.1f:%s"
        " kernels=%.3fms (%.3f us/decision) wall=%.3fms (%.3f us/decision)\n",
        material.plan.depth, options.lanes, options.serial ? " serial" : "", batch.rows,
        (static_cast<double>(material.plan.fixedTokens) * rows + batch.events) / rows, batch.transitions / rows,
        batch.actions / rows, line.c_str(), sum, sum * 1e3 / rows, wallMs, wallMs * 1e3 / rows);
    if (options.lanes > 1) {
      for (size_t i = 0; i < bench.lanes().size(); ++i) {
        double laneSum = 0.0;
        const std::string laneLine = kernelLine(bench.lanes()[i].millis, batch.rows, &laneSum);
        std::printf("PRISM-HIP-BENCH lane %zu:%s kernels=%.3fms\n", i, laneLine.c_str(), laneSum);
      }
    }
  } else {
    std::printf("PRISM-HIP-BENCH wall depth %d lanes %d%s rows %d: %.3fms per batch (%.3f us/decision)\n",
                material.plan.depth, options.lanes, options.serial ? " serial" : "", batch.rows, wallMs,
                wallMs * 1e3 / rows);
  }
}

int bench(const Options& options) {
  const Case material = readCase(options);
  const BatchShape& batch = material.batch;
  std::printf(
      "PRISM-HIP-BENCH case %s: depth %d rows %d events %d transitions %d actions %d bytes %zu weights %zu"
      " version %lld; lanes %d idle streams %d warmup %d iterations %d mode %s%s%s pads %zu/%zu\n",
      options.directory.c_str(), material.plan.depth, batch.rows, batch.events, batch.transitions, batch.actions,
      batch.bytes, material.weights.size(), static_cast<long long>(material.version), options.lanes,
      options.idleStreams, options.warmup, options.iterations,
      options.kernel >= 0 ? kKernelNames[options.kernel] : "forward", options.serial ? " serial" : "",
      options.profile ? " profile" : "", options.padBefore, options.padAfter);
  if (options.dryRun) {
    std::printf("PRISM-HIP-BENCH dry run: arguments and case files are valid; no HIP call was made\n");
    return 0;
  }
  Bench bench(options, material);
  bench.open();
  const double seconds = bench.run();
  report(options, material, bench, seconds);
  std::printf("PRISM-HIP-BENCH scores: %lld forward batches bitwise equal to the first\n", bench.batches());
  if (!options.scoresOut.empty()) {
    writeFile(options.scoresOut, bench.reference().data(), bench.reference().size());
  }
  if (!material.expected.empty()) {
    const size_t count = material.expected.size() / sizeof(float);
    const auto* x = reinterpret_cast<const uint32_t*>(material.expected.data());
    const auto* y = reinterpret_cast<const uint32_t*>(bench.reference().data());
    size_t equal = 0;
    for (size_t i = 0; i < count; ++i) {
      equal += x[i] == y[i] ? 1 : 0;
    }
    std::printf("PRISM-HIP-BENCH compare %s: %zu of %zu scores bitwise equal\n", options.compareWith.c_str(), equal,
                count);
    if (equal != count) {
      return 3;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return bench(parse(argc, argv));
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "prism_hip_bench: %s\n", e.what());
    usage();
    return 2;
  } catch (const ScoreMismatch& e) {
    std::fprintf(stderr, "prism_hip_bench: %s\n", e.what());
    return 3;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "prism_hip_bench: %s\n", e.what());
    return 1;
  }
}
