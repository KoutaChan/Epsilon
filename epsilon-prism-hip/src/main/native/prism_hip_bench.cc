// prism_hip_bench: JVM の外で Plan を直接回すベンチ。入力は epsilon-prism の HipCaseBench が書いたディレクトリ（case.txt・weights.bin・input.bin、GPU で書けば
// scores.bin も）。jar の .so には入らず、既定のビルドにも入らない（CMakeLists.txt の prism_hip_bench）。
//
// 1 つの GPU に Plan を --lanes 個作り（本番のスロットと同じく、Plan ごとに自分のストリームを持つ）、どの Plan にも同じ重みとバッチを置く。
//   既定      : 各レーンは前のバッチの終わりを待ってから次を積む（本番のスロットと同じく、1 レーンにバッチ 1 つ）。レーンどうしは同時に走る。
//   --serial  : レーンを 1 つずつ回す（積んで待つ）。ほかの Plan は同じ GPU に置かれているだけ。
//   --kernel K: 最初に forward を 1 回回して作業領域を満たし、その後は K だけを Plan::run で繰り返す。K は自分が書いた領域を読み直す
//               ので値は回ごとに変わるが、測るのは時間だけ。
// どの forward の s も、最初の s とビットまで同じでなければならない（HIP の推論はビットまで再現し、バッチの組み方に依らない）。
// --dry-run は引数と入力のファイルを読んで確かめるだけで、HIP の関数を 1 つも呼ばない。
//
// 標準出力の行は "PRISM-HIP-BENCH" で始まる。speed の行は HipForwardTest#measureSpeed の PRISM-HIP-SPEED と同じ書き方で、ms と
// µs/判断は計時した回の中央値（層ごとに回すカーネルは層の和）。経過時間は、計時した回の全体を（レーン数 × 回数）で割った値。
// 終了コード: 0 成功、1 HIP の失敗、2 引数か入力のファイルの誤り、3 s の不一致。
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
    "EMBED",    "RELATION",         "ATTENTION_ALL", "MLP_ALL", "ATTENTION_FIXED", "MLP_FIXED",
    "MLP_LAST", "REREAD_KEY_VALUE", "CANDIDATE",     "REREAD",  "VALUE_GRAPH"};

constexpr int kMaxLanes = 8;

// s の不一致（終了コード 3）。
struct ScoreMismatch : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Options {
  std::string directory;
  int device = 0;
  int lanes = 1;
  int warmup = 5;
  int iterations = 20;
  int kernel = -1;  // −1 は forward
  bool profile = false;
  bool serial = false;
  bool dryRun = false;
  std::string scoresOut;
  std::string compareWith;
};

// HipCaseBench が書いたバッチ 1 つ。
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
               "usage: prism_hip_bench --case DIR [--device N] [--lanes N] [--warmup N] [--iterations N]\n"
               "                       [--kernel NAME] [--serial] [--profile] [--scores FILE] [--compare FILE] [--dry-run]\n");
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
  throw std::invalid_argument("unknown kernel '" + name + "' (expected a HipKernel name such as MLP_ALL)");
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
    } else if (flag == "--warmup") {
      options.warmup = static_cast<int>(number(flag, value(), 0, 100000));
    } else if (flag == "--iterations") {
      options.iterations = static_cast<int>(number(flag, value(), 1, 1000000));
    } else if (flag == "--kernel") {
      options.kernel = kernelOf(value());
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

// case.txt の "名前=値" の行を読む。空行と # で始まる行は飛ばす。
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
  // 形の記述。Java の HipArchitecture.encode の int を、符号つきの 10 進でカンマ区切りに並べたもの。
  const auto architecture = fields.find("architecture");
  if (architecture == fields.end()) {
    throw std::invalid_argument("case.txt has no field 'architecture'");
  }
  {
    std::stringstream list(architecture->second);
    std::string item;
    int32_t ints[prism::kArchitectureInts];
    int count = 0;
    while (std::getline(list, item, ',')) {
      if (count == prism::kArchitectureInts) {
        throw std::invalid_argument("case.txt architecture has more than " + std::to_string(prism::kArchitectureInts) +
                                    " ints");
      }
      ints[count++] = static_cast<int32_t>(number("case.txt architecture", item, -kInt - 1, kInt));
    }
    if (count != prism::kArchitectureInts) {
      throw std::invalid_argument("case.txt architecture has " + std::to_string(count) + " ints, expected " +
                                  std::to_string(prism::kArchitectureInts));
    }
    std::memcpy(&plan.architecture, ints, sizeof(plan.architecture));
  }
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
    // 領域の先頭は blob の中にあり、K1 の 16 バイトの読み込みに合わせて 16 バイト境界にそろう（Java の HipBatch と同じ取り決め）。
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
  if (std::memcmp(result.weights.data(), &plan.architecture, sizeof(plan.architecture)) != 0) {
    throw std::invalid_argument("weights.bin was built for another network architecture than case.txt");
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

// --kernel で回す層を返す。HipKernel#acceptsLayer が受ける最初の層で、全トークンの層の K2・K3 は層 0、局・席・牌の層の K4・K3 は
// 最初の局・席・牌の層、K5 は最後の層、ほかは 0。
int layerOf(int kernel, const prism::Architecture& architecture) {
  switch (static_cast<Kernel>(kernel)) {
    case Kernel::kAttentionFixed:
    case Kernel::kMlpFixed:
      return architecture.allTokenLayers();
    case Kernel::kMlpLast:
      return architecture.depth - 1;
    default:
      return 0;
  }
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

// Plan 1 つと、そのストリーム、ページ固定した入力と s。
struct Lane {
  hipStream_t stream = nullptr;
  std::unique_ptr<Plan> plan;
  void* input = nullptr;
  void* scores = nullptr;
  std::vector<std::array<float, kKernels>> millis;  // 計時した回ごとのカーネルの時間
};

// GPU の上の全てのリソース。破棄するときは Plan → ストリーム → ホストのメモリの順に解放する（Plan はストリームより先に破棄する）。
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
  }

  Bench(const Bench&) = delete;
  Bench& operator=(const Bench&) = delete;

  void open() {
    PRISM_HIP(hipSetDevice(options_.device));
    hipDeviceProp_t properties;
    PRISM_HIP(hipGetDeviceProperties(&properties, options_.device));
    std::printf("PRISM-HIP-BENCH device %d %s %s\n", options_.device, properties.name, properties.gcnArchName);
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
  }

  // 全ての回を回し、計時した回の経過時間（秒）を返す。
  double run() {
    const int total = options_.warmup + options_.iterations;
    const bool kernelOnly = options_.kernel >= 0;
    // --kernel では、先に forward を 1 回回して作業領域を満たす。レーンを同時に走らせる forward では、回 0 を先に全てのレーンに積む。
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
          lane.plan->run(static_cast<Kernel>(options_.kernel), layerOf(options_.kernel, case_.plan.architecture),
                         case_.batch);
          if (options_.serial) {
            finishKernel(lane, timed);
          }
        } else if (options_.serial) {
          submitForward(lane);
          finish(lane, timed);
        } else {
          // forward の回 it はもう積んである。終わりを待って s を照合し、次の回を積む。
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
  void submitForward(Lane& lane) {
    lane.plan->forward(lane.input, case_.batch, lane.scores, case_.version);
  }

  // forward の終わりを待ち、s を最初の s と照合する。
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
  std::vector<char> reference_;
  long long batches_ = 0;
};

// カーネルごとの中央値（ms）を並べた行を返す。samples は計時した回ごとの時間で、sum には中央値の和を書く。
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
        material.plan.architecture.depth, options.lanes, options.serial ? " serial" : "", batch.rows,
        (static_cast<double>(material.plan.architecture.fixed()) * rows + batch.events) / rows, batch.transitions / rows,
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
                material.plan.architecture.depth, options.lanes, options.serial ? " serial" : "", batch.rows, wallMs,
                wallMs * 1e3 / rows);
  }
}

int bench(const Options& options) {
  const Case material = readCase(options);
  const BatchShape& batch = material.batch;
  std::printf(
      "PRISM-HIP-BENCH case %s: depth %d rows %d events %d transitions %d actions %d bytes %zu weights %zu"
      " version %lld; lanes %d warmup %d iterations %d mode %s%s%s\n",
      options.directory.c_str(), material.plan.architecture.depth, batch.rows, batch.events, batch.transitions, batch.actions,
      batch.bytes, material.weights.size(), static_cast<long long>(material.version), options.lanes, options.warmup,
      options.iterations, options.kernel >= 0 ? kKernelNames[options.kernel] : "forward", options.serial ? " serial" : "",
      options.profile ? " profile" : "");
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
