// com.epsilon.prism.hip.PrismHip の JNI の実装。C++ の例外は Java の例外に変える（引数の誤りは IllegalArgumentException、
// それ以外は IllegalStateException）。
#include <jni.h>

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "prism_hip_plan.h"
#include "prism_hip_runtime.h"
#include "prism_hip_training.h"
#include "prism_hip_v4.h"

namespace {

using prism::Plan;

void throwJava(JNIEnv* env, const char* type, const char* message) {
  if (env->ExceptionCheck()) {
    return;
  }
  jclass exception = env->FindClass(type);
  if (exception != nullptr) {
    env->ThrowNew(exception, message);
  }
}

template <class Body>
auto guarded(JNIEnv* env, Body&& body) -> decltype(body()) {
  try {
    return body();
  } catch (const std::invalid_argument& e) {
    throwJava(env, "java/lang/IllegalArgumentException", e.what());
  } catch (const std::exception& e) {
    throwJava(env, "java/lang/IllegalStateException", e.what());
  } catch (...) {
    throwJava(env, "java/lang/IllegalStateException", "unknown native error in libprism_hip.so");
  }
  return decltype(body())();
}

Plan* plan(jlong handle) {
  if (handle == 0) {
    throw std::invalid_argument("the HIP plan is closed");
  }
  return reinterpret_cast<Plan*>(handle);
}

prism::Region region(jint value) {
  if (value < 0 || value >= prism::kRegions) {
    throw std::invalid_argument("unknown workspace region " + std::to_string(value));
  }
  return static_cast<prism::Region>(value);
}

prism::Kernel kernel(jint value) {
  if (value < 0 || value >= prism::kKernels) {
    throw std::invalid_argument("unknown kernel " + std::to_string(value));
  }
  return static_cast<prism::Kernel>(value);
}

// direct ByteBuffer の [position, position + bytes) の先頭アドレスを返す。
char* direct(JNIEnv* env, jobject buffer, jint position, size_t bytes) {
  char* address = static_cast<char*>(env->GetDirectBufferAddress(buffer));
  const jlong capacity = env->GetDirectBufferCapacity(buffer);
  if (address == nullptr || capacity < 0) {
    throw std::invalid_argument("the buffer must be a direct ByteBuffer");
  }
  if (position < 0 || static_cast<size_t>(position) > static_cast<size_t>(capacity) ||
      bytes > static_cast<size_t>(capacity) - static_cast<size_t>(position)) {
    throw std::invalid_argument("buffer range [" + std::to_string(position) + ", +" + std::to_string(bytes) +
                                ") exceeds its capacity " + std::to_string(capacity));
  }
  return address + position;
}

size_t size(jlong bytes) {
  if (bytes < 0) {
    throw std::invalid_argument("byte count must not be negative");
  }
  return static_cast<size_t>(bytes);
}

// Java が HipAttention で配列から取り出したデバイスのアドレスを、型付きのポインタにする。
template <class T>
T* device(jlong address) {
  if (address == 0) {
    throw std::invalid_argument("device address must not be null");
  }
  return reinterpret_cast<T*>(address);
}

template <size_t N>
std::array<jlong, N> addresses(JNIEnv* env, jlongArray values) {
  if (values == nullptr || env->GetArrayLength(values) != N) {
    throw std::invalid_argument("invalid V4 attention address count");
  }
  std::array<jlong, N> result{};
  env->GetLongArrayRegion(values, 0, N, result.data());
  if (env->ExceptionCheck()) {
    throw std::invalid_argument("cannot read V4 attention addresses");
  }
  return result;
}

prism::v4::SegmentAttention segmentAttentionShape(JNIEnv* env, jintArray dimensions, float scale) {
  constexpr int inputCount = 8;
  constexpr int count = inputCount + prism::v4::kSegmentLayoutInts;
  if (dimensions == nullptr || env->GetArrayLength(dimensions) != count) {
    throw std::invalid_argument("invalid V4 attention dimensions");
  }
  std::array<jint, count> d{};
  env->GetIntArrayRegion(dimensions, 0, count, d.data());
  if (env->ExceptionCheck()) {
    throw std::invalid_argument("cannot read V4 attention dimensions");
  }
  prism::v4::SegmentLayout layout{};
  std::memcpy(&layout, d.data() + inputCount, sizeof(layout));
  return {{d[0], d[1], d[2], d[3], d[4], d[5], layout}, d[6], d[7], scale};
}

prism::v4::SegmentAttentionArrays segmentAttentionInputs(JNIEnv* env, jlongArray values) {
  const auto a = addresses<6>(env, values);
  return {device<const float>(a[0]), device<const float>(a[1]),   device<const float>(a[2]),
          device<const float>(a[3]), device<const uint8_t>(a[4]), device<const int32_t>(a[5])};
}

prism::BatchShape batch(JNIEnv* env, jint rows, jint events, jint transitions, jint actions, jlong bytes,
                        jlongArray sections) {
  if (sections == nullptr || env->GetArrayLength(sections) != prism::kSections) {
    throw std::invalid_argument("expected " + std::to_string(prism::kSections) + " section offsets");
  }
  std::array<jlong, prism::kSections> offsets{};
  env->GetLongArrayRegion(sections, 0, prism::kSections, offsets.data());
  if (env->ExceptionCheck()) {
    throw std::invalid_argument("cannot read the section offsets");
  }
  prism::BatchShape shape{};
  shape.rows = rows;
  shape.events = events;
  shape.transitions = transitions;
  shape.actions = actions;
  shape.bytes = size(bytes);
  for (int s = 0; s < prism::kSections; ++s) {
    shape.section[s] = size(offsets[s]);
  }
  return shape;
}

}  // namespace

extern "C" {

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_segmentAttentionForward(JNIEnv* env, jclass, jlong stream,
                                                                                   jintArray dimensions, jfloat scale,
                                                                                   jlongArray inputs, jlong output,
                                                                                   jlong probabilities) {
  guarded(env, [&] {
    prism::v4::launchSegmentAttention(segmentAttentionShape(env, dimensions, scale),
                                      segmentAttentionInputs(env, inputs), device<float>(output),
                                      device<float>(probabilities), reinterpret_cast<hipStream_t>(stream));
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_segmentAttentionBackward(JNIEnv* env, jclass, jlong stream,
                                                                                    jintArray dimensions, jfloat scale,
                                                                                    jlongArray inputs,
                                                                                    jlongArray gradients) {
  guarded(env, [&] {
    const auto g = addresses<11>(env, gradients);
    const prism::v4::SegmentAttentionBackward b{
        device<const float>(g[0]), device<const float>(g[1]), device<float>(g[2]),   device<float>(g[3]),
        device<float>(g[4]),       device<float>(g[5]),       device<float>(g[6]),   device<float>(g[7]),
        device<int32_t>(g[8]),     device<int32_t>(g[9]),     device<int32_t>(g[10])};
    prism::v4::launchSegmentAttentionBackward(segmentAttentionShape(env, dimensions, scale),
                                              segmentAttentionInputs(env, inputs), b,
                                              reinterpret_cast<hipStream_t>(stream));
  });
}

JNIEXPORT jobjectArray JNICALL Java_com_epsilon_prism_hip_PrismHip_runtimeInfo(JNIEnv* env, jclass) {
  return guarded(env, [&]() -> jobjectArray {
    const std::vector<std::string> info = prism::runtimeInfo();
    jobjectArray array =
        env->NewObjectArray(static_cast<jsize>(info.size()), env->FindClass("java/lang/String"), nullptr);
    if (array == nullptr) {
      return nullptr;
    }
    for (size_t i = 0; i < info.size(); ++i) {
      jstring value = env->NewStringUTF(info[i].c_str());
      env->SetObjectArrayElement(array, static_cast<jsize>(i), value);
      env->DeleteLocalRef(value);
    }
    return array;
  });
}

JNIEXPORT jstring JNICALL Java_com_epsilon_prism_hip_PrismHip_deviceArch(JNIEnv* env, jclass, jint device) {
  return guarded(env, [&]() -> jstring { return env->NewStringUTF(prism::deviceArch(device).c_str()); });
}

JNIEXPORT jlong JNICALL Java_com_epsilon_prism_hip_PrismHip_newPlan(JNIEnv* env, jclass, jint device, jlong stream,
                                                                    jintArray architecture, jintArray options,
                                                                    jint rows, jint events, jint transitions,
                                                                    jint actions, jlong inputBytes, jlong weightBytes) {
  return guarded(env, [&]() -> jlong {
    if (architecture == nullptr || env->GetArrayLength(architecture) != prism::kArchitectureInts) {
      throw std::invalid_argument("expected an architecture of " + std::to_string(prism::kArchitectureInts) + " ints");
    }
    if (options == nullptr || env->GetArrayLength(options) != prism::kPlanOptionsInts) {
      throw std::invalid_argument("expected plan options of " + std::to_string(prism::kPlanOptionsInts) + " ints");
    }
    prism::PlanShape shape{};
    static_assert(sizeof(jint) == sizeof(int32_t), "the architecture is a sequence of int32");
    env->GetIntArrayRegion(architecture, 0, prism::kArchitectureInts, reinterpret_cast<jint*>(&shape.architecture));
    env->GetIntArrayRegion(options, 0, prism::kPlanOptionsInts, reinterpret_cast<jint*>(&shape.options));
    if (env->ExceptionCheck()) {
      throw std::invalid_argument("cannot read the plan descriptor");
    }
    shape.rows = rows;
    shape.events = events;
    shape.transitions = transitions;
    shape.actions = actions;
    shape.inputBytes = size(inputBytes);
    shape.weightBytes = size(weightBytes);
    return reinterpret_cast<jlong>(new Plan(device, reinterpret_cast<hipStream_t>(stream), shape));
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_freePlan(JNIEnv* env, jclass, jlong handle) {
  guarded(env, [&] { delete plan(handle); });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_countForward(JNIEnv* env, jclass, jlong stream, jint rows,
                                                                        jint transitions, jint tiles, jint codes,
                                                                        jint width, jlong projected, jlong gates,
                                                                        jlong rowIndex, jlong ids, jlong output) {
  guarded(env, [&] {
    prism::v4::launchCountTraining({rows, transitions, tiles, codes, width}, device<const uint16_t>(projected),
                                   device<const float>(gates), device<const int32_t>(rowIndex),
                                   device<const int32_t>(ids), device<uint16_t>(output),
                                   reinterpret_cast<hipStream_t>(stream));
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_countBackward(JNIEnv* env, jclass, jlong stream, jint rows,
                                                                         jint transitions, jint tiles, jint codes,
                                                                         jint width, jlong projected, jlong gates,
                                                                         jlong rowIndex, jlong ids, jlong gradOutput,
                                                                         jlong gradProjected, jlong gradGate) {
  guarded(env, [&] {
    prism::v4::launchCountTrainingBackward(
        {rows, transitions, tiles, codes, width}, device<const uint16_t>(projected), device<const float>(gates),
        device<const int32_t>(rowIndex), device<const int32_t>(ids), device<const uint16_t>(gradOutput),
        device<uint16_t>(gradProjected), device<float>(gradGate), reinterpret_cast<hipStream_t>(stream));
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_uploadWeights(JNIEnv* env, jclass, jlong handle,
                                                                         jobject pack, jint position, jlong bytes,
                                                                         jlong version) {
  guarded(env, [&] { plan(handle)->uploadWeights(direct(env, pack, position, size(bytes)), size(bytes), version); });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_copyIn(JNIEnv* env, jclass, jlong handle, jint target,
                                                                  jobject source, jint position, jlong bytes) {
  guarded(env, [&] { plan(handle)->copyIn(region(target), direct(env, source, position, size(bytes)), size(bytes)); });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_copyOut(JNIEnv* env, jclass, jlong handle, jint source,
                                                                   jobject target, jint position, jlong bytes) {
  guarded(env, [&] { plan(handle)->copyOut(region(source), direct(env, target, position, size(bytes)), size(bytes)); });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_run(JNIEnv* env, jclass, jlong handle, jint which,
                                                               jint layer, jint rows, jint events, jint transitions,
                                                               jint actions, jlong bytes, jlongArray sections) {
  guarded(env, [&] {
    plan(handle)->run(kernel(which), layer, batch(env, rows, events, transitions, actions, bytes, sections));
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_forward(JNIEnv* env, jclass, jlong handle, jobject input,
                                                                   jint inputPosition, jint rows, jint events,
                                                                   jint transitions, jint actions, jlong bytes,
                                                                   jlongArray sections, jobject scores,
                                                                   jint scoresPosition, jlong version) {
  guarded(env, [&] {
    const prism::BatchShape shape = batch(env, rows, events, transitions, actions, bytes, sections);
    if (actions < 0) {
      throw std::invalid_argument("action count must not be negative");
    }
    char* host = direct(env, input, inputPosition, shape.bytes);
    char* out = direct(env, scores, scoresPosition, static_cast<size_t>(actions) * sizeof(float));
    plan(handle)->forward(host, shape, out, version);
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_forwardPaged(
    JNIEnv* env, jclass, jlong handle, jobject input, jint inputPosition, jint rows, jint events, jint transitions,
    jint actions, jlong bytes, jlongArray sections, jobject scores, jint scoresPosition, jlong version,
    jobject eventMap, jint misses, jint slots, jint transferBytes) {
  guarded(env, [&] {
    const prism::BatchShape shape = batch(env, rows, events, transitions, actions, bytes, sections);
    if (actions < 0) {
      throw std::invalid_argument("action count must not be negative");
    }
    char* host = direct(env, input, inputPosition, shape.bytes);
    char* out = direct(env, scores, scoresPosition, static_cast<size_t>(actions) * sizeof(float));
    if (misses < 0 || misses > events || rows < 0 || events < 0 || (eventMap == nullptr && misses != 0))
      throw std::invalid_argument("invalid event miss count");
    const size_t mapInts = static_cast<size_t>(rows) + events + misses;
    if (transferBytes < 0 || (transferBytes && (!eventMap || size_t(transferBytes) < mapInts * sizeof(int))))
      throw std::invalid_argument("invalid event transfer length");
    size_t bytesNeeded = transferBytes ? size_t(transferBytes) : mapInts * sizeof(int);
    auto* map = eventMap ? reinterpret_cast<const int*>(direct(env, eventMap, 0, bytesNeeded)) : nullptr;
    plan(handle)->forward(host, shape, out, version, map, misses, slots, size_t(transferBytes));
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_configureCandidateInput(JNIEnv* env, jclass, jlong handle,
                                                                                   jintArray descriptor) {
  guarded(env, [&] {
    if (!descriptor) throw std::invalid_argument("missing candidate input layout");
    std::array<jint, sizeof(prism::CandidateInputLayout) / sizeof(jint)> words{};
    size_t count = env->GetArrayLength(descriptor);
    if (count > words.size()) throw std::invalid_argument("oversized candidate input layout");
    env->GetIntArrayRegion(descriptor, 0, count, words.data());
    if (!env->ExceptionCheck()) plan(handle)->configureCandidateInput(words.data(), count);
  });
}

JNIEXPORT jlongArray JNICALL Java_com_epsilon_prism_hip_PrismHip_configureEventPages(JNIEnv* env, jclass, jlong handle,
                                                                                     jint remaining, jint events,
                                                                                     jdouble reserveFraction) {
  return guarded(env, [&]() -> jlongArray {
    auto allocation = plan(handle)->configureEventPages(remaining, events, reserveFraction);
    jlong values[] = {allocation[0], allocation[1], allocation[2], allocation[3]};
    jlongArray result = env->NewLongArray(4);
    if (result != nullptr) env->SetLongArrayRegion(result, 0, 4, values);
    return result;
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_profile(JNIEnv* env, jclass, jlong handle,
                                                                   jboolean enabled) {
  guarded(env, [&] { plan(handle)->setProfile(enabled == JNI_TRUE); });
}

JNIEXPORT jfloatArray JNICALL Java_com_epsilon_prism_hip_PrismHip_kernelMillis(JNIEnv* env, jclass, jlong handle) {
  return guarded(env, [&]() -> jfloatArray {
    const std::array<float, prism::kKernels> millis = plan(handle)->kernelMillis();
    jfloatArray array = env->NewFloatArray(prism::kKernels);
    if (array != nullptr) {
      env->SetFloatArrayRegion(array, 0, prism::kKernels, millis.data());
    }
    return array;
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_synchronize(JNIEnv* env, jclass, jlong handle) {
  guarded(env, [&] { plan(handle)->synchronize(); });
}

JNIEXPORT jlong JNICALL Java_com_epsilon_prism_hip_PrismHip_tensorAddress(JNIEnv* env, jclass, jlong handle,
                                                                          jint element, jint gpu) {
  return guarded(env, [&]() -> jlong {
    return reinterpret_cast<jlong>(prism::tensorData(handle, static_cast<prism::TensorElement>(element), gpu));
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_relationCodes(
    JNIEnv* env, jclass, jlong stream, jint rows, jint events, jint tokens, jint seats, jint tiles, jint rowWidth,
    jint rowFirstEvent, jint rowEvents, jint width, jint tile, jint owner, jint event, jint riichi, jint lastDiscard,
    jint link, jint safety, jint safeties, jlong relation, jlong rowColumn, jlong tileRelation, jlong codes) {
  guarded(env, [&] {
    const prism::RelationCodeShape shape{
        prism::TokenLayout{rows, seats, tiles},
        events,
        tokens,
        rowWidth,
        rowFirstEvent,
        rowEvents,
        prism::RelationColumns{width, tile, owner, event, riichi, lastDiscard, link, safety, safeties}};
    const prism::RelationCodeArrays arrays{device<const uint8_t>(relation), device<const int32_t>(rowColumn),
                                           device<const int32_t>(tileRelation), device<uint32_t>(codes)};
    prism::relationCodes(shape, arrays, reinterpret_cast<hipStream_t>(stream));
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_attentionForward(
    JNIEnv* env, jclass, jlong stream, jint rows, jint queries, jint keys, jint codeQueries, jint heads, jint words,
    jint segments, jint entries, jfloat scale, jint fixed, jlong query, jlong keyValue, jlong mask, jlong codes,
    jlong table, jlong out, jlong outFloat, jlong logSumExp) {
  guarded(env, [&] {
    const prism::AttentionShape shape{rows, queries, keys, codeQueries, heads, words, segments, entries, scale};
    const prism::AttentionForward arrays{device<const uint16_t>(query), device<const uint16_t>(keyValue),
                                         device<const float>(mask),     device<const int16_t>(codes),
                                         device<const float>(table),    device<uint16_t>(out),
                                         device<float>(outFloat),       device<float>(logSumExp)};
    const auto hipStream = reinterpret_cast<hipStream_t>(stream);
    if (fixed != 0) {
      prism::causalAttentionForward(shape, fixed, arrays, hipStream);
    } else {
      prism::attentionForward(shape, arrays, hipStream);
    }
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_attentionBackward(
    JNIEnv* env, jclass, jlong stream, jint rows, jint queries, jint keys, jint codeQueries, jint heads, jint words,
    jint segments, jint entries, jfloat scale, jint fixed, jlong query, jlong keyValue, jlong mask, jlong codes,
    jlong table, jlong outFloat, jlong logSumExp, jlong gradOut, jlong gradQuery, jlong gradKeyValue, jlong gradTable) {
  guarded(env, [&] {
    const prism::AttentionShape shape{rows, queries, keys, codeQueries, heads, words, segments, entries, scale};
    const prism::AttentionBackward arrays{
        device<const uint16_t>(query),  device<const uint16_t>(keyValue), device<const float>(mask),
        device<const int16_t>(codes),   device<const float>(table),       device<const float>(outFloat),
        device<const float>(logSumExp), device<const uint16_t>(gradOut),  device<uint16_t>(gradQuery),
        device<uint16_t>(gradKeyValue), device<float>(gradTable)};
    const auto hipStream = reinterpret_cast<hipStream_t>(stream);
    if (fixed != 0) {
      prism::causalAttentionBackward(shape, fixed, arrays, hipStream);
    } else {
      prism::attentionBackward(shape, arrays, hipStream);
    }
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_normForward(JNIEnv* env, jclass, jlong stream, jint rows,
                                                                       jint width, jfloat epsilon, jlong input,
                                                                       jlong gain, jlong out, jlong inverseRms) {
  guarded(env, [&] {
    const prism::NormShape shape{rows, width, epsilon};
    const prism::NormForward arrays{device<const uint16_t>(input), device<const float>(gain), device<uint16_t>(out),
                                    device<float>(inverseRms)};
    prism::normForward(shape, arrays, reinterpret_cast<hipStream_t>(stream));
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_normBackward(JNIEnv* env, jclass, jlong stream, jint rows,
                                                                        jint width, jfloat epsilon, jlong input,
                                                                        jlong gain, jlong inverseRms, jlong gradOut,
                                                                        jlong gradInput, jlong gradGain) {
  guarded(env, [&] {
    const prism::NormShape shape{rows, width, epsilon};
    const prism::NormBackward arrays{device<const uint16_t>(input),   device<const float>(gain),
                                     device<const float>(inverseRms), device<const uint16_t>(gradOut),
                                     device<uint16_t>(gradInput),     device<float>(gradGain)};
    prism::normBackward(shape, arrays, reinterpret_cast<hipStream_t>(stream));
  });
}

}  // extern "C"
