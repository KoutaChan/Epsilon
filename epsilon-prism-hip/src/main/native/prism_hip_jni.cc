// com.epsilon.prism.hip.PrismHip の JNI の口。C++ の例外は Java の例外に替える（引数の誤りは IllegalArgumentException、
// それ以外は IllegalStateException）。
#include <jni.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "prism_hip_plan.h"
#include "prism_hip_runtime.h"
#include "prism_hip_training.h"

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

// 直の ByteBuffer の [position, position + bytes) のアドレス。
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

// Java が HipAttention で引いたデバイスのアドレス。
template <class T>
T* device(jlong address) {
  if (address == 0) {
    throw std::invalid_argument("device address must not be null");
  }
  return reinterpret_cast<T*>(address);
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

JNIEXPORT jobjectArray JNICALL Java_com_epsilon_prism_hip_PrismHip_runtimeInfo(JNIEnv* env, jclass) {
  return guarded(env, [&]() -> jobjectArray {
    const std::vector<std::string> info = prism::runtimeInfo();
    jobjectArray array = env->NewObjectArray(static_cast<jsize>(info.size()), env->FindClass("java/lang/String"), nullptr);
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
                                                                    jint hidden, jint ffn, jint gate, jint depth,
                                                                    jfloat epsilon, jint fixedTokens, jint rows,
                                                                    jint events, jint transitions, jint actions,
                                                                    jlong inputBytes, jlong weightBytes) {
  return guarded(env, [&]() -> jlong {
    const prism::PlanShape shape{hidden, ffn,   gate,        depth,   epsilon,          fixedTokens,
                                 rows,   events, transitions, actions, size(inputBytes), size(weightBytes)};
    return reinterpret_cast<jlong>(new Plan(device, reinterpret_cast<hipStream_t>(stream), shape));
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_freePlan(JNIEnv* env, jclass, jlong handle) {
  guarded(env, [&] { delete plan(handle); });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_uploadWeights(JNIEnv* env, jclass, jlong handle, jobject pack,
                                                                         jint position, jlong bytes, jlong version) {
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

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_run(JNIEnv* env, jclass, jlong handle, jint which, jint layer,
                                                               jint rows, jint events, jint transitions, jint actions,
                                                               jlong bytes, jlongArray sections) {
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

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_profile(JNIEnv* env, jclass, jlong handle, jboolean enabled) {
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
    jint segments, jint entries, jfloat scale, jlong query, jlong keyValue, jlong mask, jlong codes, jlong table,
    jlong out, jlong outFloat, jlong logSumExp) {
  guarded(env, [&] {
    const prism::AttentionShape shape{rows, queries, keys, codeQueries, heads, words, segments, entries, scale};
    const prism::AttentionForward arrays{device<const uint16_t>(query), device<const uint16_t>(keyValue),
                                         device<const float>(mask),     device<const int16_t>(codes),
                                         device<const float>(table),    device<uint16_t>(out),
                                         device<float>(outFloat),       device<float>(logSumExp)};
    prism::attentionForward(shape, arrays, reinterpret_cast<hipStream_t>(stream));
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_hip_PrismHip_attentionBackward(
    JNIEnv* env, jclass, jlong stream, jint rows, jint queries, jint keys, jint codeQueries, jint heads, jint words,
    jint segments, jint entries, jfloat scale, jlong query, jlong keyValue, jlong mask, jlong codes, jlong table,
    jlong outFloat, jlong logSumExp, jlong gradOut, jlong gradQuery, jlong gradKeyValue, jlong gradTable) {
  guarded(env, [&] {
    const prism::AttentionShape shape{rows, queries, keys, codeQueries, heads, words, segments, entries, scale};
    const prism::AttentionBackward arrays{device<const uint16_t>(query),    device<const uint16_t>(keyValue),
                                          device<const float>(mask),        device<const int16_t>(codes),
                                          device<const float>(table),       device<const float>(outFloat),
                                          device<const float>(logSumExp),   device<const uint16_t>(gradOut),
                                          device<uint16_t>(gradQuery),      device<uint16_t>(gradKeyValue),
                                          device<float>(gradTable)};
    prism::attentionBackward(shape, arrays, reinterpret_cast<hipStream_t>(stream));
  });
}

}  // extern "C"
