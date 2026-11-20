// com.epsilon.prism.hip.PrismHip の JNI の実装。C++ の例外は Java の例外に変える（引数の誤りは IllegalArgumentException、
// それ以外は IllegalStateException）。
#include <jni.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "batch_executor.h"
#include "lane_resources.h"
#include "prism_hip_runtime.h"
#include "inference_stages.h"
#include "prism_hip_training.h"

#include "../../../../common/src/main/native/collection/plan_arguments.h"

namespace {
using namespace prism::jni;

using prism::Plan;
}  // namespace

extern "C" {
#define PRISM_COLLECTION(name) Java_com_epsilon_prism_hip_PrismHip_##name
#include "../../../../common/src/main/native/collection/plan_bindings.inc"
#undef PRISM_COLLECTION
}  // extern "C"

#define PRISM_TRAINING(name) Java_com_epsilon_prism_hip_PrismHip_##name
#include "../../../../common/src/main/native/training/training_bindings.inc"
#undef PRISM_TRAINING

extern "C" {

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

JNIEXPORT jlong JNICALL Java_com_epsilon_prism_hip_PrismHip_tensorAddress(JNIEnv* env, jclass, jlong handle,
                                                                          jint element, jint gpu) {
  return guarded(env, [&]() -> jlong {
    return reinterpret_cast<jlong>(prism::tensorData(handle, static_cast<prism::TensorElement>(element), gpu));
  });
}

}  // extern "C"
