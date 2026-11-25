#include "plan_arguments.h"
#include "device_stream.h"

using prism::Plan;
using namespace prism::jni;

extern "C" {
#define PRISM_COLLECTION(name) Java_com_epsilon_prism_cuda_CudaCollection_##name
#include "plan_bindings.inc"
#undef PRISM_COLLECTION

JNIEXPORT jstring JNICALL Java_com_epsilon_prism_cuda_CudaCollection_deviceLimitation(JNIEnv* env, jclass,
                                                                                      jint device) {
  return guarded(env, [&]() -> jstring {
    cudaDeviceProp props{};
    prism::runtime::checkCuda(cudaGetDeviceProperties(&props, device), "query CUDA device");
    if (props.major < 8 || props.warpSize != 32)
      return env->NewStringUTF("CUDA inference requires BF16 MMA and 32 lanes per warp");
    const int capability = 10 * props.major + props.minor;
    constexpr int targets[] = {__CUDA_ARCH_LIST__};
    std::string compiled;
    for (int target : targets) {
      if (target / 10 == capability) return nullptr;
      if (!compiled.empty()) compiled += ", ";
      compiled += "sm_" + std::to_string(target / 10);
    }
    const std::string reason =
        "CUDA device " + std::to_string(device) + " is sm_" + std::to_string(capability) + "; built for " + compiled;
    return env->NewStringUTF(reason.c_str());
  });
}

JNIEXPORT void JNICALL Java_com_epsilon_prism_cuda_CudaCollection_requireStream(JNIEnv* env, jclass, jint device,
                                                                                jlong stream) {
  guarded(env, [&] { prism::runtime::DeviceStream lane(device, reinterpret_cast<cudaStream_t>(stream)); });
}
}
