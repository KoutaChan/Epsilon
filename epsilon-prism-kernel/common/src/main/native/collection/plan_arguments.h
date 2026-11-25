#pragma once
#include "batch_executor.h"
#include <jni.h>
#include <array>
#include <stdexcept>

namespace prism::jni {
inline void throwJava(JNIEnv* env, const char* type, const char* message) {
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
    throwJava(env, "java/lang/IllegalStateException", "Unknown native collection error");
  }
  return decltype(body())();
}

inline Plan* plan(jlong handle) {
  if (handle == 0) {
    throw std::invalid_argument("The collection plan is closed");
  }
  return reinterpret_cast<Plan*>(handle);
}

// direct ByteBuffer の [position, position + bytes) の先頭アドレスを返す。
inline char* direct(JNIEnv* env, jobject buffer, jint position, size_t bytes) {
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

inline size_t size(jlong bytes) {
  if (bytes < 0) {
    throw std::invalid_argument("byte count must not be negative");
  }
  return static_cast<size_t>(bytes);
}

inline prism::BatchShape batch(JNIEnv* env, jint rows, jint events, jint transitions, jint actions, jlong bytes,
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
}
