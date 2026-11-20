#include "declaration_jni.h"

extern "C" JNIEXPORT jint JNICALL Java_com_epsilon_prism_hip_HipDeclarations_abiVersion(JNIEnv*, jclass) {
  return prism::schema::version;
}
extern "C" JNIEXPORT jint JNICALL Java_com_epsilon_prism_hip_HipDeclarations_jniVersion(JNIEnv*, jclass) {
  return prism::schema::jniVersion;
}
