#pragma once
#include <jni.h>
#include "model_schema.h"

namespace prism::schema {
// 共有ライブラリの JNI の口（native のメソッドの名前と引数）のバージョン。Java の NativeDeclarations と同じ値でなければならない。
// 口を変えたら上げる。
inline constexpr int32_t jniVersion =
#include "jni_version.inc"
    ;
}  // namespace prism::schema
