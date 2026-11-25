namespace {
// Java が配列から取り出したデバイスのアドレスを、型付きのポインタにする。
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
    throw std::invalid_argument("invalid attention address count");
  }
  std::array<jlong, N> result{};
  env->GetLongArrayRegion(values, 0, N, result.data());
  if (env->ExceptionCheck()) {
    throw std::invalid_argument("cannot read attention addresses");
  }
  return result;
}

prism::SegmentAttention segmentAttentionShape(JNIEnv* env, jintArray dimensions, float scale) {
  constexpr int inputCount = 8;
  constexpr int count = inputCount + prism::kSegmentLayoutInts;
  if (dimensions == nullptr || env->GetArrayLength(dimensions) != count) {
    throw std::invalid_argument("invalid attention dimensions");
  }
  std::array<jint, count> d{};
  env->GetIntArrayRegion(dimensions, 0, count, d.data());
  if (env->ExceptionCheck()) {
    throw std::invalid_argument("cannot read attention dimensions");
  }
  prism::SegmentLayout layout{};
  std::memcpy(&layout, d.data() + inputCount, sizeof(layout));
  return {{d[0], d[1], d[2], d[3], d[4], d[5], layout}, d[6], d[7], scale};
}

prism::SegmentAttentionArrays segmentAttentionInputs(JNIEnv* env, jlongArray values) {
  const auto a = addresses<6>(env, values);
  return {device<const float>(a[0]), device<const float>(a[1]),   device<const float>(a[2]),
          device<const float>(a[3]), device<const uint8_t>(a[4]), device<const int32_t>(a[5])};
}

}

extern "C" {
JNIEXPORT void JNICALL PRISM_TRAINING(segmentAttentionForward)(JNIEnv* env, jobject, jint deviceIndex, jlong stream,
                                                               jintArray dimensions, jfloat scale, jlongArray inputs,
                                                               jlong output, jlong probabilities) {
  guarded(env, [&] {
    prism::gpu::SelectedDevice selected(deviceIndex);
    prism::launchSegmentAttention(
        segmentAttentionShape(env, dimensions, scale), segmentAttentionInputs(env, inputs), device<float>(output),
        device<float>(probabilities), prism::trainingStream(reinterpret_cast<prism::gpu::Stream>(stream)));
  });
}

JNIEXPORT void JNICALL PRISM_TRAINING(segmentAttentionBackward)(JNIEnv* env, jobject, jint deviceIndex, jlong stream,
                                                                jintArray dimensions, jfloat scale, jlongArray inputs,
                                                                jlongArray gradients) {
  guarded(env, [&] {
    prism::gpu::SelectedDevice selected(deviceIndex);
    const auto g = addresses<11>(env, gradients);
    const prism::SegmentAttentionBackward b{
        device<const float>(g[0]), device<const float>(g[1]), device<float>(g[2]),   device<float>(g[3]),
        device<float>(g[4]),       device<float>(g[5]),       device<float>(g[6]),   device<float>(g[7]),
        device<int32_t>(g[8]),     device<int32_t>(g[9]),     device<int32_t>(g[10])};
    prism::launchSegmentAttentionBackward(segmentAttentionShape(env, dimensions, scale),
                                              segmentAttentionInputs(env, inputs), b,
                                              prism::trainingStream(reinterpret_cast<prism::gpu::Stream>(stream)));
  });
}

JNIEXPORT void JNICALL PRISM_TRAINING(countForward)(JNIEnv* env, jobject, jint deviceIndex, jlong stream, jint rows,
                                                    jint transitions, jint tiles, jint codes, jint width,
                                                    jlong projected, jlong gates, jlong rowIndex, jlong ids,
                                                    jlong output) {
  guarded(env, [&] {
    prism::gpu::SelectedDevice selected(deviceIndex);
    prism::launchCountTraining({rows, transitions, tiles, codes, width}, device<const uint16_t>(projected),
                                   device<const float>(gates), device<const int32_t>(rowIndex),
                                   device<const int32_t>(ids), device<uint16_t>(output),
                                   prism::trainingStream(reinterpret_cast<prism::gpu::Stream>(stream)));
  });
}

JNIEXPORT void JNICALL PRISM_TRAINING(countBackward)(JNIEnv* env, jobject, jint deviceIndex, jlong stream, jint rows,
                                                     jint transitions, jint tiles, jint codes, jint width,
                                                     jlong projected, jlong gates, jlong rowIndex, jlong ids,
                                                     jlong gradOutput, jlong gradProjected, jlong gradGate) {
  guarded(env, [&] {
    prism::gpu::SelectedDevice selected(deviceIndex);
    prism::launchCountTrainingBackward({rows, transitions, tiles, codes, width}, device<const uint16_t>(projected),
                                           device<const float>(gates), device<const int32_t>(rowIndex),
                                           device<const int32_t>(ids), device<const uint16_t>(gradOutput),
                                           device<uint16_t>(gradProjected), device<float>(gradGate),
                                           prism::trainingStream(reinterpret_cast<prism::gpu::Stream>(stream)));
  });
}

JNIEXPORT void JNICALL PRISM_TRAINING(relationCodes)(JNIEnv* env, jobject, jint deviceIndex, jlong stream, jint rows,
                                                     jint events, jint tokens, jint seats, jint tiles, jint rowWidth,
                                                     jint rowFirstEvent, jint rowEvents, jint width, jint tile,
                                                     jint owner, jint event, jint riichi, jint lastDiscard, jint link,
                                                     jint safety, jint safeties, jlong relation, jlong rowColumn,
                                                     jlong tileRelation, jlong codes) {
  guarded(env, [&] {
    prism::gpu::SelectedDevice selected(deviceIndex);
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
    prism::relationCodes(shape, arrays, prism::trainingStream(reinterpret_cast<prism::gpu::Stream>(stream)));
  });
}

JNIEXPORT void JNICALL PRISM_TRAINING(attentionForward)(JNIEnv* env, jobject, jint deviceIndex, jlong stream, jint rows,
                                                        jint queries, jint keys, jint codeQueries, jint heads,
                                                        jint words, jint segments, jint entries, jfloat scale,
                                                        jint fixed, jlong query, jlong keyValue, jlong mask,
                                                        jlong codes, jlong table, jlong out, jlong outFloat,
                                                        jlong logSumExp) {
  guarded(env, [&] {
    prism::gpu::SelectedDevice selected(deviceIndex);
    const prism::AttentionShape shape{rows, queries, keys, codeQueries, heads, words, segments, entries, scale};
    const prism::AttentionForward arrays{device<const uint16_t>(query), device<const uint16_t>(keyValue),
                                         device<const float>(mask),     device<const int16_t>(codes),
                                         device<const float>(table),    device<uint16_t>(out),
                                         device<float>(outFloat),       device<float>(logSumExp)};
    const auto activeStream = prism::trainingStream(reinterpret_cast<prism::gpu::Stream>(stream));
    if (fixed != 0) {
      prism::causalAttentionForward(shape, fixed, arrays, activeStream);
    } else {
      prism::attentionForward(shape, arrays, activeStream);
    }
  });
}

JNIEXPORT void JNICALL PRISM_TRAINING(attentionBackward)(JNIEnv* env, jobject, jint deviceIndex, jlong stream,
                                                         jint rows, jint queries, jint keys, jint codeQueries,
                                                         jint heads, jint words, jint segments, jint entries,
                                                         jfloat scale, jint fixed, jlong query, jlong keyValue,
                                                         jlong mask, jlong codes, jlong table, jlong outFloat,
                                                         jlong logSumExp, jlong gradOut, jlong gradQuery,
                                                         jlong gradKeyValue, jlong gradTable) {
  guarded(env, [&] {
    prism::gpu::SelectedDevice selected(deviceIndex);
    const prism::AttentionShape shape{rows, queries, keys, codeQueries, heads, words, segments, entries, scale};
    const prism::AttentionBackward arrays{
        device<const uint16_t>(query),  device<const uint16_t>(keyValue), device<const float>(mask),
        device<const int16_t>(codes),   device<const float>(table),       device<const float>(outFloat),
        device<const float>(logSumExp), device<const uint16_t>(gradOut),  device<uint16_t>(gradQuery),
        device<uint16_t>(gradKeyValue), device<float>(gradTable)};
    const auto activeStream = prism::trainingStream(reinterpret_cast<prism::gpu::Stream>(stream));
    if (fixed != 0) {
      prism::causalAttentionBackward(shape, fixed, arrays, activeStream);
    } else {
      prism::attentionBackward(shape, arrays, activeStream);
    }
  });
}

JNIEXPORT void JNICALL PRISM_TRAINING(normForward)(JNIEnv* env, jobject, jint deviceIndex, jlong stream, jint rows,
                                                   jint width, jfloat epsilon, jlong input, jlong gain, jlong out,
                                                   jlong inverseRms) {
  guarded(env, [&] {
    prism::gpu::SelectedDevice selected(deviceIndex);
    const prism::NormShape shape{rows, width, epsilon};
    const prism::NormForward arrays{device<const uint16_t>(input), device<const float>(gain), device<uint16_t>(out),
                                    device<float>(inverseRms)};
    prism::normForward(shape, arrays, prism::trainingStream(reinterpret_cast<prism::gpu::Stream>(stream)));
  });
}

JNIEXPORT void JNICALL PRISM_TRAINING(normBackward)(JNIEnv* env, jobject, jint deviceIndex, jlong stream, jint rows,
                                                    jint width, jfloat epsilon, jlong input, jlong gain,
                                                    jlong inverseRms, jlong gradOut, jlong gradInput, jlong gradGain) {
  guarded(env, [&] {
    prism::gpu::SelectedDevice selected(deviceIndex);
    const prism::NormShape shape{rows, width, epsilon};
    const prism::NormBackward arrays{device<const uint16_t>(input),   device<const float>(gain),
                                     device<const float>(inverseRms), device<const uint16_t>(gradOut),
                                     device<uint16_t>(gradInput),     device<float>(gradGain)};
    prism::normBackward(shape, arrays, prism::trainingStream(reinterpret_cast<prism::gpu::Stream>(stream)));
  });
}

}  // extern "C"
