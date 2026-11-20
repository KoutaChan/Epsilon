JNIEXPORT jlong JNICALL PRISM_COLLECTION(declaredPlan)(JNIEnv* env, jobject, jint device, jlong stream,
                                                       jobject declaration, jint length, jint abi, jint countOffset,
                                                       jint segmentsOffset, jint rows, jint events, jint transitions,
                                                       jint actions, jlong inputBytes, jlong weightBytes) {
  return guarded(env, [&]() -> jlong {
    prism::PlanShape capacity{};
    capacity.rows = rows;
    capacity.events = events;
    capacity.transitions = transitions;
    capacity.actions = actions;
    capacity.inputBytes = size(inputBytes);
    capacity.weightBytes = size(weightBytes);
    auto data = reinterpret_cast<const uint8_t*>(direct(env, declaration, 0, size(length)));
    auto shape = prism::declaredPlan(data, size(length), abi, capacity, countOffset, segmentsOffset);
    return reinterpret_cast<jlong>(new Plan(device, reinterpret_cast<prism::gpu::Stream>(stream), shape));
  });
}

JNIEXPORT void JNICALL PRISM_COLLECTION(freePlan)(JNIEnv* env, jobject, jlong handle) {
  guarded(env, [&] { delete plan(handle); });
}

JNIEXPORT void JNICALL PRISM_COLLECTION(uploadWeights)(JNIEnv* env, jobject, jlong handle, jobject pack, jint position,
                                                       jlong bytes, jlong version) {
  guarded(env, [&] { plan(handle)->uploadWeights(direct(env, pack, position, size(bytes)), size(bytes), version); });
}

JNIEXPORT jlong JNICALL PRISM_COLLECTION(publishedVersion)(JNIEnv* env, jobject, jlong handle) {
  return guarded(env, [&]() -> jlong { return plan(handle)->publishedVersion(); });
}

JNIEXPORT void JNICALL PRISM_COLLECTION(forward)(JNIEnv* env, jobject, jlong handle, jobject input, jint inputPosition,
                                                 jint rows, jint events, jint transitions, jint actions, jlong bytes,
                                                 jlongArray sections, jobject scores, jint scoresPosition,
                                                 jlong version) {
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

JNIEXPORT void JNICALL PRISM_COLLECTION(forwardPaged)(JNIEnv* env, jobject, jlong handle, jobject input,
                                                      jint inputPosition, jint rows, jint events, jint transitions,
                                                      jint actions, jlong bytes, jlongArray sections, jobject scores,
                                                      jint scoresPosition, jlong version, jobject eventMap, jint misses,
                                                      jint slots, jint transferBytes) {
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

JNIEXPORT void JNICALL PRISM_COLLECTION(configureCandidateInput)(JNIEnv* env, jobject, jlong handle,
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

// レーンの実効の容量を LaneCapacity の宣言の順に返す（辞書と分割は 1 か 0）。
JNIEXPORT jlongArray JNICALL PRISM_COLLECTION(configureEventPages)(JNIEnv* env, jobject, jlong handle, jint remaining,
                                                                   jint events, jdouble reserveFraction) {
  return guarded(env, [&]() -> jlongArray {
    const auto c = plan(handle)->configureEventPages(remaining, events, reserveFraction);
    const std::array<jlong, 10> values{
        c.freeBefore,   c.freeAfter, c.total,           c.pageSlots,          c.projectionRows,
        c.stagingBytes, c.pageBytes, c.projectionBytes, c.dictionary ? 1 : 0, c.splitCandidate ? 1 : 0};
    jlongArray result = env->NewLongArray(values.size());
    if (result != nullptr) env->SetLongArrayRegion(result, 0, values.size(), values.data());
    return result;
  });
}
