package com.epsilon.major.ai.decision.runtime;

import ai.djl.pytorch.jni.JniUtils;
import com.epsilon.major.ai.decision.input.DecisionBucket;
import com.epsilon.major.config.settings.DecisionInferenceSettings;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/** 指定した容量区分について演算ごとの実行時間を一度だけ計測する。 */
public final class DecisionOperatorProfiler {

  private static final AtomicInteger CALLS = new AtomicInteger();
  private static final AtomicBoolean CLAIMED = new AtomicBoolean();

  private final String outputFile;
  private final int minimumRows;
  private final int afterCalls;
  private final int transitionCapacity;

  DecisionOperatorProfiler(DecisionInferenceSettings settings) {
    outputFile = settings.profileFile().isBlank() ? null : settings.profileFile();
    minimumRows = settings.profileMinimumRows();
    afterCalls = settings.profileAfterCalls();
    transitionCapacity = settings.profileTransitionCapacity();
  }

  /** 記録完了後に呼び、固定再生の順伝播回数だけで計測結果開始位置を決める。 */
  public static void reset() {
    CALLS.set(0);
    CLAIMED.set(false);
  }

  boolean claim(DecisionBucket bucket, int rowCount) {
    if (outputFile == null || rowCount < minimumRows) {
      return false;
    }
    if (transitionCapacity > 0 && bucket.actionTransitionCapacity() != transitionCapacity) {
      return false;
    }
    return CALLS.incrementAndGet() >= afterCalls && CLAIMED.compareAndSet(false, true);
  }

  void start() {
    JniUtils.startProfile(true, true, false);
  }

  void stop() {
    JniUtils.stopProfile(outputFile);
  }
}
