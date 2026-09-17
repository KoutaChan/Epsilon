package com.epsilon.nano.ai.decision.runtime;

import ai.djl.pytorch.jni.JniUtils;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.config.settings.DecisionInferenceSettings;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/** 指定した容量区分の推論について、演算ごとの処理時間を実行全体で一度だけ計測する。 */
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

  /** 記録完了後に呼び、固定した再生の順伝播回数だけでプロファイル開始位置を決める。 */
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
