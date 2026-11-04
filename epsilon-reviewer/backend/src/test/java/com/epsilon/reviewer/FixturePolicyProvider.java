package com.epsilon.reviewer;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.spi.ReviewEvaluation;
import com.epsilon.spi.ReviewPolicy;
import com.epsilon.spi.ReviewPolicyProvider;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 試験のモデルの系列（試験の ServiceLoader に登録する）。価値は合法手の番号から決める。options の hold が true のモデルは試験が
 * {@link #release} を開けるまで評価の途中で待ち、broken が true のモデルは開けない。
 */
public final class FixturePolicyProvider implements ReviewPolicyProvider {
  public static final String SERIES = "fixture-scenario";

  /** hold のモデルの評価が始まった印。 */
  public static volatile CountDownLatch entered = new CountDownLatch(0);

  /** hold のモデルの評価を進める門。 */
  public static volatile CountDownLatch release = new CountDownLatch(0);

  /** 開いた方策の数。 */
  public static final AtomicInteger opened = new AtomicInteger();

  /** 閉じた方策の数。 */
  public static final AtomicInteger closed = new AtomicInteger();

  @Override
  public String seriesId() {
    return SERIES;
  }

  @Override
  public ReviewPolicy openReview(Path checkpoint, Map<String, String> options) {
    if (Boolean.parseBoolean(options.get("broken")))
      throw new IllegalArgumentException("The fixture checkpoint is broken.");
    boolean hold = Boolean.parseBoolean(options.get("hold"));
    opened.incrementAndGet();
    return new ReviewPolicy() {
      @Override
      public ReviewEvaluation evaluate(PublicObservation observation, List<Action> legalActions) {
        if (hold) {
          entered.countDown();
          try {
            release.await();
          } catch (InterruptedException failure) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(failure);
          }
        }
        float[] values = new float[legalActions.size()];
        for (int i = 0; i < values.length; i++) values[i] = -0.001f * legalActions.get(i).toIndex();
        return ReviewEvaluation.valued(
            values, ReviewEvaluation.softmax(values, ReviewEvaluation.DISPLAY_TEMPERATURE));
      }

      @Override
      public void close() {
        closed.incrementAndGet();
      }
    };
  }
}
