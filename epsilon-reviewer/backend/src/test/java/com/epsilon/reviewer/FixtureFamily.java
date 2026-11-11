package com.epsilon.reviewer;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.model.ModelFamilies;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.OptionalDouble;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * テスト用のモデルファミリー。価値は合法手のコードから決める。チェックポイントに hold の印のファイルがあるモデルは、評価の途中で、
 * テストが {@link #release} を開くまで待つ。broken の印のファイルがあるモデルは開けない。
 */
public final class FixtureFamily implements ModelFamily {
  /** テスト用のモデルが名乗る系列。 */
  public static final Series SERIES = Series.NANO;

  /** テスト用のモデルの方策の温度（PRISM の方策と同じ値）。 */
  public static final double TEMPERATURE = 0.002;

  /** 置くと、評価の途中で待つようになる印のファイル。 */
  public static final String HOLD = "hold";

  /** 置くとモデルを開けなくなる印のファイル。中身が io なら IOException、linkage なら LinkageError、ほかは IllegalStateException を投げる。 */
  public static final String BROKEN = "broken";

  /** hold のモデルの評価が始まったことを知らせるラッチ。 */
  public static volatile CountDownLatch entered = new CountDownLatch(0);

  /** hold のモデルの評価を先へ進めるゲート。 */
  public static volatile CountDownLatch release = new CountDownLatch(0);

  /** 開いた方策の数。 */
  public static final AtomicInteger opened = new AtomicInteger();

  /** 閉じた方策の数。 */
  public static final AtomicInteger closed = new AtomicInteger();

  /**
   * 方策が作った作業領域の数。Nano・Major の方策と同じように、評価を呼んだスレッドごとに作業領域を 1 つ作り、方策を閉じるまで持つ。
   */
  public static final AtomicInteger workspaces = new AtomicInteger();

  /** テスト用のファミリーだけを登録した表。 */
  public static ModelFamilies families() {
    return ModelFamilies.of(List.of(new FixtureFamily()));
  }

  @Override
  public Series series() {
    return SERIES;
  }

  @Override
  public Policy open(Path checkpoint, Variant variant, PolicyOptions options, NetworkRuntime runtime)
      throws IOException {
    Path broken = checkpoint.resolve(BROKEN);
    if (Files.exists(broken))
      switch (Files.readString(broken)) {
        case "io" -> throw new IOException("The fixture checkpoint cannot be read.");
        case "linkage" -> throw new LinkageError("The fixture family is missing a class.");
        default -> throw new IllegalStateException("The fixture checkpoint is broken.");
      }
    boolean hold = Files.exists(checkpoint.resolve(HOLD));
    String label = SERIES.label(checkpoint);
    opened.incrementAndGet();
    return new InlinePolicy() {
      private final ThreadLocal<Integer> workspace =
          ThreadLocal.withInitial(workspaces::incrementAndGet);

      @Override
      public Variant variant() {
        return variant;
      }

      @Override
      public String label() {
        return label;
      }

      @Override
      public int act(Observation observation) {
        return evaluate(observation).greedy();
      }

      @Override
      public Evaluation evaluate(Observation observation) {
        if (observation.variant() != variant)
          throw new IllegalArgumentException("The fixture reads " + variant);
        workspace.get();
        if (hold) {
          entered.countDown();
          try {
            release.await();
          } catch (InterruptedException failure) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(failure);
          }
        }
        return valued(observation);
      }

      @Override
      public OptionalDouble temperature() {
        return OptionalDouble.of(TEMPERATURE);
      }

      @Override
      public void close() {
        closed.incrementAndGet();
      }
    };
  }

  /** 価値は合法手のコードの -0.001 倍。確率は、価値を方策の温度で割った softmax。 */
  static Evaluation valued(Observation observation) {
    int count = observation.legalActions();
    float[] values = new float[count];
    int greedy = 0;
    for (int i = 0; i < count; i++) {
      values[i] = -0.001f * observation.action(i).code();
      if (values[i] > values[greedy]) greedy = i;
    }
    double[] weights = new double[count];
    double sum = 0;
    for (int i = 0; i < count; i++) {
      weights[i] = Math.exp((values[i] - values[greedy]) / TEMPERATURE);
      sum += weights[i];
    }
    float[] probabilities = new float[count];
    for (int i = 0; i < count; i++) probabilities[i] = (float) (weights[i] / sum);
    return new Evaluation(probabilities, values, greedy);
  }
}
