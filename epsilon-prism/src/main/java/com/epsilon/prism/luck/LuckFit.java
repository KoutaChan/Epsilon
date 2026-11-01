package com.epsilon.prism.luck;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.Pair;
import com.epsilon.core.GameState;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismShape;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 運の網 Ω の当てはめと保留の R²。
 *
 * <p>標本（{@link LuckSamples}）を半荘ごとに当てはめと保留（既定 5%）に分け、当てはめの標本を並べ替えて塊ずつ 1 歩ずつ当てる。目標は実現した
 * 局末の Φ（その経路の MC）で、損失は Σ_i w_i Σ_r (V(x_i, d_i)[r] − Ỹ_ir)² / Σ_i w_i（w = 1/標本の率、Ỹ は 4 席の和を 0 にした Y。
 * {@link LuckHead}）。最適化は AdamW（学習率は一定、重みの減衰あり）で、GPU では BF16、CPU では FP32 で計算する。
 *
 * <p>保留の R²（{@link Score}）は LedgerFit と同じ考えで、「行の中で中心化した Y」の分散が c でどれだけ減るかを測る。Ω の標本の行は 1 つの出来事
 * で、行の中の手に当たるのはその出来事の結果 37 通り（同じ出来事の前の状態 x を共有する）。行の平均は Σ_t p(t)·V(x, t) なので、
 *
 * <pre>
 * ỹ = Ỹ − Σ_t p(t)·V(x, t)          行の中で中心化した Y
 * R² = 1 − Σ w·‖ỹ − c‖² / Σ w·‖ỹ‖²   （ỹ − c = Ỹ − V(x, d)）
 * </pre>
 *
 * 白紙の Ω（c ≡ 0）では厳密に 0。分子の減り Σ w·(‖ỹ‖² − ‖ỹ − c‖²) = Σ w·(2ỹ·c − ‖c‖²) は、E[c | x] = 0 なので期待では行の平均（V の出来）に
 * 依らない（{@link Score#reduction}）。版どうしの比べにはこちらが向く。
 *
 * <p>物差しの注意: 1 つの出来事が、その出来事の後に残る Y のばらつきのうち説明する割合なので、行（判断）の全ての出来事を合わせた帳簿の R²
 * （LedgerFit）とは物差しが違い、ずっと小さく出る。出来事ごとの減りは、マルチンゲールの増分の直交性から経路の出来事について足し
 * 合わせられる（兄弟の分岐の共通乱数の重なりを除く）。偏りの確かめに、保留の行為者の c の平均の z（半荘で束ねる）も出す。
 */
public final class LuckFit {

  private static final Logger LOG = LoggerFactory.getLogger(LuckFit.class);

  /** 先に作っておく塊の数。 */
  private static final int PREFETCH = 2;

  /** 進みを記録する歩の間隔。 */
  private static final int PROGRESS_EVERY = 100;

  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final LuckEvent[] EVENTS = LuckEvent.values();

  private LuckFit() {}

  /**
   * 当てはめの計画。
   *
   * @param shape 網の形（{@link LuckNetwork#shape}）
   * @param epochs 当てはめの標本の周回数
   * @param batchSize 塊の標本の数
   * @param learningRate AdamW の学習率（一定）
   * @param weightDecay AdamW の重みの減衰
   * @param holdout 保留にする半荘の割合
   * @param seed 網の初期値と並べ替えの種
   */
  public record Plan(
      PrismShape shape,
      double epochs,
      int batchSize,
      float learningRate,
      float weightDecay,
      double holdout,
      long seed) {

    /** 塊の標本の数の既定。 */
    public static final int BATCH_SIZE = 8192;

    /** 学習率の既定。 */
    public static final float LEARNING_RATE = 2e-4f;

    /** 重みの減衰の既定。 */
    public static final float WEIGHT_DECAY = 0.01f;

    /** 保留の割合の既定（標本の半荘の 5%）。 */
    public static final double HOLDOUT = 0.05;

    /**
     * 計画を確かめる。
     *
     * @throws IllegalArgumentException 値が範囲の外のとき
     */
    public Plan {
      Objects.requireNonNull(shape, "shape");
      if (shape.depth() != LuckNetwork.DEPTH) {
        throw new IllegalArgumentException("the luck network has depth " + LuckNetwork.DEPTH);
      }
      if (!(epochs > 0) || batchSize < 1 || !(learningRate > 0) || weightDecay < 0) {
        throw new IllegalArgumentException(
            "invalid luck fit plan: epochs="
                + epochs
                + " batch="
                + batchSize
                + " lr="
                + learningRate
                + " wd="
                + weightDecay);
      }
      if (!(holdout > 0 && holdout < 1)) {
        throw new IllegalArgumentException("holdout must be in (0, 1): " + holdout);
      }
    }

    /**
     * 既定の計画（塊 {@value #BATCH_SIZE}、学習率 2e-4 一定、減衰 0.01、保留 5%）。
     *
     * @param width 網の幅
     * @param epochs 周回数
     * @param seed 種
     * @return 計画
     */
    public static Plan standard(int width, double epochs, long seed) {
      return new Plan(
          LuckNetwork.shape(width), epochs, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, HOLDOUT, seed);
    }
  }

  /**
   * 保留の区分の採点。
   *
   * @param samples 標本の数
   * @param variance 行の中で中心化した Y の 4 席の平方の和の重みつきの平均（Σ w·‖ỹ‖² / Σ w）
   * @param residual c を引いた後の同じ平均（Σ w·‖ỹ − c‖² / Σ w）
   * @param r2 1 − residual / variance（標本がなければ NaN）
   */
  public record Score(long samples, double variance, double residual, double r2) {

    /**
     * 出来事 1 つあたりの分散の減り variance − residual を返す（期待では行の平均の出来に依らない）。
     *
     * @return 減り
     */
    public double reduction() {
      return variance - residual;
    }
  }

  /**
   * 保留の採点。
   *
   * @param all 全体
   * @param byEvent 出来事の種類ごと（標本のない種類は載せない）
   * @param actorMeanZ 行為者の c の重みつきの平均の z（半荘で束ねる。c ≡ 0 なら NaN）
   */
  public record Holdout(Score all, Map<LuckEvent, Score> byEvent, double actorMeanZ) {}

  /**
   * 当てはめの報告。
   *
   * @param plan 計画
   * @param trainSamples 当てはめの標本の数
   * @param steps 歩数
   * @param firstLoss 最初の歩の損失
   * @param lastLoss 最後の歩の損失
   * @param meanLoss 全ての歩の損失の平均
   * @param holdout 保留の採点（当てはめの後）
   * @param seconds 当てはめと採点の秒
   */
  public record Report(
      Plan plan,
      long trainSamples,
      int steps,
      double firstLoss,
      double lastLoss,
      double meanLoss,
      Holdout holdout,
      double seconds) {}

  /**
   * 網を当て、保留で採点する。重みは網に残る（保存は呼び手）。
   *
   * @param samples 標本の表
   * @param network 網（パラメーターは manager の上）
   * @param manager 網のパラメーターの管理元（デバイスもここで決まる）
   * @param plan 計画
   * @param threads 符号化のスレッドの数
   * @return 報告
   * @throws IllegalArgumentException 当てはめの標本がないとき
   */
  public static Report run(
      LuckSamples samples, LuckNetwork network, NDManager manager, Plan plan, int threads) {
    long started = System.nanoTime();
    int[][] split = split(samples, plan.holdout());
    int[] train = split[0];
    if (train.length == 0) {
      throw new IllegalArgumentException("no luck samples to fit");
    }
    int[] order = order(train, plan.epochs(), plan.seed());
    int steps = Math.ceilDiv(order.length, plan.batchSize());
    Precision precision = precision(manager);
    ParameterStore store = new ParameterStore(manager, false);
    Optimizer optimizer = optimizer(plan.learningRate(), plan.weightDecay());
    LOG.info(
        "luck fit samples={} train={} holdout={} steps={} shape={} precision={} device={}",
        samples.size(),
        train.length,
        split[1].length,
        steps,
        plan.shape(),
        precision,
        manager.getDevice());
    double[] losses = new double[steps];
    Holdout holdout;
    try (LuckFeed feed = new LuckFeed(samples, plan.batchSize(), PREFETCH + 1, threads)) {
      int[] step = {0};
      forEach(
          feed,
          order,
          plan.batchSize(),
          rows -> {
            losses[step[0]] = step(network, store, manager, rows, precision, optimizer);
            if (step[0] % PROGRESS_EVERY == 0) {
              LOG.info(
                  "luck fit step {}/{} loss {} {}s",
                  step[0] + 1,
                  steps,
                  losses[step[0]],
                  String.format("%.1f", (System.nanoTime() - started) / 1e9));
            }
            step[0]++;
          });
      holdout = score(feed, split[1], network, store, manager, plan.batchSize());
    }
    return new Report(
        plan,
        train.length,
        steps,
        losses[0],
        losses[steps - 1],
        Arrays.stream(losses).average().orElse(Double.NaN),
        holdout,
        (System.nanoTime() - started) / 1e9);
  }

  /**
   * 保留の標本で網を採点する（重みは変えない）。
   *
   * @param samples 標本の表
   * @param network 網
   * @param manager 網のパラメーターの管理元
   * @param holdout 保留にする半荘の割合
   * @param batchSize 塊の標本の数
   * @param threads 符号化のスレッドの数
   * @return 保留の採点
   */
  public static Holdout score(
      LuckSamples samples,
      LuckNetwork network,
      NDManager manager,
      double holdout,
      int batchSize,
      int threads) {
    try (LuckFeed feed = new LuckFeed(samples, batchSize, PREFETCH + 1, threads)) {
      return score(
          feed,
          split(samples, holdout)[1],
          network,
          new ParameterStore(manager, false),
          manager,
          batchSize);
    }
  }

  /**
   * 1 塊で 1 回更新する。
   *
   * @return 更新の前の損失
   */
  static double step(
      LuckNetwork network,
      ParameterStore store,
      NDManager manager,
      LuckRows rows,
      Precision precision,
      Optimizer optimizer) {
    double value;
    try (NDManager scope = manager.newSubManager()) {
      LuckInputs inputs = rows.inputs(scope);
      LuckTargets targets = rows.targets(scope);
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        NDArray loss = loss(network.forward(store, inputs, precision, true), targets);
        collector.backward(loss);
        value = loss.getFloat();
      }
    }
    for (Pair<String, Parameter> pair : network.getParameters()) {
      NDArray array = pair.getValue().getArray();
      try (NDArray gradient = array.getGradient()) {
        optimizer.update(pair.getValue().getId(), array, gradient);
        gradient.fillI(0);
      }
    }
    return value;
  }

  /**
   * 損失 Σ_i w_i Σ_r (V(x_i, d_i)[r] − Ỹ_ir)² / Σ_i w_i（Ỹ は 4 席の和を 0 にした Y）。
   *
   * @param values 網の出力 V [行, 37, 4]
   * @param targets 目標
   * @return 損失（スカラー、FP32）
   */
  static NDArray loss(NDArray values, LuckTargets targets) {
    NDArray y = targets.y();
    NDArray centered = y.sub(y.mean(new int[] {1}, true));
    NDArray error =
        ControlVariate.chosen(values, targets.chosen()).sub(centered).square().sum(new int[] {1});
    return error.mul(targets.weight()).sum().div(targets.weight().sum());
  }

  /**
   * AdamW（学習率は一定）を作る。
   *
   * @param learningRate 学習率
   * @param weightDecay 重みの減衰
   * @return 最適化
   */
  static Optimizer optimizer(float learningRate, float weightDecay) {
    return Optimizer.adamW()
        .optLearningRateTracker(Tracker.fixed(learningRate))
        .optWeightDecays(weightDecay)
        .build();
  }

  /** デバイスの計算の型（GPU は BF16、CPU は FP32）。 */
  static Precision precision(NDManager manager) {
    return manager.getDevice().isGpu() ? Precision.BFLOAT16 : Precision.FLOAT32;
  }

  private static Holdout score(
      LuckFeed feed,
      int[] selected,
      LuckNetwork network,
      ParameterStore store,
      NDManager manager,
      int batchSize) {
    Precision precision = precision(manager);
    Accumulator all = new Accumulator();
    Map<LuckEvent, Accumulator> byEvent = new EnumMap<>(LuckEvent.class);
    Map<Long, Double> clusters = new HashMap<>();
    forEach(
        feed,
        selected,
        batchSize,
        rows -> {
          float[] chosen;
          float[] expected;
          try (NDManager scope = manager.newSubManager()) {
            LuckInputs inputs = rows.inputs(scope);
            LuckTargets targets = rows.targets(scope);
            NDArray values = network.forward(store, inputs, precision, false);
            chosen = ControlVariate.chosen(values, targets.chosen()).toFloatArray();
            expected = ControlVariate.expected(values, targets.pool()).toFloatArray();
          }
          for (int row = 0; row < rows.rows(); row++) {
            int at = row * SEATS;
            double mean = 0;
            for (int seat = 0; seat < SEATS; seat++) {
              mean += rows.y[at + seat];
            }
            mean /= SEATS;
            double variance = 0;
            double residual = 0;
            for (int seat = 0; seat < SEATS; seat++) {
              double centered = rows.y[at + seat] - mean;
              variance += square(centered - expected[at + seat]);
              residual += square(centered - chosen[at + seat]);
            }
            double weight = rows.weight[row];
            all.add(weight, variance, residual);
            byEvent
                .computeIfAbsent(EVENTS[rows.event[row]], event -> new Accumulator())
                .add(weight, variance, residual);
            clusters.merge(rows.game[row], weight * (chosen[at] - expected[at]), Double::sum);
          }
        });
    Map<LuckEvent, Score> events = new EnumMap<>(LuckEvent.class);
    byEvent.forEach((event, accumulator) -> events.put(event, accumulator.score()));
    double sum = 0;
    double squares = 0;
    for (double cluster : clusters.values()) {
      sum += cluster;
      squares += cluster * cluster;
    }
    double z = squares > 0 ? sum / Math.sqrt(squares) : Double.NaN;
    return new Holdout(all.score(), Collections.unmodifiableMap(events), z);
  }

  /** 重みつきの平方の和。 */
  private static final class Accumulator {
    private long samples;
    private double weight;
    private double variance;
    private double residual;

    void add(double sampleWeight, double sampleVariance, double sampleResidual) {
      samples++;
      weight += sampleWeight;
      variance += sampleWeight * sampleVariance;
      residual += sampleWeight * sampleResidual;
    }

    Score score() {
      if (samples == 0) {
        return new Score(0, Double.NaN, Double.NaN, Double.NaN);
      }
      return new Score(samples, variance / weight, residual / weight, 1 - residual / variance);
    }
  }

  /** 標本を半荘ごとに当てはめ [0] と保留 [1] に分ける（標本の順のまま）。 */
  private static int[][] split(LuckSamples samples, double holdout) {
    int[] train = new int[samples.size()];
    int[] held = new int[samples.size()];
    int trainCount = 0;
    int heldCount = 0;
    for (int sample = 0; sample < samples.size(); sample++) {
      if (LuckSamples.heldOut(samples.game(sample), holdout)) {
        held[heldCount++] = sample;
      } else {
        train[trainCount++] = sample;
      }
    }
    return new int[][] {Arrays.copyOf(train, trainCount), Arrays.copyOf(held, heldCount)};
  }

  /** 周回ごとに並べ替えた当てはめの標本の並び（長さは ⌈周回数 × 標本の数⌉）。 */
  private static int[] order(int[] train, double epochs, long seed) {
    int total = Math.toIntExact((long) Math.ceil(epochs * train.length));
    int[] order = new int[total];
    SplittableRandom random = new SplittableRandom(seed);
    int[] pass = train.clone();
    for (int at = 0; at < total; at += pass.length) {
      for (int index = pass.length - 1; index > 0; index--) {
        int other = random.nextInt(index + 1);
        int swap = pass[index];
        pass[index] = pass[other];
        pass[other] = swap;
      }
      System.arraycopy(pass, 0, order, at, Math.min(pass.length, total - at));
    }
    return order;
  }

  /** 並びの塊を先読みしながら順に処理する。 */
  private static void forEach(
      LuckFeed feed, int[] order, int batchSize, Consumer<LuckRows> work) {
    ArrayDeque<CompletableFuture<LuckRows>> pending = new ArrayDeque<>(PREFETCH);
    int next = 0;
    while (next < order.length || !pending.isEmpty()) {
      while (pending.size() < PREFETCH && next < order.length) {
        int count = Math.min(batchSize, order.length - next);
        pending.add(feed.prepare(order, next, count));
        next += count;
      }
      LuckRows rows = await(pending.remove());
      try {
        work.accept(rows);
      } finally {
        feed.release(rows);
      }
    }
  }

  /** 先読みの完了を待つ。符号化のスレッドの例外は包みを外して投げる。 */
  private static <T> T await(CompletableFuture<T> future) {
    try {
      return future.join();
    } catch (CompletionException failure) {
      if (failure.getCause() instanceof RuntimeException cause) {
        throw cause;
      }
      if (failure.getCause() instanceof Error cause) {
        throw cause;
      }
      throw failure;
    }
  }

  private static double square(double value) {
    return value * value;
  }
}
