package com.epsilon.prism.collect;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.EpsilonUtilityTargets;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpInference;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * 元の GRP（grp-epsilon-v3）を固定で使う Φ。公開系列の内容でメモ化し、同じ pivot の候補が同じ点数で局を終えた場合は1回だけ評価する。
 *
 * <p>評価は専用のスレッド（prism-grp）でまとめて行い、完了はドライバの実行キューへ戻す。メモと待ち合わせの表はドライバのスレッドだけが触る。
 *
 * <p>決定性: Φ の値は公開系列だけで決まる。バッチの組み方は実行の順（キューに入った順と poll で集まった数）で変わるが、CPU の GRU の出力は
 * 順伝播の行の数で最後のビットが変わる（1 行・2〜3 行・4 行以上で別の値。2〜3 行では隣の行にも依る）。そこで順伝播は常に
 * {@link EpsilonGrpInference#maxBatch} 行で行い、足りない行はバッチの最後の系列の写しで埋める。行の数が一つに決まれば、値は行の位置・隣の系列・
 * 詰め物の長さに依らない。メモの追い出しの後に評価し直しても同じ値になるので、同じ設定で 2 回集めた Y は最後のビットまで一致する。
 */
public final class FixedGrpPhi implements HandValue {
  private static final Object STOP = new Object();
  private final EpsilonGrpInference inference;
  private final EpsilonUtilityProfile profile;
  private final Consumer<Runnable> driver;
  private final int maxBatch;
  private final int maxCaches;
  private final double[] temperatures;
  private final PhiCalibration calibration;
  private final HashMap<Key, Entry> caches = new HashMap<>();
  private final LinkedBlockingQueue<Object> queue = new LinkedBlockingQueue<>();
  private final Thread worker;
  private volatile Throwable failure;
  private int pending;
  private long requests, hits, evaluated, batches;
  private volatile long busyNanos;

  /**
   * @param driver ドライバのスレッドで実行するための投入口
   */
  public FixedGrpPhi(
      EpsilonGrpInference inference,
      EpsilonUtilityProfile profile,
      Consumer<Runnable> driver,
      int maxBatch,
      int maxCaches) {
    this(inference, profile, driver, maxBatch, maxCaches, new double[0], null);
  }

  /**
   * 温度で較正した Φ（{@link #calibrated}）。
   *
   * @param temperatures 次局の局番号（東1 = 0）ごとの温度（最後の値はそれ以降の局にも使う）。空なら較正しない
   * @param calibration 効用の空間の較正の表（{@link PhiCalibration}。null なら使わない。温度で較正した後の Φ に掛ける）
   */
  public FixedGrpPhi(
      EpsilonGrpInference inference,
      EpsilonUtilityProfile profile,
      Consumer<Runnable> driver,
      int maxBatch,
      int maxCaches,
      double[] temperatures,
      PhiCalibration calibration) {
    this.calibration = calibration;
    for (double temperature : temperatures)
      if (!(temperature > 0)) throw new IllegalArgumentException("Φ temperatures must be positive");
    this.temperatures = temperatures.clone();
    this.inference = Objects.requireNonNull(inference);
    if (!profile.rankBased()) throw new IllegalArgumentException("Φ needs a rank utility");
    this.profile = profile;
    this.driver = Objects.requireNonNull(driver);
    this.maxBatch = maxBatch;
    this.maxCaches = maxCaches;
    worker = new Thread(this::run, "prism-grp");
    worker.setDaemon(true);
    worker.start();
  }

  @Override
  public void request(float[] prefix, int[] scores, Consumer<double[]> completion) {
    Throwable problem = failure;
    if (problem != null) throw new IllegalStateException("GRP evaluation failed", problem);
    requests++;
    Key key = new Key(prefix);
    Entry entry = caches.get(key);
    if (entry != null) {
      hits++;
      if (entry.values != null) completion.accept(entry.values);
      else entry.waiters.add(completion);
      return;
    }
    if (caches.size() >= maxCaches) caches.values().removeIf(old -> old.values != null);
    entry = new Entry(prefix);
    entry.waiters.add(completion);
    caches.put(key, entry);
    pending++;
    queue.add(entry);
  }

  @Override
  public int pending() {
    return pending;
  }

  @Override
  public Stats stats() {
    return new Stats(requests, hits, evaluated, batches, busyNanos);
  }

  private void run() {
    List<Entry> batch = new ArrayList<>(maxBatch);
    List<float[]> prefixes = new ArrayList<>(maxBatch);
    try {
      while (true) {
        Object first = queue.take();
        if (first == STOP) return;
        batch.add((Entry) first);
        Object next;
        while (batch.size() < maxBatch && (next = queue.poll(0, TimeUnit.NANOSECONDS)) != null) {
          if (next == STOP) {
            queue.add(STOP);
            break;
          }
          batch.add((Entry) next);
        }
        long started = System.nanoTime();
        for (Entry entry : batch) prefixes.add(entry.prefix);
        // 不変条件: 順伝播の行の数は常に inference.maxBatch()（詰め物の結果は捨てる）
        while (prefixes.size() % inference.maxBatch() != 0) prefixes.add(batch.getLast().prefix);
        List<float[]> marginals = inference.predictMarginalProbabilities(prefixes);
        if (marginals.size() != prefixes.size())
          throw new IllegalStateException("GRP returned a different batch size");
        for (int index = 0; index < batch.size(); index++) {
          Entry entry = batch.get(index);
          double[] raw = utilities(calibrated(marginals.get(index), entry.prefix, temperatures));
          double[] values = calibration == null ? raw : calibration.apply(raw, entry.prefix);
          driver.accept(() -> complete(entry, values));
        }
        busyNanos += System.nanoTime() - started;
        batches++;
        batch.clear();
        prefixes.clear();
      }
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
    } catch (Throwable problem) {
      failure = problem;
      driver.accept(
          () -> {
            throw new IllegalStateException("GRP evaluation failed", problem);
          });
    }
  }

  private void complete(Entry entry, double[] values) {
    entry.values = values;
    pending--;
    evaluated++;
    for (Consumer<double[]> waiter : entry.waiters) waiter.accept(values);
    entry.waiters.clear();
  }

  /**
   * GRP の順位分布の温度による較正: 次局（公開系列の最後の局境界）の局番号の温度 T で、席ごとに p' ∝ p^(1/T)（和 1）にする。T > 1 は軟らかくする
   * （GRP は今の順位が続くことに自信過剰になる）。温度が空か T = 1 なら元の分布をそのまま返す。
   *
   * @param probabilities 席 i・順位 r が [4i + r] の順位分布
   * @param prefix GRP の公開系列（{@link EpsilonGrpFeature#FEATURE_SIZE} ずつの局境界）
   */
  static float[] calibrated(float[] probabilities, float[] prefix, double[] temperatures) {
    if (temperatures.length == 0) return probabilities;
    int base = prefix.length - EpsilonGrpFeature.FEATURE_SIZE;
    int kyoku = Math.round(prefix[base + EpsilonGrpFeature.KYOKU_INDEX] * EpsilonGrpFeature.KYOKU_SCALE);
    double temperature = temperatures[Math.max(0, Math.min(kyoku, temperatures.length - 1))];
    if (temperature == 1) return probabilities;
    float[] result = new float[probabilities.length];
    for (int seat = 0; seat < 4; seat++) {
      double total = 0;
      double[] weights = new double[4];
      for (int rank = 0; rank < 4; rank++) {
        weights[rank] = Math.pow(Math.max(probabilities[seat * 4 + rank], 1e-9), 1 / temperature);
        total += weights[rank];
      }
      for (int rank = 0; rank < 4; rank++) result[seat * 4 + rank] = (float) (weights[rank] / total);
    }
    return result;
  }

  private double[] utilities(float[] probabilities) {
    double[] utilities = new double[4];
    for (int seat = 0; seat < 4; seat++)
      utilities[seat] =
          EpsilonUtilityTargets.expectedRankUtilityTrustedDouble(
              profile, probabilities, seat * 4);
    return utilities;
  }

  @Override
  public void close() {
    queue.add(STOP);
    try {
      worker.join();
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  private static final class Entry {
    final float[] prefix;
    final List<Consumer<double[]>> waiters = new ArrayList<>(2);
    double[] values;

    Entry(float[] prefix) {
      this.prefix = prefix;
    }
  }

  private record Key(float[] prefix, int hash) {
    Key(float[] prefix) {
      this(prefix, Arrays.hashCode(prefix));
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Key key && hash == key.hash && Arrays.equals(prefix, key.prefix);
    }

    @Override
    public int hashCode() {
      return hash;
    }
  }
}
