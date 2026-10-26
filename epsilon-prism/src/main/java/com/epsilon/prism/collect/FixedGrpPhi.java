package com.epsilon.prism.collect;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.EpsilonUtilityTargets;
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
 * 元の GRP（grp-epsilon-v3）を固定で使う Φ。公開系列の内容でメモ化し、同じ pivot の候補が同じ点数で局を終えた場合（約45%）は1回だけ評価する。
 *
 * <p>評価は専用のスレッド（prism-grp）でまとめて行い、完了はドライバの実行キューへ戻す。メモと待ち合わせの表はドライバのスレッドだけが触る。
 *
 * <p>決定性: バッチの組み方は実行の順（キューに入った順と poll で集まった数）で変わり、GRU の出力はバッチの形で最後のビットが変わりうる。メモは最初に
 * 評価した値を以後の要求に返すので、この Φ の収集は同じ設定で 2 回集めても Y の最後のビットまで一致するとは限らない（打ち筋・行の採否・候補集合は Φ に
 * 依らないので一致する）。行のビット単位の再現は {@link HandValue#scoreRank} の Φ でだけ保証する。
 */
public final class FixedGrpPhi implements HandValue {
  private static final Object STOP = new Object();
  private final EpsilonGrpInference inference;
  private final EpsilonUtilityProfile profile;
  private final Consumer<Runnable> driver;
  private final int maxBatch;
  private final int memoLimit;
  private final HashMap<Key, Entry> memo = new HashMap<>();
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
      int memoLimit) {
    this.inference = Objects.requireNonNull(inference);
    if (!profile.rankBased()) throw new IllegalArgumentException("Φ needs a rank utility");
    this.profile = profile;
    this.driver = Objects.requireNonNull(driver);
    this.maxBatch = maxBatch;
    this.memoLimit = memoLimit;
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
    Entry entry = memo.get(key);
    if (entry != null) {
      hits++;
      if (entry.values != null) completion.accept(entry.values);
      else entry.waiters.add(completion);
      return;
    }
    if (memo.size() >= memoLimit) memo.values().removeIf(old -> old.values != null);
    entry = new Entry(prefix);
    entry.waiters.add(completion);
    memo.put(key, entry);
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
        List<float[]> marginals = inference.predictMarginalProbabilities(prefixes);
        if (marginals.size() != batch.size())
          throw new IllegalStateException("GRP returned a different batch size");
        for (int index = 0; index < batch.size(); index++) {
          Entry entry = batch.get(index);
          double[] values = utilities(marginals.get(index));
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
