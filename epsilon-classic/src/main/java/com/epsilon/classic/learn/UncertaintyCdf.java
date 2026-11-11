package com.epsilon.classic.learn;

import com.epsilon.classic.learn.Exploration.Kind;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * 探索の倍率を決めるための、方策の迷い u の累積分布。前の macro で数えた分布を、次の macro の間は変えずに使う。u ∈ [0, 1] を
 * {@value #BINS} 個の区間に分け、判断の種類ごと、手番の判断か応答の判断か、全体の 3 通りで数える。百分位は、その判断の種類の数が
 * {@value #MINIMUM} 以上ならその種類の分布から引く。足りなければ手番か応答かの分布から、それも足りなければ全体の分布から引く。
 */
public final class UncertaintyCdf {

  static final int BINS = 2048;

  /** 分布から百分位を引くのに要る、数えた判断の数の下限。 */
  static final int MINIMUM = 1024;

  /** 数える分布。判断の種類ごとの分布は {@link Kind} の順に先頭に並ぶ。 */
  enum Key {
    AGARI_OR_ABORT,
    CALL,
    KAN,
    RIICHI,
    DISCARD,
    TURN,
    RESPONSE,
    GLOBAL;

    static Key of(Kind kind) {
      return values()[kind.ordinal()];
    }
  }

  private static final Key[] KEYS = Key.values();

  /** [分布][区間] → その区間までの数の和。数えていない分布は null。 */
  private final long[][] cumulative;

  private UncertaintyCdf(long[][] cumulative) {
    this.cumulative = cumulative;
  }

  /** 何も数えていない分布。百分位を引けないので、探索の倍率は 1 になる。 */
  public static UncertaintyCdf empty() {
    return new UncertaintyCdf(new long[KEYS.length][]);
  }

  /**
   * 区間ごとの数から作る。
   *
   * @param counts [分布（{@link Key} の順）][区間] → 数
   * @throws IllegalArgumentException 形が違うか、負の数がある
   */
  public static UncertaintyCdf of(long[][] counts) {
    if (counts.length != KEYS.length) {
      throw new IllegalArgumentException("Counts of " + KEYS.length + " distributions are needed, not " + counts.length);
    }
    long[][] cumulative = new long[KEYS.length][];
    for (int key = 0; key < KEYS.length; key++) {
      if (counts[key].length != BINS) {
        throw new IllegalArgumentException("A distribution holds " + BINS + " bins, not " + counts[key].length);
      }
      long[] running = new long[BINS];
      long sum = 0;
      for (int bin = 0; bin < BINS; bin++) {
        if (counts[key][bin] < 0) {
          throw new IllegalArgumentException("Negative count " + counts[key][bin]);
        }
        sum = Math.addExact(sum, counts[key][bin]);
        running[bin] = sum;
      }
      cumulative[key] = running;
    }
    return new UncertaintyCdf(cumulative);
  }

  /**
   * 区間ごとの数を返す（書き出すときに使う）。
   *
   * @return [分布][区間] → 数。数えていない分布ならどの区間も 0
   */
  public long[][] counts() {
    long[][] counts = new long[KEYS.length][BINS];
    for (int key = 0; key < KEYS.length; key++) {
      if (cumulative[key] == null) {
        continue;
      }
      for (int bin = 0; bin < BINS; bin++) {
        counts[key][bin] = cumulative[key][bin] - (bin == 0 ? 0 : cumulative[key][bin - 1]);
      }
    }
    return counts;
  }

  /**
   * 迷いの百分位。同じ区間に入った判断の百分位が一様に散らばるように、区間の中の位置は乱数で決める。
   *
   * @param kind 判断の種類
   * @param response 応答の判断なら true
   * @param uncertainty 迷い（0 以上 1 以下）
   * @param tie 0 以上 1 未満の乱数
   * @return 0 以上 1 以下の百分位。どの分布も数が足りなければ NaN
   */
  public double percentile(Kind kind, boolean response, double uncertainty, double tie) {
    int bin = bin(uncertainty);
    double found = percentile(Key.of(kind), bin, tie);
    if (Double.isNaN(found)) {
      found = percentile(response ? Key.RESPONSE : Key.TURN, bin, tie);
    }
    return Double.isNaN(found) ? percentile(Key.GLOBAL, bin, tie) : found;
  }

  /** 1 つの分布の百分位。数が足りなければ NaN。 */
  private double percentile(Key key, int bin, double tie) {
    long[] running = cumulative[key.ordinal()];
    if (running == null || running[BINS - 1] < MINIMUM) {
      return Double.NaN;
    }
    long below = bin == 0 ? 0 : running[bin - 1];
    double position = (below + tie * (running[bin] - below)) / running[BINS - 1];
    return Math.max(0, Math.min(1, position));
  }

  static int bin(double uncertainty) {
    if (!(uncertainty >= 0 && uncertainty <= 1)) {
      throw new IllegalArgumentException("Uncertainty must be in [0, 1]: " + uncertainty);
    }
    return Math.min(BINS - 1, (int) (uncertainty * BINS));
  }

  /**
   * 1 回の macro の迷いを数える。どのスレッドから数えてもよい。スレッドごとに別の表に数え、{@link #freeze} で足し合わせる。
   */
  public static final class Histogram {

    private final ConcurrentLinkedQueue<long[][]> tables = new ConcurrentLinkedQueue<>();
    private final ThreadLocal<long[][]> local =
        ThreadLocal.withInitial(
            () -> {
              long[][] table = new long[KEYS.length][BINS];
              tables.add(table);
              return table;
            });

    /** 迷いを数える。 */
    public void observe(Kind kind, boolean response, double uncertainty) {
      int bin = bin(uncertainty);
      long[][] table = local.get();
      table[Key.of(kind).ordinal()][bin]++;
      table[(response ? Key.RESPONSE : Key.TURN).ordinal()][bin]++;
      table[Key.GLOBAL.ordinal()][bin]++;
    }

    /**
     * 数えた迷いの分布。数え終えた後（数えるスレッドが全て止まった後）に呼ぶ。
     *
     * @return 次の macro が引く分布
     */
    public UncertaintyCdf freeze() {
      long[][] total = new long[KEYS.length][BINS];
      for (long[][] table : tables) {
        for (int key = 0; key < KEYS.length; key++) {
          for (int bin = 0; bin < BINS; bin++) {
            total[key][bin] = Math.addExact(total[key][bin], table[key][bin]);
          }
        }
      }
      return of(total);
    }
  }
}
