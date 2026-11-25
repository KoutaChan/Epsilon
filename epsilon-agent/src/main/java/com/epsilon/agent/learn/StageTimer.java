package com.epsilon.agent.learn;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * 段ごとに、経過時間と回数を足し合わせる。段はモデルの系列ごとに enum で宣言する。どのスレッドからも足せる。1 回の計測の負荷は
 * {@code System.nanoTime} を 2 回呼ぶだけなので、本番でも常に使う。
 *
 * <p>足す先はスレッドごとの配列で、そのスレッドだけが書く。{@link #snapshot} は全てのスレッドの配列を足し合わせる。足しているスレッドと
 * 同時に読むと、そのスレッドの最近の分が入らないことがある。足し終えたスレッドの分は、足し終えた後に読めば全て入る（終わったスレッドの
 * 分も残る）。
 *
 * @param <S> 段の enum
 */
public final class StageTimer<S extends Enum<S>> {

  /** スレッドごとの配列を、ほかのスレッドから読むときの読み方。 */
  private static final VarHandle CELLS = MethodHandles.arrayElementVarHandle(long[].class);

  private final Class<S> stages;
  private final int size;

  /** スレッドごとの配列。並びは [段 × 2]（時間）と [段 × 2 ＋ 1]（回数）。 */
  private final ThreadLocal<long[]> local = ThreadLocal.withInitial(this::register);

  /** 全てのスレッドの配列。 */
  private final ConcurrentLinkedQueue<long[]> registered = new ConcurrentLinkedQueue<>();

  public StageTimer(Class<S> stages) {
    this.stages = stages;
    size = stages.getEnumConstants().length;
  }

  public void add(S stage, long elapsedNanos) {
    add(stage, elapsedNanos, 1);
  }

  /** 何回分かの時間を、回数と一緒にまとめて足す。1 回ごとに足すと加算が重くなる段で使う。 */
  public void add(S stage, long elapsedNanos, long count) {
    long[] cells = local.get();
    int at = 2 * stage.ordinal();
    CELLS.setOpaque(cells, at, cells[at] + elapsedNanos);
    CELLS.setOpaque(cells, at + 1, cells[at + 1] + count);
  }

  /** 1 回以上数えた段だけについて、段ごとの時間（秒）と回数を返す。 */
  public Map<S, Entry> snapshot() {
    long[] sums = new long[2 * size];
    for (long[] cells : registered) {
      for (int at = 0; at < sums.length; at++) sums[at] += (long) CELLS.getOpaque(cells, at);
    }
    Map<S, Entry> result = new EnumMap<>(stages);
    for (S stage : stages.getEnumConstants()) {
      int at = 2 * stage.ordinal();
      if (sums[at + 1] > 0) {
        result.put(stage, new Entry(sums[at] / 1e9, sums[at + 1]));
      }
    }
    return result;
  }

  private long[] register() {
    long[] cells = new long[2 * size];
    registered.add(cells);
    return cells;
  }

  public record Entry(double seconds, long count) {}
}
