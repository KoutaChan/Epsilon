package com.epsilon.agent.learn;

import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;

/**
 * 段ごとに、経過時間と回数を足し合わせる。段はモデルの系列ごとに enum で宣言する。どのスレッドからも足せる。1 回の計測の負荷は
 * {@code System.nanoTime} を 2 回呼ぶだけなので、本番でも常に使う。
 *
 * @param <S> 段の enum
 */
public final class StageTimer<S extends Enum<S>> {

  private final Class<S> stages;
  private final LongAdder[] nanos;
  private final LongAdder[] counts;

  public StageTimer(Class<S> stages) {
    this.stages = stages;
    int size = stages.getEnumConstants().length;
    nanos = new LongAdder[size];
    counts = new LongAdder[size];
    for (int index = 0; index < size; index++) {
      nanos[index] = new LongAdder();
      counts[index] = new LongAdder();
    }
  }

  public void add(S stage, long elapsedNanos) {
    add(stage, elapsedNanos, 1);
  }

  /** 何回分かの時間を、回数と一緒にまとめて足す。1 回ごとに足すと共有のカウンターへの加算が重くなる段で使う。 */
  public void add(S stage, long elapsedNanos, long count) {
    nanos[stage.ordinal()].add(elapsedNanos);
    counts[stage.ordinal()].add(count);
  }

  /** 1 回以上数えた段だけについて、段ごとの時間（秒）と回数を返す。 */
  public Map<S, Entry> snapshot() {
    Map<S, Entry> result = new EnumMap<>(stages);
    for (S stage : stages.getEnumConstants()) {
      long count = counts[stage.ordinal()].sum();
      if (count > 0) {
        result.put(stage, new Entry(nanos[stage.ordinal()].sum() / 1e9, count));
      }
    }
    return result;
  }

  public record Entry(double seconds, long count) {}
}
