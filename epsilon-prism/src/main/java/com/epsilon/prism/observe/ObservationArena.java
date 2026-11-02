package com.epsilon.prism.observe;

import java.util.Arrays;

/**
 * 記録 O を続けて置く {@code long[]} の置き場。run ごとに 1 つ持ち、{@link #clear} で使い回す。
 *
 * <p>{@link #allocate} は 0 で埋めた場所の先頭を返す。足りなければ配列を伸ばすので、O を読むときは位置を {@code int} で持ち、配列は
 * {@link #words} で引き直す（伸ばしても前の O の位置と中身は変わらない）。書くのは持ち主のスレッドだけで、読む側は書き終えた O を
 * 次の {@link #clear} まで読む。
 */
public final class ObservationArena {

  private long[] words;
  private int size;

  /**
   * 置き場を作る。
   *
   * @param initialLongs 初めに用意する {@code long} の数
   */
  public ObservationArena(int initialLongs) {
    if (initialLongs < 1) {
      throw new IllegalArgumentException("initial capacity must be positive: " + initialLongs);
    }
    words = new long[initialLongs];
  }

  /**
   * 0 で埋めた {@code longs} 個の場所を取る。
   *
   * @param longs {@code long} の数
   * @return 場所の先頭
   */
  public int allocate(int longs) {
    if (longs < 0) {
      throw new IllegalArgumentException("negative length: " + longs);
    }
    int at = size;
    int end = Math.addExact(at, longs);
    if (end > words.length) {
      words = Arrays.copyOf(words, Math.max(end, words.length * 2));
    }
    Arrays.fill(words, at, end, 0L);
    size = end;
    return at;
  }

  /** 今の配列（{@link #allocate} で伸ばすと替わる）。 */
  public long[] words() {
    return words;
  }

  /** 使っている {@code long} の数。 */
  public int size() {
    return size;
  }

  /** 全ての O を捨てて、初めから使い回す。 */
  public void clear() {
    size = 0;
  }
}
