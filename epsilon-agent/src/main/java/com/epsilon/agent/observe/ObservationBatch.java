package com.epsilon.agent.observe;

import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.Objects;

/**
 * 1 つのルールの記録 O を、{@code long[]} に続けて並べたバッチ。推論にまとめて渡す単位になる。使う側（run や方策の区分）ごとに
 * 1 つ持ち、{@link #clear} で使い回す。
 *
 * <p>O の番号は足した順に振る。容量が足りなければ配列を伸ばすので、O は番号で覚えておき、配列は {@link #words} で取り直す。伸ばしても、
 * 既にある O の位置と中身は変わらない。配列を伸ばすとき以外は、メモリを確保しない。書くのはバッチを持つスレッドだけで、読む側は
 * 書き終えた O を、次に {@link #clear} するまで読める。
 */
public final class ObservationBatch {

  private static final int INITIAL_OBSERVATIONS = 16;

  private final Variant variant;
  private long[] words;

  /** O ごとに、配列の中で始まる添字。最後の O の次の要素には、使っている {@code long} の数を入れる。 */
  private int[] starts = new int[INITIAL_OBSERVATIONS + 1];

  private int size;

  /**
   * @param initialLongs 初めに用意する {@code long} の数
   */
  public ObservationBatch(Variant variant, int initialLongs) {
    if (initialLongs < 1) {
      throw new IllegalArgumentException("Initial capacity must be positive: " + initialLongs);
    }
    this.variant = Objects.requireNonNull(variant, "variant");
    words = new long[initialLongs];
  }

  public Variant variant() {
    return variant;
  }

  /** O の数。 */
  public int size() {
    return size;
  }

  /** 今の配列。O を足して配列が伸びると、別の配列に替わる。 */
  public long[] words() {
    return words;
  }

  /** 指定した番号の O が、配列の中で始まる添字。 */
  public int start(int index) {
    return starts[Objects.checkIndex(index, size)];
  }

  /** 指定した番号の O の長さ（{@code long} の数）。 */
  public int longs(int index) {
    return starts[Objects.checkIndex(index, size) + 1] - starts[index];
  }

  /** 渡したリーダーが今読んでいる O を末尾にコピーし、その番号を返す。ルールが違えば IllegalArgumentException を投げる。 */
  public int add(Observation observation) {
    if (observation.variant() != variant) {
      throw new IllegalArgumentException(
          "Variant mismatch: expected " + variant + " but found " + observation.variant());
    }
    return add(observation.words(), observation.base(), observation.longs());
  }

  /** 全ての O を捨て、空のバッチとして使い回す。 */
  public void clear() {
    size = 0;
  }

  /** 配列の指定した位置から始まる、指定した長さ（{@code long} の数）の O を末尾にコピーし、その番号を返す。 */
  int add(long[] source, int from, int longs) {
    if (longs < 1) {
      throw new IllegalArgumentException("An observation holds at least one long: " + longs);
    }
    int at = starts[size];
    int end = Math.addExact(at, longs);
    if (end > words.length) {
      words = Arrays.copyOf(words, Math.max(end, words.length * 2));
    }
    if (size + 1 >= starts.length) {
      starts = Arrays.copyOf(starts, starts.length * 2);
    }
    System.arraycopy(source, from, words, at, longs);
    starts[++size] = end;
    return size - 1;
  }
}
