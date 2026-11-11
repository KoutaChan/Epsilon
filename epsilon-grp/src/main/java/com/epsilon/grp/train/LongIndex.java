package com.epsilon.grp.train;

import com.epsilon.mahjong.util.SeedMixer;
import java.util.Arrays;

/**
 * long のキーから番号（0 からキーの数 − 1 まで）を引く表。オープンアドレス法と線形探索を使い、スロットの使用率は半分以下に保つ。
 * 作った後は読むだけなので、どのスレッドからも引ける。引くときはメモリを確保しない。
 */
public final class LongIndex {

  /** キーが無いことを表す番号。 */
  public static final int ABSENT = -1;

  private final long[] keys;

  /** スロットごとの番号。空のスロットは ABSENT。 */
  private final int[] values;

  private final int mask;
  private final int size;

  private LongIndex(long[] keys, int[] values, int size) {
    this.keys = keys;
    this.values = values;
    mask = keys.length - 1;
    this.size = size;
  }

  /**
   * キーの列から作る。キー {@code keys[i]} の番号は i。
   *
   * @param keys 番号の順に並べたキー（読むだけ）
   * @throws IllegalArgumentException 同じキーが 2 度あるとき
   */
  public static LongIndex of(long[] keys) {
    int slots = Integer.highestOneBit(Math.max(1, keys.length) * 2 - 1) << 1;
    long[] table = new long[slots];
    int[] values = new int[slots];
    Arrays.fill(values, ABSENT);
    int mask = slots - 1;
    for (int index = 0; index < keys.length; index++) {
      long key = keys[index];
      int slot = (int) SeedMixer.mix64(key) & mask;
      while (values[slot] != ABSENT) {
        if (table[slot] == key) {
          throw new IllegalArgumentException("Duplicate key " + Long.toHexString(key));
        }
        slot = slot + 1 & mask;
      }
      table[slot] = key;
      values[slot] = index;
    }
    return new LongIndex(table, values, keys.length);
  }

  /** キーの番号を返す。キーが無ければ ABSENT。 */
  public int get(long key) {
    int slot = (int) SeedMixer.mix64(key) & mask;
    while (true) {
      int value = values[slot];
      if (value == ABSENT || keys[slot] == key) {
        return value;
      }
      slot = slot + 1 & mask;
    }
  }

  /** キーの数。 */
  public int size() {
    return size;
  }

  /** スロットの数。 */
  public int slots() {
    return keys.length;
  }
}
