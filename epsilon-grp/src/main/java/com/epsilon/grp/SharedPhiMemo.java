package com.epsilon.grp;

import com.epsilon.mahjong.util.SeedMixer;
import java.util.Arrays;

/**
 * 全てのスレッドで共有する Φ のメモ。卓のキー（GrpKey）からスロットを 1 つ決めるダイレクトマップ方式で、スロットにはキー全体と Φ を
 * 書く。キーが完全に一致したときだけ値を使い、書くときはスロットを上書きする。
 *
 * <p>スロットは、いくつかに分けたロックで守る。キーと Φ は同じロックの中でまとめて書き、まとめて読むので、読むスレッドが別の卓の Φ を
 * 見ることはない。2 つのスレッドが同じ卓のメモを同時に引いて、どちらも当たらなければ、両方が Φ を求めて書く。Φ は卓だけで決まるので、どちらが書いても
 * 同じビットになる。
 *
 * <p>1 スロットは (S ＋ 1) × 8 バイト（キーと Φ）。
 */
final class SharedPhiMemo {

  /** ロックの数の上限。スロットがこれより少なければ、スロットごとに 1 つ持つ。 */
  private static final int MAX_LOCKS = 1 << 10;

  private final int seats;

  /** スロットの番号は mix64(キー) の下位ビットで決める。 */
  private final int mask;

  /** スロットごとのキー。空のスロットは GrpKey.NONE。 */
  private final long[] keys;

  /** スロットごとの Φ。並びは [スロット × S ＋ 絶対席]。 */
  private final double[] values;

  /** スロット s は locks[s & (ロックの数 − 1)] で守る。 */
  private final Object[] locks;

  /** スロットの数が 2 の冪でないときと、スロットの数と席の数の積が int に収まらないときは IllegalArgumentException を投げる。 */
  SharedPhiMemo(int slots, int seats) {
    if (slots < 1 || Integer.bitCount(slots) != 1) {
      throw new IllegalArgumentException("Memo slots must be a power of two: " + slots);
    }
    int length;
    try {
      length = Math.multiplyExact(slots, seats);
    } catch (ArithmeticException overflow) {
      throw new IllegalArgumentException(
          "A memo of " + slots + " slots for " + seats + " seats needs more than " + Integer.MAX_VALUE + " values",
          overflow);
    }
    this.seats = seats;
    mask = slots - 1;
    keys = new long[slots];
    Arrays.fill(keys, GrpKey.NONE);
    values = new double[length];
    locks = new Object[Math.min(slots, MAX_LOCKS)];
    for (int index = 0; index < locks.length; index++) {
      locks[index] = new Object();
    }
  }

  /**
   * メモを引き、当たれば Φ をコピーする。
   *
   * @param key 卓のキー。空のスロットと区別できないので、GrpKey.NONE は渡さない
   * @param out Φ のコピー先
   * @return 当たれば true
   */
  boolean find(long key, double[] out) {
    int slot = slot(key);
    synchronized (lock(slot)) {
      if (keys[slot] != key) {
        return false;
      }
      System.arraycopy(values, slot * seats, out, 0, seats);
      return true;
    }
  }

  /** 卓の Φ を書く。スロットにあった別の卓は追い出す。 */
  void store(long key, double[] phi) {
    int slot = slot(key);
    synchronized (lock(slot)) {
      keys[slot] = key;
      System.arraycopy(phi, 0, values, slot * seats, seats);
    }
  }

  private Object lock(int slot) {
    return locks[slot & locks.length - 1];
  }

  private int slot(long key) {
    return (int) SeedMixer.mix64(key) & mask;
  }
}
