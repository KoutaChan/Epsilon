package com.epsilon.grp;

import com.epsilon.mahjong.util.SeedMixer;
import java.util.Arrays;

/**
 * Φ のメモ。卓のキー（GrpKey）と履歴のバージョン（RoundHistory が持つ値。履歴を読まないモデルでは 0）の組をキーにし、組から
 * スロットを 1 つ決めるダイレクトマップ方式で持つ。組が完全に一致したときだけ値を使い、書くときはスロットの値を上書きする。
 * Φ はこの組だけで決まるので、当たっても外れても値はビット単位で同じになる。同時に使えるのは 1 スレッドだけ。
 */
public final class PhiMemo {

  private final int seats;

  /** スロットの番号は mix64(キー ^ mix64(バージョン)) の下位ビットで決める。 */
  private final int mask;

  /** スロットごとのキー。空のスロットは GrpKey.NONE。 */
  private final long[] keys;

  private final long[] stamps;

  /** スロットごとの Φ。並びは [スロット × S ＋ 絶対席]。 */
  private final double[] values;

  private long hits;
  private long misses;
  private long bypasses;

  /** スロットの数が 2 の冪でなければ IllegalArgumentException を投げる。 */
  PhiMemo(int slots, int seats) {
    if (slots < 1 || Integer.bitCount(slots) != 1) {
      throw new IllegalArgumentException("Memo slots must be a power of two: " + slots);
    }
    this.seats = seats;
    mask = slots - 1;
    keys = new long[slots];
    Arrays.fill(keys, GrpKey.NONE);
    stamps = new long[slots];
    values = new double[slots * seats];
  }

  /**
   * メモを引き、当たれば Φ をコピーする。
   *
   * @param stamp 履歴のバージョン
   * @param key 卓のキー。空のスロットと区別できないので、GrpKey.NONE は渡さない
   * @param out Φ のコピー先
   * @return 当たれば true
   */
  boolean find(long stamp, long key, double[] out) {
    int slot = slot(stamp, key);
    if (keys[slot] != key || stamps[slot] != stamp) {
      misses++;
      return false;
    }
    hits++;
    System.arraycopy(values, slot * seats, out, 0, seats);
    return true;
  }

  void store(long stamp, long key, double[] phi) {
    int slot = slot(stamp, key);
    keys[slot] = key;
    stamps[slot] = stamp;
    System.arraycopy(phi, 0, values, slot * seats, seats);
  }

  /** キーに収まらずメモを使わなかった卓を 1 つ数える。 */
  void bypass() {
    bypasses++;
  }

  public int slots() {
    return keys.length;
  }

  public long hits() {
    return hits;
  }

  public long misses() {
    return misses;
  }

  /** キーに収まらずメモを使わなかった卓の数を返す。 */
  public long bypasses() {
    return bypasses;
  }

  private int slot(long stamp, long key) {
    return (int) SeedMixer.mix64(key ^ SeedMixer.mix64(stamp)) & mask;
  }
}
