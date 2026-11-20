package com.epsilon.grp;

/**
 * Φ のキャッシュの大きさ。どちらもスロットの数で、2 の冪にする。1 スロットは (S ＋ 2) × 8 バイト（S は席の数）。大きさは速さとメモリに
 * だけ効き、Φ の値はビット単位で変わらない。
 *
 * @param thread スレッドごとのキャッシュのスロットの数。ロックが要らず、当たったときがいちばん速い
 * @param shared 全てのスレッドで共有するキャッシュのスロットの数。同じ局から分けた経路の局末は別々のスレッドで求めることが多く、
 *     スレッドごとのキャッシュでは当たらない分をここで拾う
 */
public record PhiCacheSizes(int thread, int shared) {

  /** 既定の大きさ。スレッドごとに 4,096 スロット、共有に 65,536 スロット。 */
  public static final PhiCacheSizes DEFAULT = new PhiCacheSizes(1 << 12, 1 << 16);

  /** どちらかのスロットの数が 2 の冪でなければ IllegalArgumentException を投げる。 */
  public PhiCacheSizes {
    if (thread < 1
        || Integer.bitCount(thread) != 1
        || shared < 1
        || Integer.bitCount(shared) != 1) {
      throw new IllegalArgumentException(
          "Phi cache slots must be powers of two: thread " + thread + ", shared " + shared);
    }
  }
}
