package com.epsilon.mahjong.util;

/** 乱数の種を混ぜて、用途ごとのソルトと番号から別の種を作る。同じ入力には、スレッドや呼び出しの順によらず同じ値を返す。 */
public final class SeedMixer {

  /** SplitMix64 が 1 ステップで状態に足す数。2^64 を黄金比で割った値に最も近い奇数。 */
  private static final long GOLDEN_GAMMA = 0x9E3779B97F4A7C15L;

  private SeedMixer() {}

  /** SplitMix64 の出力関数（Stafford の Mix13）。64 ビットの全単射で、入力が 1 ビット違うだけで出力の全体が変わる。 */
  public static long mix64(long value) {
    value ^= value >>> 30;
    value *= 0xBF58476D1CE4E5B9L;
    value ^= value >>> 27;
    value *= 0x94D049BB133111EBL;
    value ^= value >>> 31;
    return value;
  }

  /**
   * 番号ごとに別の種を返す。ソルトが 0 なら、元の種を初めの状態にした SplitMix64 が {@code index} 回目（1 回目から数える）に
   * 出す値と同じ。
   *
   * @param baseSeed 元の種
   * @param salt 用途ごとのソルト
   * @param index 番号
   * @return {@code mix64(baseSeed + salt + index × γ)}
   */
  public static long indexed(long baseSeed, long salt, long index) {
    return mix64(baseSeed + salt + index * GOLDEN_GAMMA);
  }

  /**
   * 座標の列ごとに別の種を返す。{@code r = mix64(baseSeed ^ salt)} から始め、座標 c ごとに {@code r = mix64(r ^ mix64(c))}
   * とする。そのため座標の順が違えば値も変わる。呼ぶたびに可変長引数の配列を作るので、速さが要るループの中では使わない。
   */
  public static long derive(long baseSeed, long salt, long... coordinates) {
    long result = mix64(baseSeed ^ salt);
    for (long coordinate : coordinates) {
      result = mix64(result ^ mix64(coordinate));
    }
    return result;
  }
}
