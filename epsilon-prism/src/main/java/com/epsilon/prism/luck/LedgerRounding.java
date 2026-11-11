package com.epsilon.prism.luck;

import com.epsilon.prism.core.CrnSeeds;

/**
 * 帳簿を行に書くときに使う、偏りのない確率的丸め。x を挟む fp16 の隣り合う 2 つの値を low ≤ x &lt; low + ulp とすると、確率
 * (x − low)/ulp で上へ、残りの確率で下へ丸める。よって E[結果 | x] = x となり、fp16 にしても帳簿の平均は 0 のまま保たれる。乱数は
 * unit(derive(コホートの種, LEDGER, 半荘, 判断, 手·列の数 + 列)) で、牌山とも手の抽選とも独立。半荘と判断までの部分は行の全ての手と列で
 * 同じなので、行ごとに 1 回だけ求めて乱数のキーにする（{@link #key}）。
 */
public final class LedgerRounding {

  private static final int COLUMNS = LedgerColumn.COUNT;

  private LedgerRounding() {}

  /**
   * 行の丸めに使う乱数のキー derive(コホートの種, LEDGER, 半荘, 判断) を返す。
   *
   * @param iterationSeed コホートの種
   * @param gameId 半荘の ID
   * @param decisionId 判断の ID
   */
  public static long key(long iterationSeed, long gameId, long decisionId) {
    return CrnSeeds.derive(iterationSeed, CrnSeeds.Domain.LEDGER, gameId, decisionId);
  }

  /**
   * 1 つの手の経路の帳簿を丸めて、行の帳簿に書く。
   *
   * @param key 行の丸めに使う乱数のキー（{@link #key}）
   * @param action 手の位置（行の中の何番目の手か）
   * @param terms その手の経路の帳簿（列ごと）
   * @param ledger 行の帳簿。手 a の列 c は {@code ledger[a·列の数 + c]}
   */
  public static void write(long key, int action, double[] terms, float[] ledger) {
    for (int column = 0; column < COLUMNS; column++)
      ledger[action * COLUMNS + column] = round(terms[column], coin(key, action, column));
  }

  /**
   * x を確率的に丸める。0 と NaN はそのまま返す。
   *
   * @param unit [0, 1) の一様な乱数（{@link #coin}）
   * @throws IllegalStateException |x| ≥ 65504 のとき（帳簿は有界なので起きない）
   */
  static float round(double x, double unit) {
    if (x == 0 || Double.isNaN(x)) return (float) x;
    if (!(Math.abs(x) < 65504))
      throw new IllegalStateException("Ledger value outside the fp16 range: " + x);
    int exponent = Math.max(Math.getExponent(Math.abs(x)), -14);
    double ulp = Math.scalb(1.0, exponent - 10);
    double low = Math.floor(x / ulp) * ulp;
    double fraction = (x - low) / ulp;
    return (float) (unit < fraction ? low + ulp : low);
  }

  /** 丸めに使う乱数 unit(derive(コホートの種, LEDGER, 半荘, 判断, 手·列の数 + 列)) を、行のキー（{@link #key}）から引く。 */
  static double coin(long key, int action, int column) {
    return CrnSeeds.unit(CrnSeeds.extend(key, (long) action * COLUMNS + column));
  }
}
