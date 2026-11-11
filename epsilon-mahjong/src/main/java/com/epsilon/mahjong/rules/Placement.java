package com.epsilon.mahjong.rules;

/**
 * 点から決める順位（天鳳）。順位は 0 始まりで、点の多い順に並ぶ。同点なら起家に近い席（席の番号が小さい席）が上。
 * 席の数は配列の長さで決まり、どのメソッドもメモリを確保しない。
 */
public final class Placement {

  private Placement() {}

  /**
   * 全ての席の順位を書く。
   *
   * @param scores 席ごとの点
   * @param out 席ごとの順位を書く先
   */
  public static void ranks(int[] scores, int[] out) {
    for (int seat = 0; seat < scores.length; seat++) {
      out[seat] = rank(scores, seat);
    }
  }

  public static int rank(int[] scores, int seat) {
    int rank = 0;
    for (int other = 0; other < scores.length; other++) {
      if (scores[other] > scores[seat] || scores[other] == scores[seat] && other < seat) {
        rank++;
      }
    }
    return rank;
  }

  /** トップ（順位 0）の席。 */
  public static int top(int[] scores) {
    int top = 0;
    for (int seat = 1; seat < scores.length; seat++) {
      if (scores[seat] > scores[top]) {
        top = seat;
      }
    }
    return top;
  }

  /**
   * 終局で残った供託（1 本 1000 点）をトップに渡す。
   *
   * @param scores 席ごとの点。渡した後の点に書き換える
   * @param kyotaku 残った供託の本数
   */
  public static void awardDeposits(int[] scores, int kyotaku) {
    scores[top(scores)] += Variant.DEPOSIT * kyotaku;
  }
}
