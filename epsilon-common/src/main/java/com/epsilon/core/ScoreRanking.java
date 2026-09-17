package com.epsilon.core;

/** 得点降順と同点時の席順から、各席の一意な順位を計算する。 */
public final class ScoreRanking {

  private ScoreRanking() {}

  /**
   * 得点降順、同点時は小さい席番号を上位として0始まり順位を返す。
   *
   * @param scores 絶対席順の得点
   * @return 絶対席ごとの0始まり順位
   */
  public static int[] byScoreThenSeat(int[] scores) {
    int[] ranks = new int[scores.length];
    byScoreThenSeat(scores, ranks);
    return ranks;
  }

  /** 呼び出し側所有バッファへ順位を書き込み、一時配列を生成しない。 */
  public static void byScoreThenSeat(int[] scores, int[] ranks) {
    if (ranks.length < scores.length) {
      throw new IllegalArgumentException("rank buffer is too small");
    }
    for (int seat = 0; seat < scores.length; seat++) {
      ranks[seat] = 0;
      for (int other = 0; other < scores.length; other++) {
        if (scores[other] > scores[seat] || scores[other] == scores[seat] && other < seat) {
          ranks[seat]++;
        }
      }
    }
  }

  /** 4人麻雀の局面の実データから指定席の順位だけを直接求める。 */
  public static int rankOf(GameState state, int seat) {
    int rank = 0;
    int score = state.getScore(seat);
    for (int other = 0; other < GameState.NUM_PLAYERS; other++) {
      int otherScore = state.getScore(other);
      if (otherScore > score || otherScore == score && other < seat) {
        rank++;
      }
    }
    return rank;
  }
}
