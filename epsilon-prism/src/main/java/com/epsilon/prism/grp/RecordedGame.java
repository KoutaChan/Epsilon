package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.core.ScoreRanking;
import com.epsilon.prism.collect.HandValue;
import java.util.List;

/**
 * 主経路の 1 半荘の記録。局末 k（0 から）のうち半荘が続くものの次局の整数を順に持ち、最後の局末（半荘の終わり）は最終の点数だけ。Φ の入力
 * （次局 root の公開系列）は {@link #prefix} で組み直し、収集の RootState の公開系列とビットで同じ。
 *
 * @param gameId 半荘 ID（run 全体で一意）
 * @param starts 半荘が続く局末ごとの次局の整数（局末の順。書き換えない）
 * @param finalScores 最終の点数（書き換えない）
 */
public record RecordedGame(long gameId, List<HandStart> starts, int[] finalScores) {

  public RecordedGame {
    starts = List.copyOf(starts);
    if (finalScores.length != 4) throw new IllegalArgumentException("Four final scores required");
  }

  /** 半荘が続く局末の数（Φ を GRP で評価する局末。最後の局末の Φ は最終順位の効用そのもの）。 */
  public int continuingHandEnds() {
    return starts.size();
  }

  /** 半荘が続く局末 k の次局 root の公開系列（始めの局頭 + 次局 0..k の特徴）。 */
  public float[] prefix(int k) {
    float[] first = HandStart.INITIAL.feature();
    int width = first.length;
    float[] prefix = new float[width * (k + 2)];
    System.arraycopy(first, 0, prefix, 0, width);
    for (int step = 0; step <= k; step++)
      System.arraycopy(starts.get(step).feature(), 0, prefix, width * (step + 1), width);
    return prefix;
  }

  /** 半荘が続く局末 k の Φ の入力。 */
  public HandEndInput input(int k) {
    return new HandEndInput(prefix(k), starts.get(k));
  }

  /** 最終順位（0 = 1 位。同点は席の順）。 */
  public int[] finalRanks() {
    return ScoreRanking.byScoreThenSeat(finalScores);
  }

  /** 4 席の最終の効用。 */
  public double[] finalUtilities(EpsilonUtilityProfile profile) {
    return HandValue.terminal(finalScores, profile);
  }
}
