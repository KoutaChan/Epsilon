package com.epsilon.prism.grp;

import java.util.List;

/**
 * 1 コホートの主経路の対局（{@link GameLog} の中身）。
 *
 * @param header 集めたときの Φ と挙動
 * @param games 半荘（半荘 ID の昇順）
 */
public record CohortGames(GameLogHeader header, List<RecordedGame> games) {

  public CohortGames {
    games = List.copyOf(games);
  }

  /** 局末の数（半荘が続く局末と、半荘ごとに 1 つの最後の局末）。収集の報告の hands と同じ数。 */
  public long handEnds() {
    long total = 0;
    for (RecordedGame game : games) total += game.continuingHandEnds() + 1;
    return total;
  }
}
