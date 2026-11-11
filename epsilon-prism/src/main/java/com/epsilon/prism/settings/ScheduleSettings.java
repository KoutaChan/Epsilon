package com.epsilon.prism.settings;

import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import java.util.ArrayList;
import java.util.List;

/**
 * 反復の数と、コホートの大きさを決めるセクション。
 *
 * @param iterations 反復の数
 * @param cohorts 1 コホートの半荘の数を反復ごとに変える表。始める反復の昇順に並べ、最初は反復 1 から始める。
 *     反復 t では、始める反復が t 以下の行のうち最も後の行の数を使う
 */
public record ScheduleSettings(
    @Key(doc = "反復の数", value = "1000", change = Change.FIT) @Range(min = 1) int iterations,
    @Key(
            doc = "1 コホートの半荘の数を反復ごとに変える表（始める反復の昇順で、最初の行は反復 1 から）",
            value =
                "[{from = 1, games = 8192}, {from = 9, games = 16384}, {from = 17, games = 32768},"
                    + " {from = 25, games = 65536}]",
            change = Change.COLLECT)
        List<CohortStep> cohorts) {

  /** 1 コホートの半荘の数の上限。反復 t の半荘 ID は (t − 1)·この値 + コホートの中の番号。 */
  public static final int MAX_GAMES_PER_COHORT = 1 << 20;

  /**
   * 半荘の数の表の 1 行。
   *
   * @param from この行を使い始める反復
   * @param games 1 コホートの半荘の数
   */
  public record CohortStep(
      @Key(doc = "この行を使い始める反復", presence = Presence.REQUIRED) @Range(min = 1) int from,
      @Key(doc = "1 コホートの半荘の数", presence = Presence.REQUIRED) @Range(min = 1, max = MAX_GAMES_PER_COHORT)
          int games) {}

  public ScheduleSettings {
    cohorts = List.copyOf(cohorts);
    if (cohorts.isEmpty() || cohorts.getFirst().from() != 1) {
      throw new InvalidSetting("cohorts", "the cohort steps start at iteration 1");
    }
    for (int index = 1; index < cohorts.size(); index++) {
      if (cohorts.get(index).from() <= cohorts.get(index - 1).from()) {
        throw new InvalidSetting("cohorts", "the cohort steps' iterations must increase: " + cohorts);
      }
    }
  }

  /** 反復 t で集めるコホートの半荘の数を返す。 */
  public int games(int t) {
    int games = cohorts.getFirst().games();
    for (CohortStep step : cohorts) {
      if (step.from() <= t) {
        games = step.games();
      }
    }
    return games;
  }

  /** コホート t の半荘 ID を返す。ID は (t − 1)·{@value #MAX_GAMES_PER_COHORT} + コホートの中の番号。 */
  public List<Long> gameIds(int t) {
    int games = games(t);
    List<Long> ids = new ArrayList<>(games);
    for (int game = 0; game < games; game++) {
      ids.add((long) (t - 1) * MAX_GAMES_PER_COHORT + game);
    }
    return ids;
  }
}
