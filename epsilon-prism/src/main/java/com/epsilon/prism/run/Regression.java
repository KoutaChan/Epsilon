package com.epsilon.prism.run;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import java.util.ArrayList;
import java.util.List;

/**
 * 王者との決闘でのリグレッションの記録（state.json）。決闘の結果はバージョンの順に反映し、同じ決闘を 2 度反映しても 1 度と同じ結果になる。
 *
 * @param streak 王者との決闘で「悪い」が続いた回数
 * @param lastDuel 最後に反映した王者との決闘のジョブの名前（まだ無ければ無い）
 * @param rollbacks 王者へ戻した決闘のバージョン（古い順）
 */
public record Regression(
    int streak,
    @Key(doc = "最後に反映した王者との決闘のジョブの名前", presence = Presence.ABSENT) String lastDuel,
    List<Integer> rollbacks) {

  /** まだ決闘を反映していない記録。 */
  public static final Regression NONE = new Regression(0, null, List.of());

  public Regression {
    if (streak < 0) {
      throw new IllegalArgumentException("The worse streak must not be negative: " + streak);
    }
    rollbacks = List.copyOf(rollbacks);
  }

  /** 決闘 {@code duel} を反映した後の記録。 */
  public Regression after(String duel, int streak) {
    return new Regression(streak, duel, rollbacks);
  }

  /** 決闘 {@code duel} で王者へ戻した後の記録。 */
  public Regression rolledBack(String duel, int streak, int version) {
    List<Integer> versions = new ArrayList<>(rollbacks);
    versions.add(version);
    return new Regression(streak, duel, versions);
  }
}
