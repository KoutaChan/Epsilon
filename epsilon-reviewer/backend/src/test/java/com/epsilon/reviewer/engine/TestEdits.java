package com.epsilon.reviewer.engine;

import com.epsilon.reviewer.engine.scenario.DoraEdit;
import com.epsilon.reviewer.engine.scenario.HandEdit;
import com.epsilon.reviewer.engine.scenario.RiverEdit;
import com.epsilon.reviewer.engine.scenario.ScenarioEdits;
import java.util.List;
import java.util.stream.Stream;

/** もしもの試験の変更。 */
public final class TestEdits {
  /** 変更の無いもしも（評価は保存の判断と同じになる）。 */
  public static final ScenarioEdits NONE = new ScenarioEdits(null, null, null, null, null, null);

  private TestEdits() {}

  static ScenarioEdits rivers(RiverEdit... edits) {
    return new ScenarioEdits(List.of(edits), null, null, null, null, null);
  }

  static ScenarioEdits dora(DoraEdit... edits) {
    return new ScenarioEdits(null, List.of(edits), null, null, null, null);
  }

  static ScenarioEdits hand(HandEdit... edits) {
    return new ScenarioEdits(null, null, List.of(edits), null, null, null);
  }

  /** 牌の変更を first、second の順に並べた変更（点・本場・供託は変えない）。 */
  static ScenarioEdits merge(ScenarioEdits first, ScenarioEdits second) {
    return new ScenarioEdits(
        Stream.concat(first.rivers().stream(), second.rivers().stream()).toList(),
        Stream.concat(first.dora().stream(), second.dora().stream()).toList(),
        Stream.concat(first.hand().stream(), second.hand().stream()).toList(),
        null,
        null,
        null);
  }
}
