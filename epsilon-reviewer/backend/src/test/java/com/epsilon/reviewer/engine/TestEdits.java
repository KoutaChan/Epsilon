package com.epsilon.reviewer.engine;

import com.epsilon.reviewer.engine.scenario.ScenarioEdits;

/** もしもの試験の変更。 */
public final class TestEdits {
  /** 変更の無いもしも（評価は保存の判断と同じになる）。 */
  public static final ScenarioEdits NONE = new ScenarioEdits(null, null, null, null, null, null);

  private TestEdits() {}
}
