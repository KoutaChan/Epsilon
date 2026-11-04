package com.epsilon.reviewer.engine.scenario;

/** もしもの編集の限界。どのモデル系列の入力にも入る値の幅（Prism の点の欄は 100 点単位の 16 bit の符号つき）。 */
final class ScenarioLimits {
  static final int POINT_UNIT = 100;

  static final int MIN_SCORE = Short.MIN_VALUE * POINT_UNIT;
  static final int MAX_SCORE = Short.MAX_VALUE * POINT_UNIT;
  static final int MAX_HONBA = 255;
  static final int MAX_KYOTAKU = 100;

  private ScenarioLimits() {}
}
