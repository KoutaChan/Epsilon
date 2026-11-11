package com.epsilon.reviewer.engine.scenario;

import com.epsilon.agent.observe.ObservationSchema;
import com.epsilon.mahjong.rules.Variant;

/** もしもで変えられる値の範囲。O のフィールドに書ける範囲と同じ。範囲を外れた値は、O を書く前に、変更を確かめるところで断る。 */
final class ScenarioLimits {
  static final int POINT_UNIT = ObservationSchema.POINT_UNIT;

  /** 点のフィールド（符号つき 16 bit、点の単位で数える）で表せる範囲。 */
  static final int MIN_SCORE = Short.MIN_VALUE * POINT_UNIT;

  static final int MAX_SCORE = Short.MAX_VALUE * POINT_UNIT;
  static final int MAX_HONBA = ObservationSchema.MAX_HONBA;

  private ScenarioLimits() {}

  /** 供託の本数の上限（席の数 × 配給原点 / 供託 1 本の点）。O の供託のフィールドに書ける範囲と同じ。 */
  static int maxKyotaku(Variant variant) {
    return variant.seats() * variant.startPoints() / Variant.DEPOSIT;
  }
}
