package com.epsilon.agent.eval;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;

/** 評価の卓の止め方。 */
@Choice(key = "kind", doc = "卓を止める決まり")
public sealed interface StopSettings {

  /** 計画の牌山を全て打つ。 */
  @Case("FIXED")
  record Fixed() implements StopSettings {}

  /**
   * 判定の時点で標準誤差が目標以下なら止める。
   *
   * @param error 目標の標準誤差（判定に使う量の単位）
   * @param alpha 両側の有意水準
   * @param checkEvery 判定の時点の間隔（牌山の数）
   */
  @Case("PRECISION")
  record Precision(
      @Key(doc = "目標の標準誤差（判定に使う量の単位）", presence = Presence.REQUIRED) @Range(min = 0) double error,
      @Key(doc = "両側の有意水準", presence = Presence.REQUIRED) @Range(min = 0, minInclusive = false, max = 0.2)
          double alpha,
      @Key(doc = "判定の時点の間隔（牌山の数）", presence = Presence.REQUIRED) @Range(min = 512) int checkEvery)
      implements StopSettings {}

  /**
   * 良い・悪い・互角のどれかが出た時点で止める。
   *
   * @param evenMargin 互角の幅（0 なら互角では止めない）
   * @param alpha 両側の有意水準
   * @param spending 有意水準を判定の時点に配る消費関数の指数（大きいほど、早い時点に配る有意水準が小さい）
   * @param checkEvery 判定の時点の間隔（牌山の数）
   * @param promote 判定が「良い」なら、王者を候補に替える
   */
  @Case("TEST")
  record Test(
      @Key(doc = "互角の幅（0 なら互角では止めない）", presence = Presence.REQUIRED) @Range(min = 0) double evenMargin,
      @Key(doc = "両側の有意水準", presence = Presence.REQUIRED) @Range(min = 0, minInclusive = false, max = 0.2)
          double alpha,
      @Key(doc = "有意水準を判定の時点に配る消費関数の指数（大きいほど、早い時点に配る有意水準が小さい）", presence = Presence.REQUIRED) @Range(min = 0.5, max = 4) double spending,
      @Key(doc = "判定の時点の間隔（牌山の数）", presence = Presence.REQUIRED) @Range(min = 512) int checkEvery,
      @Key(doc = "判定が「良い」なら、王者を候補に替える", presence = Presence.REQUIRED) boolean promote)
      implements StopSettings {}

  /** 判定の時点の間隔（牌山の数）。計画の牌山を全て打つ止め方では 0。 */
  default int checkEvery() {
    return 0;
  }
}
