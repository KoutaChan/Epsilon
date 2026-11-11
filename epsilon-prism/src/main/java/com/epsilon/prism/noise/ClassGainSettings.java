package com.epsilon.prism.noise;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;

/**
 * 判断のクラスごとのゲインのセクション。クラスごとの教師のゲインと、学習の 1 ステップの大きさの比を測る。モードによっては、測った値で
 * クラスごとのアンカーの重みを決める。計測のキーはどのモードでも同じ。
 */
@Choice(key = "mode", doc = "クラスごとのゲインを測るか、測った値をどう使うか")
public sealed interface ClassGainSettings {

  /** 測らない。 */
  @Case("OFF")
  record Off() implements ClassGainSettings {}

  /** クラスごとの教師のゲインと 1 ステップの大きさの比を報告するだけ。 */
  @Case("REPORT")
  record Report(
      @Key(doc = "学習の 1 ステップの大きさの比を測る間隔（反復）", value = "4") @Range(min = 1) int kappaEvery,
      @Key(doc = "比を測るとき、学習の揺れを分母から引くか", value = "false") boolean debias,
      @Key(doc = "倍率を動かす、比と 1 の差の絶対値のしきい値", value = "0.3") @Range(min = 0, minInclusive = false) double goDelta,
      @Key(doc = "倍率を動かす z のしきい値", value = "3.0") @Range(min = 0, minInclusive = false) double goZ,
      @Key(doc = "クラスの報告を止める、比と 1 の差の絶対値のしきい値", value = "0.15") @Range(min = 0, minInclusive = false) double killDelta,
      @Key(doc = "報告するクラスが判断に占める割合の下限", value = "0.15") @Range(min = 0, minInclusive = false) double minShare)
      implements ClassGainSettings {}

  /** クラスごとの運の精算の決定係数から、クラスごとの教師のゲインを決める。 */
  @Case("CLASS_R2")
  record ClassR2(
      @Key(doc = "学習の 1 ステップの大きさの比を測る間隔（反復）", value = "4") @Range(min = 1) int kappaEvery,
      @Key(doc = "比を測るとき、学習の揺れを分母から引くか", value = "false") boolean debias,
      @Key(doc = "倍率を動かす、比と 1 の差の絶対値のしきい値", value = "0.3") @Range(min = 0, minInclusive = false) double goDelta,
      @Key(doc = "倍率を動かす z のしきい値", value = "3.0") @Range(min = 0, minInclusive = false) double goZ,
      @Key(doc = "クラスの報告を止める、比と 1 の差の絶対値のしきい値", value = "0.15") @Range(min = 0, minInclusive = false) double killDelta,
      @Key(doc = "報告するクラスが判断に占める割合の下限", value = "0.15") @Range(min = 0, minInclusive = false) double minShare)
      implements ClassGainSettings {}

  /**
   * クラスごとの決定係数から決めたゲインに倍率を掛け、上限で抑える。倍率は、学習の 1 ステップの大きさの比に合わせて動かす。
   *
   * @param stepMin 倍率を 1 回で動かす比の下限
   * @param stepMax 倍率を 1 回で動かす比の上限
   * @param multiplierMin 倍率の下限
   * @param multiplierMax 倍率の上限
   * @param cap クラスごとのゲインの上限
   */
  @Case("ADAPT")
  record Adapt(
      @Key(doc = "学習の 1 ステップの大きさの比を測る間隔（反復）", value = "4") @Range(min = 1) int kappaEvery,
      @Key(doc = "比を測るとき、学習の揺れを分母から引くか", value = "false") boolean debias,
      @Key(doc = "倍率を動かす、比と 1 の差の絶対値のしきい値", value = "0.3") @Range(min = 0, minInclusive = false) double goDelta,
      @Key(doc = "倍率を動かす z のしきい値", value = "3.0") @Range(min = 0, minInclusive = false) double goZ,
      @Key(doc = "クラスの報告を止める、比と 1 の差の絶対値のしきい値", value = "0.15") @Range(min = 0, minInclusive = false) double killDelta,
      @Key(doc = "報告するクラスが判断に占める割合の下限", value = "0.15") @Range(min = 0, minInclusive = false) double minShare,
      @Key(doc = "倍率を 1 回で動かす比の下限", value = "0.7") @Range(min = 0, minInclusive = false, max = 1) double stepMin,
      @Key(doc = "倍率を 1 回で動かす比の上限", value = "1.4") @Range(min = 1) double stepMax,
      @Key(doc = "倍率の下限", value = "0.5") @Range(min = 0, minInclusive = false, max = 1) double multiplierMin,
      @Key(doc = "倍率の上限", value = "2.0") @Range(min = 1) double multiplierMax,
      @Key(doc = "クラスごとのゲインの上限", value = "0.9") @Range(min = 0, minInclusive = false, max = 1, maxInclusive = false)
          double cap)
      implements ClassGainSettings {}
}
