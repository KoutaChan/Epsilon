package com.epsilon.prism.noise;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.prism.core.DecisionClass;
import java.util.Map;

/** 判断のクラスごとのラベル行の率を見直す方法。 */
@Choice(key = "mode", doc = "率を見直すか")
public sealed interface RateReview {

  /** 見直さず、率の表のまま使う。 */
  @Case("OFF")
  record Off() implements RateReview {}

  /**
   * 複製の行で測ったクラスごとのノイズとコストから、決まった反復の間隔で率を見直す。
   *
   * @param every 見直す間隔（反復）
   * @param stepMin 1 回の見直しで率に掛ける倍率の下限
   * @param stepMax 1 回の見直しで率に掛ける倍率の上限
   * @param minEfficiency 見直しを採るのに要る、見込みの効率の下限
   * @param weights クラスごとの設計の重み（無ければ率の表を使う）。書くなら、そのルールの全てのクラスを書く
   */
  @Case("MEASURED")
  record Measured(
      @Key(doc = "見直す間隔（反復）", value = "16") @Range(min = 1) int every,
      @Key(doc = "1 回の見直しで率に掛ける倍率の下限", value = "0.5") @Range(min = 0, minInclusive = false, max = 1) double stepMin,
      @Key(doc = "1 回の見直しで率に掛ける倍率の上限", value = "2.0") @Range(min = 1) double stepMax,
      @Key(doc = "見直しを採るのに要る、見込みの効率の下限", value = "1.05") @Range(min = 1) double minEfficiency,
      @Key(doc = "クラスごとの設計の重み（書かなければ率の表を使う）", presence = Presence.ABSENT)
          Map<DecisionClass, @Range(min = 0, minInclusive = false) Double> weights)
      implements RateReview {

    public Measured {
      if (weights != null) {
        weights = Map.copyOf(weights);
        if (weights.isEmpty()) {
          throw new InvalidSetting("weights", "weights, when written, cover the variant's classes");
        }
      }
    }
  }
}
