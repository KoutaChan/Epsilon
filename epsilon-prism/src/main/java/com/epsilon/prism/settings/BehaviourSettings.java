package com.epsilon.prism.settings;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.Settings;
import com.epsilon.prism.core.TwinDiscard;
import java.util.List;
import java.util.Objects;

/**
 * 行動方策のセクション。収集の全席は、凍結した方策の貪欲な手に一様分布を混ぜた方策で打つ。同値な打牌をどちらで切るかは、収集のほかに
 * 評価の卓・決闘・牌譜でも使う。run の設定を読まずに系列の入口で開く方策（天鳳・RiichiLab・Reviewer）は、既定値で切る。
 *
 * @param uniformFloor 一様分布を混ぜる割合
 * @param exploreTemperature 主経路の判断だけは、ネットワークの値の Boltzmann 分布に一様分布を混ぜたものから引く。そのときの温度
 *     （効用の単位で、0 なら探索しない）
 * @param twinDiscard 同じ牌を手出しでもツモ切りでも切れるとき、どちらで切るか
 */
public record BehaviourSettings(
    @Key(doc = "一様分布を混ぜる割合", value = "0.005") @Range(min = 0, max = 1, maxInclusive = false)
        double uniformFloor,
    @Key(doc = "主経路の探索の温度（効用の単位で、0 なら探索しない）", value = "0.0") @Range(min = 0)
        double exploreTemperature,
    @Key(doc = "同じ牌を手出しでもツモ切りでも切れるとき、どちらで切るか", value = "TSUMOGIRI") TwinDiscard twinDiscard) {

  private static final BehaviourSettings DEFAULTS =
      Settings.section(BehaviourSettings.class, "behaviour", List.of()).value();

  public BehaviourSettings {
    Objects.requireNonNull(twinDiscard, "twinDiscard");
  }

  /** 全てのキーを、注釈に書いた既定値にした設定。 */
  public static BehaviourSettings defaults() {
    return DEFAULTS;
  }
}
