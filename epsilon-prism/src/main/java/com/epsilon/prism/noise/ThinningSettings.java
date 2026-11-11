package com.epsilon.prism.noise;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import com.epsilon.prism.core.ClassTables;
import com.epsilon.prism.core.DecisionClass;
import java.util.Map;

/**
 * 間引きのセクション。打った手以外の候補は、方策の確率を温度で割り、下限と 1 の間に収めた確率で測る。測った手の教師には、その確率の
 * 逆数の重みを掛ける。
 */
@Choice(key = "enabled", doc = "打った手以外の候補を間引くか")
public sealed interface ThinningSettings {

  /** 間引かない。 */
  @Case("false")
  record Off() implements ThinningSettings {}

  /**
   * 間引く。
   *
   * @param classes 判断のクラスごとの間引き方。書いたクラスだけを間引く。間引くクラスが 1 つ以上要る
   * @param full 監査のために全ての候補を測る行の割合
   */
  @Case("true")
  record On(
      @Key(
              doc = "判断のクラスごとの間引き方（書いたクラスだけを間引く。間引くクラスが 1 つ以上要る）",
              value = "{DISCARD_EARLY = {}, DISCARD_MID = {}, DISCARD_LATE = {}}")
          Map<DecisionClass, ClassThinning> classes,
      @Key(doc = "監査のために全ての候補を測る行の割合", value = "0.005") @Range(min = 0, minInclusive = false, max = 1, maxInclusive = false)
          double full)
      implements ThinningSettings {

    public On {
      classes = ClassTables.copy(classes);
      if (classes.values().stream().noneMatch(ClassThinning::enabled)) {
        throw new InvalidSetting("classes", "thinning needs at least one class");
      }
    }
  }

  /**
   * 1 つのクラスの間引き方。
   *
   * @param enabled このクラスを間引くか
   * @param tau 測る確率の温度
   * @param qMin 測る確率の下限
   */
  record ClassThinning(
      @Key(doc = "このクラスを間引くか", value = "true") boolean enabled,
      @Key(doc = "測る確率の温度", value = "0.1") @Range(min = 0, minInclusive = false) double tau,
      @Key(doc = "測る確率の下限", value = "0.05") @Range(min = 0x1.0p-14, max = 1) double qMin) {}
}
