package com.epsilon.prism.settings;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.prism.core.ClassTables;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.DeviationTarget;
import java.util.Map;

/**
 * 根のクラス 1 つの、写しの設定（{@link LabelSettings#copies}）。書かないキーは、labels のセクションにある同じ名前のキーの値を使う。
 *
 * @param deviationFloor 逸脱の分布に混ぜる一様分布の割合
 * @param deviation 逸脱 d を引く手の集合
 * @param copyLabelRate 写しの中で、逸脱した席の判断をラベル行にする率
 * @param laterRates 逸脱した席の判断をラベル行にする率の、クラスごとの値（書かないクラスは、この設定の率。書けるのはそのルールにあるクラス）
 * @param maxCopyLabels 写しごとのラベル行の上限
 */
public record CopySettings(
    @Key(doc = "逸脱の分布に混ぜる一様分布の割合（無ければセクションの値）", presence = Presence.ABSENT) @Range(min = 0, max = 1)
        Double deviationFloor,
    @Key(doc = "逸脱を引く手（ANY は候補の全て、OTHER は打った手のほかの候補。無ければセクションの値）", presence = Presence.ABSENT)
        DeviationTarget deviation,
    @Key(doc = "写しの中で、逸脱した席の判断をラベル行にする率（無ければセクションの値）", presence = Presence.ABSENT)
        @Range(min = 0, max = 1)
        Double copyLabelRate,
    @Key(doc = "逸脱した席の判断をラベル行にする率の、クラスごとの値（書かないクラスは、この設定の率）")
        Map<DecisionClass, @Range(min = 0, max = 1) Double> laterRates,
    @Key(doc = "写しごとのラベル行の上限（無ければセクションの値）", presence = Presence.ABSENT) @Range(min = 0)
        Integer maxCopyLabels) {

  public CopySettings {
    laterRates = ClassTables.copy(laterRates);
  }
}
