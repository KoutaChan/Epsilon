package com.epsilon.prism.noise;

import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.prism.core.ClassTables;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Openness;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * 判断のクラスごとのラベル行の率のセクション。使う率は、率に倍率を掛けた値（1 を超えれば 1）。副露している席の判断では、
 * 副露の表に書いたクラスだけその率を使い、ほかのクラスは率の表の率を使う。全ての合法手を測るクラスでは、候補集合を全ての合法手にする。
 *
 * @param table 判断のクラスごとのラベル行の率
 * @param open 副露している席の判断で使う、クラスごとのラベル行の率（書けるのはそのルールのクラスだけ）
 * @param multiplier 率の倍率
 * @param allActions 全ての合法手を測るクラス
 * @param review 率の見直し
 */
public record RateSettings(
    @Key(doc = "判断のクラスごとのラベル行の率", presence = Presence.REQUIRED)
        Map<DecisionClass, @Range(min = 0, max = 1) Double> table,
    @Key(doc = "副露している席の判断で使う、クラスごとのラベル行の率（書かないクラスは率の表の率を使う）")
        Map<DecisionClass, @Range(min = 0, max = 1) Double> open,
    @Key(doc = "率に掛ける倍率（掛けた値が 1 を超えれば 1 にする）", value = "1.0") @Range(min = 0, minInclusive = false)
        double multiplier,
    @Key(doc = "全ての合法手を測るクラス", value = "[DISCARD_EARLY, DISCARD_MID, DISCARD_LATE]") Set<DecisionClass> allActions,
    @Key(doc = "率の見直し", value = "OFF", change = Change.FIT) RateReview review) {

  public RateSettings {
    table = ClassTables.copy(table);
    open = ClassTables.copy(open);
    allActions = ClassTables.copy(allActions);
    Objects.requireNonNull(review, "review");
  }

  /**
   * 率の表から、そのクラスに使う率（倍率を掛け、1 を超えれば 1）を返す。
   *
   * @throws IllegalArgumentException 率の表に無いクラスのとき
   */
  public double rate(DecisionClass c) {
    Double rate = table.get(c);
    if (rate == null) {
      throw new IllegalArgumentException("No label rate for " + c);
    }
    return Math.min(1, multiplier * rate);
  }

  /**
   * 判断する席の手が副露しているかも見て、そのクラスに使う率を返す。副露していて副露の表にそのクラスがあれば、その率を使う。
   *
   * @param c 判断のクラス
   * @param hand 判断する席の手が門前か副露か
   * @return 使う率（倍率を掛け、1 を超えれば 1）
   * @throws IllegalArgumentException 副露の表に無く、率の表にもそのクラスが無いとき
   */
  public double rate(DecisionClass c, Openness hand) {
    Double override = hand == Openness.OPEN ? open.get(c) : null;
    return override == null ? rate(c) : Math.min(1, multiplier * override);
  }
}
