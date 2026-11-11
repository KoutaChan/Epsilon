package com.epsilon.prism.settings;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.prism.core.ClassTables;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.DeviationTarget;
import java.util.Map;
import java.util.Objects;

/**
 * ラベル行と中立ロールインのセクション。写しの設定（逸脱の分布に混ぜる割合・逸脱を引く手・ラベル行にする率・写しごとの上限）は、根のクラスごとに {@code copies} で
 * 替えられる。替わるのは書いたキーだけで、ほかはこのセクションのキーを使う（{@link #deviationFloor(DecisionClass)} ほか）。
 *
 * @param holdout ホールドアウトにする半荘の割合
 * @param rollin 判断のクラスごとの中立ロールインの率
 * @param maxCandidates 確約のクラスの候補集合の大きさの上限
 * @param uniformCandidate 確約のクラスの候補集合に、一様に選んだ 1 手を足す確率
 * @param deviationFloor 逸脱の分布に混ぜる一様分布の割合
 * @param deviation 逸脱 d を引く手の集合
 * @param copyLabelRate 写しの中で、逸脱した席の判断をラベル行にする率
 * @param maxCopyLabels 写しごとのラベル行の上限
 * @param copies 根のクラスごとの写しの設定（書けるのはそのルールにあるクラス）
 */
public record LabelSettings(
    @Key(doc = "ホールドアウトにする半荘の割合", value = "0.05") @Range(min = 0, max = 1, maxInclusive = false) double holdout,
    @Key(doc = "判断のクラスごとの中立ロールインの率", presence = Presence.REQUIRED)
        Map<DecisionClass, @Range(min = 0, max = 1) Double> rollin,
    @Key(doc = "確約のクラスの候補集合の大きさの上限", value = "6") @Range(min = 2, max = 32) int maxCandidates,
    @Key(doc = "確約のクラスの候補集合に、一様に選んだ 1 手を足す確率", value = "0.1") @Range(min = 0, max = 1)
        double uniformCandidate,
    @Key(doc = "逸脱の分布に混ぜる一様分布の割合", value = "0.1") @Range(min = 0, max = 1) double deviationFloor,
    @Key(doc = "逸脱を引く手（ANY は候補の全て、OTHER は打った手のほかの候補）", value = "ANY") DeviationTarget deviation,
    @Key(doc = "写しの中で、逸脱した席の判断をラベル行にする率", value = "0.3") @Range(min = 0, max = 1)
        double copyLabelRate,
    @Key(doc = "写しごとのラベル行の上限", value = "2") @Range(min = 0) int maxCopyLabels,
    @Key(doc = "根のクラスごとの写しの設定（書いたキーだけがこのセクションのキーに替わる）") Map<DecisionClass, CopySettings> copies) {

  public LabelSettings {
    rollin = ClassTables.copy(rollin);
    Objects.requireNonNull(deviation, "deviation");
    copies = ClassTables.copy(copies);
  }

  /** 根のクラス {@code root} の写しで、逸脱の分布に混ぜる一様分布の割合。 */
  public double deviationFloor(DecisionClass root) {
    CopySettings copy = copies.get(root);
    return copy != null && copy.deviationFloor() != null ? copy.deviationFloor() : deviationFloor;
  }

  /** 根のクラス {@code root} の写しで、逸脱 d を引く手の集合。 */
  public DeviationTarget deviation(DecisionClass root) {
    CopySettings copy = copies.get(root);
    return copy != null && copy.deviation() != null ? copy.deviation() : deviation;
  }

  /**
   * 根のクラス {@code root} の写しの中で、逸脱した席の判断をラベル行にする率。
   *
   * @param later 逸脱した席の判断のクラス
   */
  public double copyLabelRate(DecisionClass root, DecisionClass later) {
    CopySettings copy = copies.get(root);
    if (copy == null) return copyLabelRate;
    Double byLater = copy.laterRates().get(later);
    if (byLater != null) return byLater;
    return copy.copyLabelRate() != null ? copy.copyLabelRate() : copyLabelRate;
  }

  /** 根のクラス {@code root} の、写しごとのラベル行の上限。 */
  public int maxCopyLabels(DecisionClass root) {
    CopySettings copy = copies.get(root);
    return copy != null && copy.maxCopyLabels() != null ? copy.maxCopyLabels() : maxCopyLabels;
  }
}
