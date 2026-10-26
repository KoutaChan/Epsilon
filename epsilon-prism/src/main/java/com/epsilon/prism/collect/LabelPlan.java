package com.epsilon.prism.collect;

import com.epsilon.core.Action;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Twins;
import java.util.List;

/**
 * 1 判断の収集の予定。{@link GameRunner#prepare} が推論の前に決める（payload を作るか = {@link #retain}）。決め方は硬貨と類だけで、推論の結果
 * にも結果の値にも依らない（行の採否が o と独立な乱数だけでその場で決まる、論文 命題 3 (iv)）。硬貨の座標の判断 ID は他家の伏せた手牌で枝ごとに
 * ずれるので、硬貨は観測だけの関数ではないが、鍵は山と独立なハッシュなので H を固定しても一様で、採否の確率は H に依らない（{@link
 * com.epsilon.prism.core.CrnSeeds}）。
 *
 * <ul>
 *   <li>MAIN: label = {@link CollectContext#label}（率 ρ_c）、rollIn = {@link CollectContext#rollIn}（率 β_c）。どちらも合法手が双子だけでない
 *       判断（{@link Twins#branchable}）に限る。
 *   <li>COPY: 逸脱した席の判断で、その写しのラベル行がまだ maxCopyLabels 未満なら label = {@link CollectContext#copyLabel}（率 ρ_br）。rollIn は
 *       常に false（入れ子のロールインはしない）。ほかの席の判断は何もしない。
 *   <li>FORK: 何もしない（{@link #NONE}）。
 * </ul>
 *
 * @param decisionClass 判断の類
 * @param threats 判断の席から見た公開の脅威の数（候補集合の体制の代表に使う）
 * @param label ラベル行にする
 * @param rollIn 中立ロールインを起こす（主経路だけ）
 */
record LabelPlan(DecisionClass decisionClass, int threats, boolean label, boolean rollIn) {
  /** 何もしない予定（分岐・評価・強制手）。類は使わない。 */
  static final LabelPlan NONE = new LabelPlan(DecisionClass.DISCARD_MID, 0, false, false);

  /** payload を作るか（ラベル行か根の記録を書く判断）。 */
  boolean retain() {
    return label || rollIn;
  }

  /**
   * 予定を決める。
   *
   * @param role 進行の役割
   * @param coinKey 進行の硬貨の鍵（分岐では使わない）
   * @param seat 判断の席
   * @param deviator 写しの逸脱した席（主経路と分岐は -1）
   * @param copyLabels この写しでこれまでに予約したラベル行の数（主経路と分岐は 0）
   */
  static LabelPlan decide(
      CollectContext context,
      Role role,
      long coinKey,
      long decisionId,
      int seat,
      int deviator,
      int copyLabels,
      DecisionClass decisionClass,
      int threats,
      List<Action> legal) {
    if (role == Role.FORK || !Twins.branchable(legal)) return NONE;
    boolean label, rollIn;
    if (role == Role.MAIN) {
      label = context.label(coinKey, decisionId, decisionClass);
      rollIn = context.rollIn(coinKey, decisionId, decisionClass);
    } else {
      // 写し: 逸脱した席の判断だけを、写しごとの上限まで率 ρ_br で行にする（入れ子のロールインはしない）
      label =
          seat == deviator
              && copyLabels < context.sampling.maxCopyLabels()
              && context.copyLabel(coinKey, decisionId);
      rollIn = false;
    }
    return label || rollIn ? new LabelPlan(decisionClass, threats, label, rollIn) : NONE;
  }
}
