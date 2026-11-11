package com.epsilon.prism.collect;

import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Openness;
import com.epsilon.prism.core.Twins;

/**
 * 1 つの判断で何を集めるかの予定。{@link CollectionPath#query} が推論の前に決め、行を作るかは {@link #retain} で分かる。決め方は乱数と
 * 判断のクラスと手の形だけで、推論の結果にも局末の値にも依らない。行の採否は、o と独立な乱数だけでその場で決まる。乱数を引く位置の判断 ID は
 * 他家の伏せた手牌で枝ごとにずれるので、この乱数は観測だけの関数ではない。ただしキーは山と独立なハッシュなので、H を固定しても乱数は一様で、
 * 採否の確率は H に依らない（{@link CrnSeeds}）。
 *
 * <p>予定を作るのは、同値な打牌をまとめても候補が 2 つ以上残る判断（{@link Twins#branchable}）だけ。
 *
 * <ul>
 *   <li>主経路（{@link #main}）: ラベル行にするかを {@link CollectContext#label}（クラスと手の形ごとの率 ρ）で、ロールインするかを
 *       {@link CollectContext#rollIn}（根の判断のクラスごとの率 β）で決める。
 *   <li>写しで逸脱した席（{@link #copy}）: その写しのラベル行の数が、根の判断のクラスで決まる上限より少なければ、ラベル行にするかを
 *       {@link CollectContext#copyLabel}（判断のクラスごとの率 ρ_br）で決める。入れ子のロールインはしないので、ロールインはいつもしない。
 *   <li>写しのほかの席と分岐: 何もしない（{@link #NONE}）。
 * </ul>
 *
 * @param decisionClass 判断のクラス
 * @param hand 判断する席の手の形
 * @param threats 判断する席から見た公開の脅威の数（候補集合に入れる体制の代表を選ぶのに使う）
 * @param label ラベル行にするか
 * @param rollIn 中立ロールインを行うか（主経路だけ）
 */
record LabelPlan(DecisionClass decisionClass, Openness hand, int threats, boolean label, boolean rollIn) {
  /** 何もしない予定（分岐・評価・強制の手）。クラスと手の形は使わない。 */
  static final LabelPlan NONE = new LabelPlan(DecisionClass.DISCARD_MID, Openness.CLOSED, 0, false, false);

  /** 行を作る判断か。ラベル行か根の記録を書く判断では、方策が p0 を書き、記録 O を取り込む。 */
  boolean retain() {
    return label || rollIn;
  }

  /**
   * 主経路の判断の予定。
   *
   * @param coinKey 主経路の乱数のキー
   */
  static LabelPlan main(
      CollectContext context, long coinKey, long decisionId, DecisionClass decisionClass, Openness hand, int threats) {
    boolean label = context.label(coinKey, decisionId, decisionClass, hand);
    boolean rollIn = context.rollIn(coinKey, decisionId, decisionClass);
    return label || rollIn ? new LabelPlan(decisionClass, hand, threats, label, rollIn) : NONE;
  }

  /**
   * 写しの中で、逸脱した席の判断の予定。
   *
   * @param coinKey 写しの乱数のキー
   * @param rule 写しの根の判断のクラスで使う、中立ロールインの決まり
   * @param copyLabels この写しでこれまでに予約したラベル行の数
   */
  static LabelPlan copy(
      CollectContext context,
      long coinKey,
      long decisionId,
      RollInRule rule,
      int copyLabels,
      DecisionClass decisionClass,
      Openness hand,
      int threats) {
    boolean label =
        copyLabels < rule.maxLabels() && context.copyLabel(coinKey, decisionId, rule.laterRate(decisionClass));
    return label ? new LabelPlan(decisionClass, hand, threats, true, false) : NONE;
  }
}
