package com.epsilon.prism.collect;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.prism.core.DecisionClass;
import java.util.List;

/**
 * ラベル行の候補を外の測定器（例: Epsilon-Nano の貪欲）の 1 手に替える仕掛け（probe-nano）。通常の収集では使わない（{@code
 * CollectContext.probe} は null）。
 *
 * <p>使うとき、{@code GameRunner} は主経路のラベル行に予約した判断ごとに次のどれか 1 つだけを数える（{@link #counted}）。
 *
 * <ul>
 *   <li>{@link Outcome#FLOOR}: b が床 ε_b で貪欲でない手を打った（測定器に問わず、行も分岐も作らない。床の手と測定器の比べは PRISM の比べではない
 *       ため。選別は手の抽選の共通乱数だけで決まるので、残る判断は偏らない）。
 *   <li>{@link Outcome#AGREED}: 測定器の手が打った手と同じか双子（{@link com.epsilon.prism.core.Twins#same}）。行も分岐も作らない。
 *   <li>{@link Outcome#DISAGREED}: 違う。候補集合を {played, 測定器の手} の 2 手にしたラベル行を作り、測定器の手の分岐を作る（打った手は主経路の
 *       値）。
 * </ul>
 *
 * <p>ロールインと写しは使わない（{@link CollectionDriver#collect(CollectionDriver.Cohort,
 * com.epsilon.prism.policy.PolicyService, HandValue, com.epsilon.prism.rows.RowSink, CandidateProbe)} が率 0 を確かめる）。
 *
 * <p>{@link #choose} と {@link #counted} は複数の進行ワーカーから並行に呼ばれる。
 */
public interface CandidateProbe {
  /** ラベル行に予約した判断の数え方。 */
  enum Outcome {
    FLOOR,
    AGREED,
    DISAGREED
  }

  /**
   * 測定器の手。観測と合法手は呼出中だけ借用する。打った手を返すと一致として数え、行も分岐も作らない。
   *
   * @param played PRISM が打った手（合法手の位置）
   * @return 合法手の位置
   */
  int choose(PublicObservation observation, List<Action> legal, int played);

  /** 類 decisionClass の判断を 1 つ数える。 */
  void counted(DecisionClass decisionClass, Outcome outcome);
}
