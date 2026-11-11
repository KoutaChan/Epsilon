package com.epsilon.prism.policy;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.Query;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.core.Kind;

/**
 * PRISM の方策に出す、1 判断の問い合わせ。卓の問い合わせ（{@link Query}）に、判断の種別とクラス、行を作る判断かどうかを足し、答えとして
 * ルールベースの事前分布 p0 を受け取る。
 *
 * <p>評価器は {@link #sigma} に打つ方策を書く。ネットワークの方策では貪欲の one-hot で、s が最大の手を選び、同点なら p0 の大きい手を選ぶ
 * （{@link Greedy#argmax}）。一様分布を混ぜるのは {@link Behaviour} の仕事。ネットワークの方策は {@link #scores} に合法手ごとの s（FP32、
 * 効用の単位）も書く。{@link #retain} の問い合わせでは {@link #prior} も書き、ネットワークの方策は s の最大が同点のときにも書く。
 */
public final class PolicyRequest extends Query {
  private final Kind kind;
  private final DecisionClass decisionClass;
  private final boolean retain;

  /**
   * 合法手ごとのルールベースの事前分布 p0（{@link Prior}。和は 1）。ネットワークの方策は、{@code retain} の問い合わせと、s の最大が同点の
   * 問い合わせ（貪欲の同点を p0 で決める）にだけ書く。ほかの問い合わせでは null。inline の方策は書かない。
   */
  public float[] prior;

  /**
   * @param owner 問い合わせを出した側（評価器は参照しない）
   * @param wall 半荘の牌山の番号
   * @param slot 判断の時点の中で何番目の判断か
   * @param kind 判断の種別（O だけを読む呼び出し元は null）
   * @param decisionClass 判断のクラス（O だけを読む呼び出し元は null）
   * @param retain 行を作る判断か
   * @param admission 受付の区分（{@link PolicyService#admission}）
   * @param records O のバッチ
   * @param record バッチの中の O の番号
   * @param observation 同じ O を読むリーダー。席、判断の種類、合法手をここからコピーする
   */
  public PolicyRequest(
      Object owner,
      long wall,
      int slot,
      Kind kind,
      DecisionClass decisionClass,
      boolean retain,
      int admission,
      ObservationBatch records,
      int record,
      Observation observation) {
    super(owner, wall, slot, admission, records, record, observation);
    this.kind = kind;
    this.decisionClass = decisionClass;
    this.retain = retain;
  }

  public Kind kind() {
    return kind;
  }

  public DecisionClass decisionClass() {
    return decisionClass;
  }

  /** 行を作る判断か。ラベル行か根の記録を書く判断で、方策は p0 も書く。 */
  public boolean retain() {
    return retain;
  }
}
