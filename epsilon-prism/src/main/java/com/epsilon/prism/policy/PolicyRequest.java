package com.epsilon.prism.policy;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.rows.ObservationPayload;
import java.util.List;

/**
 * 1判断の方策の問い合わせ。観測と合法手はエンジンから借用するので、結果が返るまで持ち主の対局を進めない。
 *
 * <p>評価器は {@link #sigma} を（網の方策なら {@link #scores} も）書き、{@link #retain} なら {@link #prior} と {@link #payload} も書く（網の方策は
 * s の最大が同点のときも {@link #prior} を書く）。書き込みは
 * 完了通知の前に終わり、通知を受けたスレッドから読む。
 */
public final class PolicyRequest {
  private final Object owner;
  private final long game;
  private final int point;
  private final int seat;
  private final PublicObservation observation;
  private final List<Action> legal;
  private final Kind kind;
  private final DecisionClass decisionClass;
  private final boolean retain;
  private final Capacity capacity;

  /**
   * 打つ方策（網の方策は貪欲の one-hot。{@link com.epsilon.prism.core.Greedy#argmax}: s の最大、同点は p0 の大きい手。{@link
   * com.epsilon.prism.collect.Behaviour} が床を混ぜる）。
   */
  public float[] sigma;

  /** 合法手ごとの s(o, a)（FP32、効用の単位。{@link #legal()} と同じ並び）。inline の方策は書かない（null）。 */
  public float[] scores;

  /**
   * 合法手ごとの規則事前 p0（{@link Prior}。和 1）。網の方策は retain の問い合わせと、s の最大が同点の問い合わせ（貪欲の同点を割る）に書き、ほかは
   * null。inline の方策は書かない。
   */
  public float[] prior;

  /** 行に保存する観測（retain のときだけ）。 */
  public ObservationPayload payload;

  public PolicyRequest(
      Object owner,
      long game,
      int point,
      int seat,
      PublicObservation observation,
      List<Action> legal,
      Kind kind,
      DecisionClass decisionClass,
      boolean retain) {
    this.owner = owner;
    this.game = game;
    this.point = point;
    this.seat = seat;
    this.observation = observation;
    this.legal = legal;
    this.kind = kind;
    this.decisionClass = decisionClass;
    this.retain = retain;
    capacity = Capacity.of(legal);
  }

  /** 問い合わせを出した対局。評価器は参照しない。 */
  public Object owner() {
    return owner;
  }

  /** 半荘 ID。 */
  public long game() {
    return game;
  }

  /** 境界内の判断の位置。 */
  public int point() {
    return point;
  }

  public int seat() {
    return seat;
  }

  public PublicObservation observation() {
    return observation;
  }

  public List<Action> legal() {
    return legal;
  }

  public Kind kind() {
    return kind;
  }

  public DecisionClass decisionClass() {
    return decisionClass;
  }

  /** payload を作るか（ラベル行か根の記録を書く判断）。 */
  public boolean retain() {
    return retain;
  }

  public Capacity capacity() {
    return capacity;
  }
}
