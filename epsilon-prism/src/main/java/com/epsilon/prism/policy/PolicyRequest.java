package com.epsilon.prism.policy;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.prism.collect.Behaviour;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.observe.ObservationArena;
import com.epsilon.prism.observe.ObservationWriter;
import java.util.List;

/**
 * 1判断の方策の問い合わせ。観測と合法手はエンジンから借用するので、結果が返るまで持ち主の対局を進めない。
 *
 * <p>評価器は {@link #sigma} を（網の方策なら {@link #scores} も）書き、{@link #retain} なら {@link #prior} も書く（網の方策は s の最大が同点の
 * ときも {@link #prior} を書く）。書き込みは完了通知の前に終わり、通知を受けたスレッドから読む。
 *
 * <p>記録 O を読む方策（{@link PolicyService#readsRecord}。網の方策）に出す問い合わせは、出す側が自分のスレッドで O を 1 回だけ捕捉して運ぶ
 * （{@link #capture}）。その評価器は O だけを読み、公開の観測（その先の局面）には触れない。
 */
public final class PolicyRequest {
  private final Object owner;
  private final long game;
  private final int point;
  private final int seat;
  private final PublicObservation observation;
  private final EngineDecisionKind decisionKind;
  private final List<Action> legal;
  private final Kind kind;
  private final DecisionClass decisionClass;
  private final boolean retain;
  private final Capacity capacity;

  /** 捕捉した O の置き場（捕捉の前は null）。置き場は捕捉した側の持ち物。 */
  private ObservationArena record;

  /** 置き場の中の O の先頭。 */
  private int recordBase;

  /**
   * 打つ方策（網の方策は貪欲の one-hot。{@link Greedy#argmax}: s の最大、同点は p0 の大きい手。{@link
   * Behaviour} が床を混ぜる）。
   */
  public float[] sigma;

  /** 合法手ごとの s(o, a)（FP32、効用の単位。{@link #legal()} と同じ並び）。inline の方策は書かない（null）。 */
  public float[] scores;

  /**
   * 合法手ごとの規則事前 p0（{@link Prior}。和 1）。網の方策は retain の問い合わせと、s の最大が同点の問い合わせ（貪欲の同点を割る）に書き、ほかは
   * null。inline の方策は書かない。
   */
  public float[] prior;

  public PolicyRequest(
      Object owner,
      long game,
      int point,
      int seat,
      PublicObservation observation,
      EngineDecisionKind decisionKind,
      List<Action> legal,
      Kind kind,
      DecisionClass decisionClass,
      boolean retain) {
    this.owner = owner;
    this.game = game;
    this.point = point;
    this.seat = seat;
    this.observation = observation;
    this.decisionKind = decisionKind;
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

  /** エンジンの判断の種類（live は判断点の kind()、牌譜は {@link EngineDecisionKind#of}。O の頭に書く）。 */
  public EngineDecisionKind decisionKind() {
    return decisionKind;
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

  /** 行を作る判断か（ラベル行か根の記録を書く判断。方策は p0 も書く）。 */
  public boolean retain() {
    return retain;
  }

  public Capacity capacity() {
    return capacity;
  }

  /**
   * 観測・判断の種類・合法手から記録 O を置き場に書き、この問い合わせに載せる（{@link ObservationWriter#capture}）。出す側のスレッドで、評価器に
   * 渡す前に 1 回だけ呼ぶ。置き場は呼ぶ側の持ち物で、答えを書き終えるまで使い直さない（O の位置は {@code int} で持つので、その間に置き場が
   * 伸びてもよい）。
   *
   * @param writer 呼ぶスレッドの書き手
   * @param into 書く先
   * @throws IllegalStateException 既に捕捉したとき
   */
  public void capture(ObservationWriter writer, ObservationArena into) {
    if (record != null) {
      throw new IllegalStateException("the request already carries its observation record");
    }
    recordBase = writer.capture(observation, decisionKind, legal, into);
    record = into;
  }

  /**
   * 捕捉した O を置いた配列（置き場が伸びると替わるので、読むたびに引く）。
   *
   * @throws IllegalStateException 捕捉していないとき
   */
  public long[] recordWords() {
    return captured().words();
  }

  /**
   * 捕捉した O の先頭（{@link #recordWords} の中の位置）。
   *
   * @throws IllegalStateException 捕捉していないとき
   */
  public int recordBase() {
    captured();
    return recordBase;
  }

  /**
   * 同じ問い合わせ（持ち主・位置・観測・種類・合法手・類・捕捉した O）の、答えの欄が空の写し。O は写さず同じ置き場を指す。
   *
   * @param retain 写しの retain
   * @return 写し
   */
  public PolicyRequest copy(boolean retain) {
    var copy =
        new PolicyRequest(
            owner, game, point, seat, observation, decisionKind, legal, kind, decisionClass, retain);
    copy.record = record;
    copy.recordBase = recordBase;
    return copy;
  }

  private ObservationArena captured() {
    if (record == null) {
      throw new IllegalStateException("the request carries no observation record");
    }
    return record;
  }
}
