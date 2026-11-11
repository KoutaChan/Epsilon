package com.epsilon.agent.play;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.mahjong.engine.EngineDecisionKind;

/**
 * 1 つの判断について、席の打ち手に出す問い合わせ。出す側が自分のスレッドで取り込んだ記録 O を、O のバッチとその中の番号で指す。打ち手は
 * O だけを読み、局面には触れない。バッチは出す側が持ち、答えが返るまで使い直さない。
 *
 * <p>打ち手は打つ分布 {@link #sigma} を書く。手の価値を出す打ち手は {@link #scores} も書く。系列が答えのフィールドを足すときは、このクラスを
 * 継承する。書き込みは完了を知らせる前に終わり、知らせを受けたスレッドが読む。
 */
public class Query {
  private final Object owner;
  private final long wall;
  private final int slot;
  private final int seat;
  private final EngineDecisionKind decisionKind;
  private final int[] legal;
  private final int admission;
  private final ObservationBatch records;
  private final int record;

  /** 打つ分布（合法手の順）。和が 0 の分布は一様とみなす。 */
  public float[] sigma;

  /** 合法手ごとの価値（効用の単位、合法手の順）。価値を出さない打ち手は書かない。 */
  public float[] scores;

  /**
   * @param owner 問い合わせを出した側（卓では経路。打ち手は参照しない）
   * @param wall 半荘の牌山の番号
   * @param slot 判断の時点の中で何番目の判断か
   * @param admission 受付の区分（{@link TableSeat#admission}）
   * @param records O のバッチ
   * @param record バッチの中の O の番号
   * @param observation 同じ O に付けたリーダー。席、判断の種類、合法手をここからコピーする
   */
  public Query(
      Object owner,
      long wall,
      int slot,
      int admission,
      ObservationBatch records,
      int record,
      Observation observation) {
    this.owner = owner;
    this.wall = wall;
    this.slot = slot;
    this.admission = admission;
    this.records = records;
    this.record = record;
    seat = observation.observer();
    decisionKind = observation.decisionKind();
    legal = new int[observation.legalActions()];
    for (int action = 0; action < legal.length; action++) {
      legal[action] = observation.action(action).code();
    }
  }

  /** 問い合わせを出した側。打ち手は参照しない。 */
  public Object owner() {
    return owner;
  }

  /** 半荘の牌山の番号。 */
  public long wall() {
    return wall;
  }

  /** 判断の時点の中で何番目の判断か。 */
  public int slot() {
    return slot;
  }

  public int seat() {
    return seat;
  }

  public EngineDecisionKind decisionKind() {
    return decisionKind;
  }

  /** 合法手の行動の番号（O の並び順）。呼び出し側は書き換えない。 */
  public int[] legal() {
    return legal;
  }

  public int legalCount() {
    return legal.length;
  }

  public int admission() {
    return admission;
  }

  public ObservationBatch records() {
    return records;
  }

  /** バッチの中の O の番号。 */
  public int record() {
    return record;
  }

  /** O が入った配列。バッチが伸びると別の配列に替わるので、使うたびに取り直す。 */
  public long[] recordWords() {
    return records.words();
  }

  /** {@link #recordWords} の中で O が始まる位置。 */
  public int recordBase() {
    return records.start(record);
  }

  /** リーダーをこの問い合わせの O に付けて返す。 */
  public Observation bind(Observation reader) {
    return reader.bind(records, record);
  }
}
