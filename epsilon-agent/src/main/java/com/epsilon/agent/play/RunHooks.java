package com.epsilon.agent.play;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;

/**
 * 卓の経路ごとに 1 つ持つ、系列の処理と状態。{@link #settle} のほかは、経路を今進めている進行のスレッドだけが呼ぶ。経路は判断の
 * 時点ごとに別のスレッドへ移るが、同時に触るスレッドはいつも 1 つ。どの処理も局面を書き換えず、卓が手を引く乱数も使わない。
 *
 * @param <Q> 問い合わせの型
 */
public interface RunHooks<Q extends Query> {

  /** 判断の時点の始めに呼ぶ（問い合わせを作る前）。 */
  default void boundary(Run<Q> run, Decisions decisions, RuleWorkspace ws) {}

  /**
   * 強制でない判断の問い合わせを作る。記録はもう取り込んである。
   *
   * @param slot 判断の時点の中で何番目の判断か
   * @param view 判断する席から見た局面
   * @param admission 席の打ち手の受付の区分
   * @param records 記録のバッチ
   * @param record バッチの中の記録の番号
   * @param observation その記録に付けたリーダー（この呼び出しの間だけ有効）
   */
  Q query(
      Run<Q> run,
      Decisions decisions,
      int slot,
      SeatView view,
      int admission,
      ObservationBatch records,
      int record,
      Observation observation);

  /**
   * 引いた手を、実際に打つ手に替える。強制でない判断ごとに手を引いた直後に呼び、返した手を主経路の観測・{@link #decided}・エンジンに
   * 渡す。既定は引いた手のまま。系列が同じ値と見る手のうちどれを打つかを決めるためのもの。
   *
   * @param query 判断の問い合わせ
   * @param drawn 引いた手（合法手の位置）
   * @return 打つ手（合法手の位置）
   */
  default int play(Q query, int drawn) {
    return drawn;
  }

  /**
   * 全ての判断の手を選んだ後、エンジンを進める前に呼ぶ。分けた続きは、ここで {@link Run#fork} を使って作る。
   *
   * @param chosen 判断ごとに打つ合法手の位置（{@link #play} で替えた後。読むだけ）
   */
  default void decided(Run<Q> run, Decisions decisions, int[] chosen, RuleWorkspace ws) {}

  /**
   * 局が終わったとき、局末の局面のまま、次局を配る前に呼ぶ。半荘が終わる局でも呼ぶ。
   *
   * @param gameEnded 半荘が終わった局なら true
   */
  default void roundEnded(Run<Q> run, RoundEnd end, boolean gameEnded) {}

  /**
   * 経路を次に止まるところまで進めた後に呼ぶ。
   *
   * @param nanos 進めるのにかかった時間（ナノ秒）
   */
  default void advanced(Run<Q> run, long nanos) {}

  /** ドライバのスレッドが、進めた経路を受け取ったときに呼ぶ。進行のスレッドが経路に貯めた局末は、ここで受け渡す。 */
  default void settle(Run<Q> run) {}
}
