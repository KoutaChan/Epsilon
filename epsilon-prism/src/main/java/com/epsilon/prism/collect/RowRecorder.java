package com.epsilon.prism.collect;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.prism.auxiliary.HandEndAnalyzer;
import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.auxiliary.OracleCapture;
import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.auxiliary.TenpaiReach;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.metrics.StageTimer;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.observe.ObservationSchema;
import com.epsilon.prism.policy.PolicyRequest;
import java.util.Arrays;
import java.util.List;

/**
 * 経路（{@link GameRunner}）ごとの行の記録（行を書く収集だけ。{@link CollectContext#rows}）。進行・乱数には触らない（状態を読むだけ）。
 *
 * <ul>
 *   <li>O: 残す判断（{@link LabelPlan#retain}）の記録 O は経路の置き場（{@link QueryRecords}）に prepare が 1 回だけ捕捉する（O を読む方策の
 *       問い合わせは符号化のためにも捕捉するので、その O を使う）。強制の判断（合法手 1 つ）は網に問わず行にもならないので捕捉しない。
 *   <li>判断の記録（record、行を作る判断だけ）: 問い合わせの O を置き場から写し（{@link ObservationSchema#longs} の数を {@link
 *       System#arraycopy}）、神託を {@link OracleCapture} で作る（{@link Decision}）。2 回目の捕捉はしない。
 *   <li>到達の追跡: 境界ごと（prepare の最初）と局末に、前に見てから河に増えた打牌を {@link TenpaiReach} に渡す。エンジンは打牌のたびに判断の
 *       境界を作るので、境界 1 つで河が増えるのは 1 枚までで、その席は打牌の後の 13 枚のまま。到達の教師は枠の判断の席のものだけなので、枠の席の
 *       打牌だけを確かめ、ほかの打牌は数だけ合わせる（枠ができる前の打牌は判断の時点の出来事の数より前なので、教師に効かない）。
 *   <li>局末（枠のある経路だけ）: {@link HandEndAnalyzer} の結果（{@link HandEndFacts}）を作る。
 * </ul>
 *
 * <p>写しと分岐は、親の到達と河の数を写して始める（{@link #RowRecorder(RowRecorder)}。分岐の根の打牌は分岐の最初の境界で数える）。主経路は局ごとに
 * {@link #reset} する。解析器・採点器は進行ワーカーのスレッドごとに 1 つ持つ。1 度に触るのは 1 スレッドだけ。計測は経路に貯め、
 * {@link #flush} で足す。記録のどこかが失敗したら、その例外で収集を止める。
 */
final class RowRecorder {

  /** 枠の判断の席のビットをまだ求めていないこと。 */
  private static final int UNKNOWN_SEATS = -1;

  private static final Step[] STEPS = Step.values();

  /** 進行ワーカーのスレッドごとの道具（解析器・採点器はスレッド安全でない）。 */
  private static final ThreadLocal<Tools> TOOLS = ThreadLocal.withInitial(Tools::new);

  private final TenpaiReach reach;

  /** 席ごとの、前に見た河の打牌の数。 */
  private final int[] discards = new int[GameState.NUM_PLAYERS];

  private final long[] nanos = new long[STEPS.length];
  private final long[] calls = new long[STEPS.length];

  /** 主経路の記録を、局の始めの状態で作る。 */
  RowRecorder() {
    reach = new TenpaiReach();
  }

  /** 写しか分岐の記録を、親の今の状態（親の判断の境界）から作る。 */
  RowRecorder(RowRecorder parent) {
    reach = new TenpaiReach(parent.reach);
    System.arraycopy(parent.discards, 0, discards, 0, discards.length);
  }

  /** 境界の始め（prepare の最初）: 前に見てから増えた打牌を到達に渡す。 */
  void boundary(GameState state, List<Slot> targets) {
    discards(state, targets);
  }

  /**
   * 行を作る判断の記録を、問い合わせが運ぶ O の写しから作る。state は commit の前（神託は 4 人の手を読む）。
   *
   * @param request 行を作る判断の問い合わせ（O を捕捉したもの）
   * @param state commit の前の局面
   * @return 記録
   */
  Decision decision(PolicyRequest request, GameState state) {
    long started = System.nanoTime();
    long[] words = request.recordWords();
    int base = request.recordBase();
    long[] observation = new long[ObservationSchema.longs(words, base)];
    System.arraycopy(words, base, observation, 0, observation.length);
    Tools tools = TOOLS.get();
    int events = tools.reader.bind(observation, 0).events();
    List<Action> legal = request.legal();
    int targets = 0;
    for (int action = 0, actions = legal.size(); action < actions; action++) {
      if (TargetSchema.hasCandidateTargets(ActionRole.of(legal.get(action).type()))) {
        targets |= 1 << action;
      }
    }
    Decision decision =
        new Decision(observation, tools.oracles.capture(state, request.seat()), events, targets);
    add(Step.ORACLE, started);
    return decision;
  }

  /**
   * 枠のある経路の局末（endHand の最初、次局を始める前）: 最後の打牌を到達に渡し、局末の事実を作る。
   *
   * @return 局末の事実
   */
  HandEndFacts handEnd(GameState state, RoundSettlement settlement, List<Slot> targets) {
    discards(state, targets);
    long started = System.nanoTime();
    HandEndFacts facts = TOOLS.get().analyzer.analyze(state, settlement, reach);
    add(Step.HAND_END, started);
    return facts;
  }

  /** 主経路が次局へ移る（到達と河の数を消す）。 */
  void reset() {
    reach.reset();
    Arrays.fill(discards, 0);
  }

  /** 計測した時間と回数を timer に足して空にする（{@link GameRunner#advance} の終わりに 1 回）。 */
  void flush(StageTimer timer) {
    for (Step step : STEPS) {
      int at = step.ordinal();
      if (calls[at] == 0) {
        continue;
      }
      timer.add(step.stage, nanos[at], calls[at]);
      nanos[at] = 0;
      calls[at] = 0;
    }
  }

  /** 前に見てから河に増えた打牌を数え、枠の席の打牌を到達に渡す。 */
  private void discards(GameState state, List<Slot> targets) {
    int seats = UNKNOWN_SEATS;
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      int size = state.river(seat).size();
      int seen = discards[seat];
      if (size == seen) {
        continue;
      }
      if (size != seen + 1) {
        throw new IllegalStateException(
            "seat " + seat + " discarded " + (size - seen) + " tiles between two boundaries");
      }
      discards[seat] = size;
      if (seats == UNKNOWN_SEATS) {
        seats = seats(targets);
      }
      if ((seats & 1 << seat) != 0) {
        long started = System.nanoTime();
        reach.discarded(state, seat);
        add(Step.REACH, started);
      }
    }
  }

  /** 枠の判断の席のビット。 */
  private static int seats(List<Slot> targets) {
    int seats = 0;
    for (int index = 0, size = targets.size(); index < size; index++) {
      int seat = targets.get(index).target().seat();
      if (seat >= 0) {
        seats |= 1 << seat;
      }
    }
    return seats;
  }

  private void add(Step step, long started) {
    nanos[step.ordinal()] += System.nanoTime() - started;
    calls[step.ordinal()]++;
  }

  /**
   * 行を作る判断の記録。判断の時点で決まり、同じ判断のラベル行と根で共有する（配列は書き換えない）。
   *
   * @param observation 記録 O（{@link ObservationSchema#longs} の数の {@code long}）
   * @param oracle 神託（{@link OracleCodec}）
   * @param events 判断の時点の出来事の数（O の頭。局末の 3 語の到達の基準）
   * @param targets 局末の 3 語を記録する手（{@link TargetSchema#hasCandidateTargets} が真の手）の位置のビット
   */
  record Decision(long[] observation, long[] oracle, int events, int targets) {

    /**
     * 手 action の経路の局末を、判断の席から見た 3 語にして outcomes に書く（記録しない手は書かない）。ドライバのスレッドで呼ぶ。
     *
     * @param codec ドライバの書き手
     * @param end 経路の局末の事実
     * @param seat 判断の席
     * @param action 手の位置
     * @param outcomes 手ごとの 3 語（長さ 合法手 × {@link OutcomeCodec#WORDS}）
     */
    void outcome(OutcomeCodec codec, HandEndFacts end, int seat, int action, long[] outcomes) {
      if ((targets >>> action & 1) != 0) {
        codec.write(end, seat, events, outcomes, action * OutcomeCodec.WORDS);
      }
    }
  }

  /** 計測する段（O の捕捉は {@link QueryRecords} が測る）。 */
  private enum Step {
    ORACLE(StageTimer.Stage.V2_ORACLE),
    REACH(StageTimer.Stage.V2_REACH),
    HAND_END(StageTimer.Stage.V2_HAND_END);

    private final StageTimer.Stage stage;

    Step(StageTimer.Stage stage) {
      this.stage = stage;
    }
  }

  /** スレッドごとの道具。 */
  private static final class Tools {
    final Observation reader = new Observation();
    final OracleCapture oracles = new OracleCapture();
    final HandEndAnalyzer analyzer = new HandEndAnalyzer();
  }
}
