package com.epsilon.prism.collect;

import com.epsilon.agent.learn.StageTimer;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.auxiliary.HandEndAnalyzer;
import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.auxiliary.OracleCapture;
import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.auxiliary.TenpaiReach;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.policy.PolicyRequest;
import java.util.Arrays;
import java.util.List;

/**
 * 経路（{@link CollectionPath}）ごとの行の記録。行を書く収集（{@link CollectContext#rows}）だけで使う。局面を読むだけで、進行と乱数には
 * 触らない。
 *
 * <ul>
 *   <li>記録 O: 強制でない全ての判断の O は、卓の経路が判断の時点の始めに 1 回だけ取り込む。強制の判断（合法手が 1 つ）はネットワークに
 *       問わず、行にもならないので取り込まない。
 *   <li>判断の記録（行を作る判断だけ）: 推論の要求の O をバッチから写し、Oracle を {@link OracleCapture} で作る（{@link Decision}）。
 *       O を 2 回取り込むことはしない。
 *   <li>聴牌への到達: 判断の時点の始めと局末に、前に見てから河に増えた打牌を {@link TenpaiReach} に渡す。エンジンは打牌のたびに判断の時点を
 *       作るので、時点 1 つの間に河が増えるのは 1 枚までで、その席は打牌の後の 13 枚のままになる。到達の教師を使うのは受け取り先を持つ判断の
 *       席だけなので、その席の打牌だけを調べ、ほかの席の打牌は数だけを合わせる。受け取り先ができる前の打牌は、判断の時点のイベントの数より
 *       前なので、教師には効かない。
 *   <li>局末（受け取り先のある経路だけ）: {@link HandEndAnalyzer} の結果（{@link HandEndFacts}）を作る。
 * </ul>
 *
 * <p>写しと分岐は、親の到達と河の数を写して始める（{@link #RowRecorder(RowRecorder)}）。分岐の根の打牌は、分岐の最初の判断の時点で数える。
 * 主経路は局ごとに {@link #reset} する。同時に触るのは 1 スレッドだけ。計測は経路に貯め、{@link #flush} で足す。記録のどこかで失敗したら、
 * その例外で収集を止める。
 */
final class RowRecorder {

  /** 受け取り先の判断の席のビットを、まだ求めていない印。 */
  private static final int UNKNOWN_SEATS = -1;

  private static final Step[] STEPS = Step.values();

  /** 行を書く入力の形式。 */
  private final InputFormat input;

  private final TenpaiReach reach;

  /** 前に見たときの、席ごとの河の打牌の数。 */
  private final int[] discards;

  private final long[] nanos = new long[STEPS.length];
  private final long[] calls = new long[STEPS.length];

  /**
   * 主経路の記録を、局の始めの状態で作る。
   *
   * @param variant 収集するルール（四人打ち・三人打ち）
   */
  RowRecorder(Variant variant) {
    input = InputFormat.of(variant);
    reach = new TenpaiReach(variant);
    discards = new int[variant.seats()];
  }

  /** 写しか分岐の記録を、親の今の状態（親の判断の時点）から作る。 */
  RowRecorder(RowRecorder parent) {
    input = parent.input;
    reach = new TenpaiReach(parent.reach);
    discards = parent.discards.clone();
  }

  /** 判断の時点の始めに呼ぶ。前に見てから増えた打牌を、聴牌への到達の追跡に渡す。 */
  void boundary(FullView full, List<Slot> targets) {
    discards(full, targets);
  }

  /**
   * 行を作る判断の記録を、推論の要求が運ぶ O を写して作る。
   *
   * @param request 行を作る判断の推論の要求（O を取り込んだもの）
   * @param o 要求の O を読むリーダー
   * @param full エンジンを進める前の局面（Oracle は全ての席の手を読む）
   * @return 判断の記録
   */
  Decision decision(PolicyRequest request, Observation o, FullView full, RuleWorkspace ws) {
    long started = System.nanoTime();
    ObservationBatch batch = request.records();
    long[] observation = new long[batch.longs(request.record())];
    System.arraycopy(batch.words(), batch.start(request.record()), observation, 0, observation.length);
    int events = o.events();
    int targets = 0;
    for (int action = 0, actions = o.legalActions(); action < actions; action++) {
      if (TargetSchema.hasCandidateTargets(input.role(o.action(action).type()))) {
        targets |= 1 << action;
      }
    }
    Decision decision = new Decision(observation, OracleCapture.capture(full, request.seat(), ws), events, targets);
    add(Step.ORACLE, started);
    return decision;
  }

  /**
   * 受け取り先のある経路の局末で、次局を始める前に呼ぶ。最後の打牌を聴牌への到達の追跡に渡し、局末の事実を作る。
   *
   * @return 局末の事実
   */
  HandEndFacts handEnd(FullView full, RoundEnd end, List<Slot> targets) {
    discards(full, targets);
    long started = System.nanoTime();
    HandEndFacts facts = HandEndAnalyzer.analyze(full, end, reach);
    add(Step.HAND_END, started);
    return facts;
  }

  /** 主経路が次局へ移るときに、到達と河の数を消す。 */
  void reset() {
    reach.reset();
    Arrays.fill(discards, 0);
  }

  /** 計測した時間と回数を {@code timer} に足して空にする。経路を 1 回進めた後に 1 回呼ぶ。 */
  void flush(StageTimer<CollectStage> timer) {
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

  /** 前に見てから河に増えた打牌を数え、受け取り先の判断の席の打牌を到達の追跡に渡す。 */
  private void discards(FullView full, List<Slot> targets) {
    EventLog events = full.seat(0).events();
    int seats = UNKNOWN_SEATS;
    for (int seat = 0; seat < discards.length; seat++) {
      int size = events.riverSize(seat);
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
        reach.discarded(full, seat, events.river(seat, size - 1));
        add(Step.REACH, started);
      }
    }
  }

  /** 受け取り先の判断の席を、席ごとのビットにして返す。 */
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
   * 行を作る判断の記録。判断の時点で決まり、同じ判断のラベル行と根の記録が共有する（配列は書き換えない）。
   *
   * @param observation 記録 O
   * @param oracle Oracle（{@link OracleCodec}）
   * @param events 判断の時点のイベントの数（O のヘッダーにある数。局末の 3 ワードで到達を数える基準）
   * @param targets 局末の 3 ワードを記録する手（{@link TargetSchema#hasCandidateTargets} が真の手）の位置のビット
   */
  record Decision(long[] observation, long[] oracle, int events, int targets) {

    /**
     * 手を打った経路の局末を、判断の席から見た 3 ワードにして書く。記録しない手では何も書かない。ドライバのスレッドで呼ぶ。
     *
     * @param codec ドライバのコーデック
     * @param end 経路の局末の事実
     * @param seat 判断の席
     * @param action 手の位置
     * @param outcomes 手ごとの 3 ワードの書き先（長さは合法手の数 × {@link OutcomeCodec#WORDS}）
     */
    void outcome(OutcomeCodec codec, HandEndFacts end, int seat, int action, long[] outcomes) {
      if ((targets >>> action & 1) != 0) {
        codec.write(end, seat, events, outcomes, action * OutcomeCodec.WORDS);
      }
    }
  }

  /** 計測する段。O の取り込みは卓のコアが測る。 */
  private enum Step {
    ORACLE(CollectStage.V2_ORACLE),
    REACH(CollectStage.V2_REACH),
    HAND_END(CollectStage.V2_HAND_END);

    private final CollectStage stage;

    Step(CollectStage stage) {
      this.stage = stage;
    }
  }
}
