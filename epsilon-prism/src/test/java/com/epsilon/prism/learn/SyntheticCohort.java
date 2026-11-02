package com.epsilon.prism.learn;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.prism.DecisionWalk;
import com.epsilon.prism.auxiliary.HandEndAnalyzer;
import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.auxiliary.OracleCapture;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.auxiliary.TenpaiReach;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.observe.ObservationArena;
import com.epsilon.prism.observe.ObservationSchema;
import com.epsilon.prism.observe.ObservationWriter;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * 試験の合成のコホート。GREEDY で打った半荘の判断のうち {@code every} 判断に 1 つを行にし、実際の O・神託・局末の 3 語（主経路の局末を、測った
 * 手のうち候補の教師を使う手に付ける）に、乱数の Y を付ける。行の 7 つに 1 つは測った手が 1 つ（学習の行に入らない）、10 に 1 つは HOLDOUT、
 * 5 つに 1 つは神託なし。帳簿は持たない（欄の数 0）。
 */
final class SyntheticCohort implements DecisionWalk.Visitor {

  /** 局末を待つ判断。 */
  private record Pending(
      EngineDecisionKind kind, int seat, int events, long[] observation, long[] oracle) {}

  private final ObservationWriter writer = new ObservationWriter();
  private final ObservationArena arena = new ObservationArena(256);
  private final Observation observation = new Observation();
  private final OracleCapture oracles = new OracleCapture();
  private final HandEndAnalyzer analyzer = new HandEndAnalyzer();
  private final OutcomeCodec outcomes = new OutcomeCodec();
  private final TenpaiReach reach = new TenpaiReach();
  private final int[] riverSizes = new int[GameState.NUM_PLAYERS];
  private final List<Pending> pending = new ArrayList<>();
  private final List<Row> rows = new ArrayList<>();
  private final SplittableRandom random;
  private final int every;
  private int visited;

  private SyntheticCohort(long seed, int every) {
    random = new SplittableRandom(seed);
    this.every = every;
  }

  /**
   * 行を作る。
   *
   * @param seed 最初の半荘の種
   * @param games 半荘の数
   * @param every 行にする判断の間隔
   * @return 行
   */
  static List<Row> rows(long seed, int games, int every) {
    SyntheticCohort cohort = new SyntheticCohort(seed, every);
    new DecisionWalk(DecisionWalk.Policy.GREEDY).run(seed, games, cohort);
    return cohort.rows;
  }

  /**
   * 行をコホートに書いて封印する。
   *
   * @param directory コホートのディレクトリ
   * @param iteration コホート
   * @param rows 行
   * @throws IOException 書けないとき
   */
  static void write(Path directory, int iteration, List<Row> rows) throws IOException {
    try (CohortWriter writer = new CohortWriter(directory, iteration, 0, 2, CohortWriter.Tap.NONE)) {
      for (Row row : rows) {
        writer.accept(row);
      }
      writer.seal();
    }
  }

  /**
   * 学習の行（TRAIN で測った手が 2 つ以上）の数を返す。
   *
   * @param rows 行
   * @return 数
   */
  static int training(List<Row> rows) {
    int count = 0;
    for (Row row : rows) {
      if (row.purpose() == Row.Purpose.TRAIN && row.measuredCount() >= 2) {
        count++;
      }
    }
    return count;
  }

  @Override
  public void awaiting(GameEngine engine, List<EngineDecisionPoint> points, int round) {
    discards(engine.getState());
  }

  @Override
  public void visit(GameEngine engine, EngineDecisionPoint point, int round) {
    if (visited++ % every != 0) {
      return;
    }
    int seat = point.player();
    arena.clear();
    int base = writer.capture(engine.observation(seat), point.kind(), point.legalActions(), arena);
    long[] words = arena.words();
    long[] o = Arrays.copyOfRange(words, base, base + ObservationSchema.longs(words, base));
    int events = observation.bind(o, 0).events();
    long[] oracle =
        (rows.size() + pending.size()) % 5 == 4
            ? new long[0]
            : oracles.capture(engine.getState(), seat);
    pending.add(new Pending(point.kind(), seat, events, o, oracle));
  }

  @Override
  public void settled(GameEngine engine, RoundSettlement settlement, int round) {
    GameState state = engine.getState();
    discards(state);
    HandEndFacts facts = analyzer.analyze(state, settlement, reach);
    for (Pending decision : pending) {
      rows.add(row(rows.size(), decision, facts));
    }
    pending.clear();
    reach.reset();
    Arrays.fill(riverSizes, 0);
  }

  /** 前に見てから増えた打牌ごとに、聴牌への到達の追跡を進める。 */
  private void discards(GameState state) {
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      while (riverSizes[seat] < state.river(seat).size()) {
        riverSizes[seat]++;
        reach.discarded(state, seat);
      }
    }
  }

  private Row row(int id, Pending decision, HandEndFacts facts) {
    Observation o = observation.bind(decision.observation(), 0);
    int actions = o.legalActions();
    boolean single = id % 7 == 3;
    float[] values = new float[actions];
    float[] prior = new float[actions];
    long[] words = new long[actions * OutcomeCodec.WORDS];
    Arrays.fill(values, Float.NaN);
    Arrays.fill(prior, 1f / actions);
    int support = 0;
    for (int action = 0; action < actions; action++) {
      if (single && action > 0) {
        continue;
      }
      support |= 1 << action;
      values[action] = random.nextFloat() * 2 - 1;
      Action chosen = o.action(action);
      if (TargetSchema.hasCandidateTargets(ActionRole.of(chosen.type()))) {
        outcomes.write(
            facts, decision.seat(), decision.events(), words, action * OutcomeCodec.WORDS);
      }
    }
    boolean response =
        decision.kind() == EngineDecisionKind.RESPONSE
            || decision.kind() == EngineDecisionKind.CHANKAN;
    return new Row(
        id,
        id,
        response ? DecisionClass.CALL : DecisionClass.DISCARD_MID,
        id % 10 == 9 ? Row.Purpose.HOLDOUT : Row.Purpose.TRAIN,
        id % 2 == 0 ? Row.Group.A : Row.Group.B,
        Row.Source.MAIN,
        0,
        values,
        prior,
        support,
        -1,
        Float.NaN,
        Float.NaN,
        -1,
        new float[0],
        words,
        decision.oracle(),
        decision.observation());
  }
}
