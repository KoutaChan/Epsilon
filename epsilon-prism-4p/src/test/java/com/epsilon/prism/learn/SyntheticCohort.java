package com.epsilon.prism.learn;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.prism.HandEndTracker;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.auxiliary.OracleCapture;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism4p.input.InputSchema.ActionRole;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * テスト用に合成したコホート。GREEDY で打った半荘の判断を {@code every} 個に 1 つ行にする。行には実際の観測（O）と Oracle、局末の結果（手ごとに
 * 3 ワード）を入れ、Y は乱数にする。局末の結果は主経路のもので、測った手のうち候補の教師を使う手にだけ付ける。行の 7 つに 1 つは
 * 測った手が 1 つだけ（学習の行に入らない）、10 に 1 つは HOLDOUT、5 つに 1 つは Oracle なし。帳簿は持たない（帳簿のフィールドの数は 0）。
 */
final class SyntheticCohort implements TableWalk.Visitor {

  /** 局末を待っている判断。 */
  private record Pending(EngineDecisionKind kind, int seat, int events, long[] observation, long[] oracle) {}

  private final ObservationWriter writer = new ObservationWriter(TENHOU_4P);
  private final ObservationBatch batch = new ObservationBatch(TENHOU_4P, 1 << 12);
  private final Observation observation = new Observation(TENHOU_4P);
  private final RuleWorkspace ws = new RuleWorkspace(TENHOU_4P);
  private final OutcomeCodec outcomes = OutcomeCodec.of(TENHOU_4P);
  private final HandEndTracker tracker = new HandEndTracker(TENHOU_4P);
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
   * @param seed 最初の半荘の乱数の種
   * @param games 半荘の数
   * @param every 行にする判断の間隔
   * @return 行
   */
  static List<Row> rows(long seed, int games, int every) {
    SyntheticCohort cohort = new SyntheticCohort(seed, every);
    new TableWalk(TENHOU_4P, TableWalk.Policy.GREEDY).run(seed, games, cohort);
    return cohort.rows;
  }

  /**
   * 行をコホートに書いて確定する。
   *
   * @param directory コホートのディレクトリ
   * @param iteration コホートの番号
   * @param rows 行
   * @throws IOException 書けないとき
   */
  static void write(Path directory, int iteration, List<Row> rows) throws IOException {
    try (CohortWriter writer = new CohortWriter(directory, TENHOU_4P, iteration, 0, 2, CohortWriter.Tap.NONE)) {
      for (Row row : rows) {
        writer.accept(row);
      }
      writer.seal();
    }
  }

  /**
   * 学習の行（TRAIN で、測った手が 2 つ以上ある行）の数を返す。
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
  public void decision(Engine engine, int slot) {
    tracker.scan(engine.full());
    if (visited++ % every != 0) {
      return;
    }
    Decisions decisions = engine.decisions();
    int seat = decisions.seat(slot);
    batch.clear();
    int index = writer.capture(engine.view(seat), decisions, slot, batch);
    long[] o = Arrays.copyOfRange(batch.words(), batch.start(index), batch.start(index) + batch.longs(index));
    int events = observation.bind(o, 0).events();
    long[] oracle =
        (rows.size() + pending.size()) % 5 == 4 ? new long[0] : OracleCapture.capture(engine.full(), seat, ws);
    pending.add(new Pending(decisions.kind(slot), seat, events, o, oracle));
  }

  @Override
  public void roundEnd(Engine engine) {
    HandEndFacts facts = tracker.end(engine);
    for (Pending decision : pending) {
      rows.add(row(rows.size(), decision, facts));
    }
    pending.clear();
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
      if (TargetSchema.hasCandidateTargets(ActionRole.of(o.action(action).type()))) {
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
