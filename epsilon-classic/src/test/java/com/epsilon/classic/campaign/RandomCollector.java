package com.epsilon.classic.campaign;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.classic.learn.GameTrajectory;
import com.epsilon.classic.learn.TrajectoryBatch;
import com.epsilon.classic.learn.UncertaintyCdf;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * テスト用の自己対局。全ての席が合法手を macro の種から一様に選んで打ち、学習器の席（半荘の番号を席の数で割った余り）の判断を、振る舞いの
 * 確率 1/合法手の数で書く。選び替えで局の進みが変わりえたかは、判断の 1 割で false にする。奇数番目の半荘は、局ごとに乱数の GRP の
 * 周辺確率を持つ。半荘は決めた判断の数で打ち切り、終局の点は乱数で並べた決まった点にする。結果は macro の種だけで決まる。相手はいつも
 * 今の王者にする。
 *
 * <p>迷いの分布は、macro t なら全体の分布の区間 t に t を数えたもの。受け取った計画を覚え、前の macro の分布が渡されたかを確かめる。
 */
final class RandomCollector implements Collector {

  /** 1 つの半荘で書く判断の数の上限。 */
  static final int DECISIONS = 24;

  private static final int[] SCORES = {35_000, 28_000, 22_000, 15_000};

  private final List<Macro> plans;

  private RandomCollector(List<Macro> plans) {
    this.plans = plans;
  }

  /**
   * @param plans 受け取った計画を足していく先（開き直しても同じ列に足す）
   */
  static Factory factory(List<Macro> plans) {
    return (machine, runtime, table) -> new RandomCollector(plans);
  }

  /** macro の迷いの分布（macro 0 は空の分布と同じ数）。 */
  static UncertaintyCdf calibration(int macro) {
    long[][] counts = UncertaintyCdf.empty().counts();
    if (macro > 0) {
      counts[counts.length - 1][macro] = macro;
    }
    return UncertaintyCdf.of(counts);
  }

  @Override
  public Collected collect(Macro macro) {
    plans.add(macro);
    if (!Arrays.deepEquals(macro.calibration().counts(), calibration(macro.number() - 1).counts())) {
      throw new AssertionError("Macro " + macro.number() + " did not get the calibration of the macro before it");
    }
    Variant variant = macro.settings().run().variant();
    Random random = new Random(macro.seed());
    TrajectoryBatch.Collector games = new TrajectoryBatch.Collector(variant);
    for (int game = 0; game < macro.settings().schedule().games(); game++) {
      long id = ((long) macro.number() << 20) + game;
      games.add(game(variant, random, id, game % variant.seats(), game % 2 == 1));
    }
    return new Collected(games.build(), calibration(macro.number()), macro.champion());
  }

  @Override
  public void close() {}

  private static GameTrajectory game(Variant variant, Random random, long id, int learner, boolean priors) {
    Engine engine = new Engine(variant);
    RuleWorkspace ws = new RuleWorkspace(variant);
    ObservationWriter writer = new ObservationWriter(variant);
    ObservationBatch records = new ObservationBatch(variant, 1 << 12);
    Observation reader = new Observation(variant);
    int[] choices = new int[variant.seats() - 1];
    GameTrajectory.Builder builder = new GameTrajectory.Builder(variant, id);
    engine.startGame(random.nextLong(), DrawOrder.SEAT_STREAMS);
    Boundary boundary = engine.advance(ws);
    builder.hand(priors ? prior(variant, random) : null);
    int written = 0;
    while (written < DECISIONS && boundary != Boundary.GAME_END) {
      if (boundary == Boundary.DECISIONS) {
        Decisions decisions = engine.decisions();
        records.clear();
        for (int slot = 0; slot < decisions.size(); slot++) {
          int seat = decisions.seat(slot);
          Observation o = reader.bind(records, writer.capture(engine.view(seat), decisions, slot, records));
          int legal = o.legalActions();
          choices[slot] = random.nextInt(legal);
          if (seat == learner && written < DECISIONS) {
            builder.decision(o, choices[slot], 1f / legal, random.nextInt(10) != 0);
            written++;
          }
        }
        boundary = engine.commit(choices, ws);
      } else {
        boundary = engine.advance(ws);
        if (boundary != Boundary.GAME_END) {
          builder.hand(priors ? prior(variant, random) : null);
        }
      }
    }
    int[] scores = new int[variant.seats()];
    List<Integer> order = new ArrayList<>(List.of(0, 1, 2, 3));
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = SCORES[order.remove(random.nextInt(order.size()))];
    }
    return builder.finish(scores);
  }

  /** 絶対席 × 順位の乱数の周辺確率（席ごとに和が 1）。 */
  private static float[] prior(Variant variant, Random random) {
    int seats = variant.seats();
    float[] prior = new float[seats * seats];
    for (int seat = 0; seat < seats; seat++) {
      double sum = 0;
      for (int rank = 0; rank < seats; rank++) {
        prior[seat * seats + rank] = 0.05f + random.nextFloat();
        sum += prior[seat * seats + rank];
      }
      for (int rank = 0; rank < seats; rank++) {
        prior[seat * seats + rank] = (float) (prior[seat * seats + rank] / sum);
      }
    }
    return prior;
  }
}
