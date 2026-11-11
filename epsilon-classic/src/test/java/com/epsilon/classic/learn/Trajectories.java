package com.epsilon.classic.learn;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * テスト用の自己対局の軌跡。全ての席が合法手を一様に選んで打ち、全ての席の判断を振る舞いの確率 1/合法手の数で書く。選び替えで局の進みが
 * 変わりえたかは、判断の 1 割で false にする。奇数番目の半荘は、局ごとに乱数の GRP の周辺確率（行の和が 1）を持つ。
 * 半荘は決めた判断の数で打ち切り、終局の点は乱数で並べた 35,000・28,000・22,000・15,000 点にする。シードだけで決まる。
 */
final class Trajectories {

  private static final int[] SCORES = {35_000, 28_000, 22_000, 15_000};

  private Trajectories() {}

  /**
   * @param games 半荘の数
   * @param decisions 1 つの半荘で書く判断の数の上限
   */
  static TrajectoryBatch play(Variant variant, long seed, int games, int decisions) {
    Random random = new Random(seed);
    TrajectoryBatch.Collector collector = new TrajectoryBatch.Collector(variant);
    for (int game = 0; game < games; game++) {
      collector.add(game(variant, random, 1_000L + game, game % 2 == 1, decisions));
    }
    return collector.build();
  }

  private static GameTrajectory game(Variant variant, Random random, long id, boolean priors, int limit) {
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
    while (written < limit && boundary != Boundary.GAME_END) {
      if (boundary == Boundary.DECISIONS) {
        Decisions decisions = engine.decisions();
        records.clear();
        for (int slot = 0; slot < decisions.size(); slot++) {
          Observation o = reader.bind(records, writer.capture(engine.view(decisions.seat(slot)), decisions, slot, records));
          int legal = o.legalActions();
          choices[slot] = random.nextInt(legal);
          if (written < limit) {
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
  static float[] prior(Variant variant, Random random) {
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

  /** 席ごとに、ランダムな自己対局から集めた判断。 */
  static final class Pool {

    private final ObservationBatch store;
    private final List<List<Integer>> bySeat = new ArrayList<>();

    /**
     * @param perSeat 席ごとに集める判断の数
     */
    Pool(Variant variant, long seed, int perSeat) {
      store = new ObservationBatch(variant, 1 << 14);
      for (int seat = 0; seat < variant.seats(); seat++) {
        bySeat.add(new ArrayList<>());
      }
      Random random = new Random(seed);
      Engine engine = new Engine(variant);
      RuleWorkspace ws = new RuleWorkspace(variant);
      ObservationWriter writer = new ObservationWriter(variant);
      ObservationBatch scratch = new ObservationBatch(variant, 1 << 12);
      Observation reader = new Observation(variant);
      int[] choices = new int[variant.seats() - 1];
      engine.startGame(random.nextLong(), DrawOrder.SEAT_STREAMS);
      Boundary boundary = engine.advance(ws);
      while (!full(perSeat)) {
        switch (boundary) {
          case DECISIONS -> {
            Decisions decisions = engine.decisions();
            scratch.clear();
            for (int slot = 0; slot < decisions.size(); slot++) {
              int seat = decisions.seat(slot);
              Observation o = reader.bind(scratch, writer.capture(engine.view(seat), decisions, slot, scratch));
              choices[slot] = random.nextInt(o.legalActions());
              if (bySeat.get(o.observer()).size() < perSeat) {
                bySeat.get(o.observer()).add(store.add(o));
              }
            }
            boundary = engine.commit(choices, ws);
          }
          case ROUND_END -> boundary = engine.advance(ws);
          case GAME_END -> {
            engine.startGame(random.nextLong(), DrawOrder.SEAT_STREAMS);
            boundary = engine.advance(ws);
          }
        }
      }
    }

    /** その席の {@code index} 番目の判断を読む。 */
    Observation bind(int seat, int index, Observation reader) {
      return reader.bind(store, bySeat.get(seat).get(index));
    }

    private boolean full(int perSeat) {
      for (List<Integer> seat : bySeat) {
        if (seat.size() < perSeat) {
          return false;
        }
      }
      return true;
    }
  }
}
