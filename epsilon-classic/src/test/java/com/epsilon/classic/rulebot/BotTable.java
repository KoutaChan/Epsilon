package com.epsilon.classic.rulebot;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Variant;
import java.util.Random;

/**
 * 全ての席をルールベースの打ち手が打つテスト用の卓。判断の時点ごとに、判断する全ての席の記録を作って手を選ぶ。局末では次の局を配り、
 * 終局では次の半荘を始める。各半荘のシードは、作るときに渡したシードから決まる。ウォームアップの後はメモリを確保しない。
 */
final class BotTable {

  private final RuleBot bot;
  private final Engine engine;
  private final RuleWorkspace ws;
  private final ObservationWriter writer;
  private final ObservationBatch batch;
  private final Observation observation;
  private final Random seeds;
  private final int[] choices;
  private Boundary boundary;
  private int games;
  private int decisions;
  private int rounds;
  private int wins;
  private int riichi;
  private int nuki;
  private long digest;

  BotTable(RuleBot bot, long seed) {
    this.bot = bot;
    Variant variant = bot.variant();
    engine = new Engine(variant);
    ws = new RuleWorkspace(variant);
    writer = new ObservationWriter(variant);
    batch = new ObservationBatch(variant, 1_024);
    observation = new Observation(variant);
    seeds = new Random(seed);
    choices = new int[variant.seats() - 1];
    engine.startGame(seeds.nextLong(), DrawOrder.WALL_ORDER);
    boundary = engine.advance(ws);
  }

  /** 1 ステップ進めて、止まった時点を返す。 */
  Boundary step() {
    switch (boundary) {
      case DECISIONS -> {
        Decisions open = engine.decisions();
        batch.clear();
        for (int slot = 0; slot < open.size(); slot++) {
          observation.bind(batch, writer.capture(engine.view(open.seat(slot)), open, slot, batch));
          choices[slot] = bot.act(observation);
          nuki += observation.action(choices[slot]).type() == Action.Type.NUKI ? 1 : 0;
          digest = digest * 31 + open.seat(slot) * 64 + choices[slot];
          decisions++;
        }
        boundary = engine.commit(choices, ws);
        if (boundary != Boundary.DECISIONS) {
          RoundOutcome outcome = engine.roundEnd().outcome();
          rounds++;
          wins += outcome.kind().win() ? 1 : 0;
          riichi += Integer.bitCount(outcome.riichiMask());
        }
      }
      case ROUND_END -> boundary = engine.advance(ws);
      case GAME_END -> {
        games++;
        engine.startGame(seeds.nextLong(), DrawOrder.WALL_ORDER);
        boundary = engine.advance(ws);
      }
    }
    return boundary;
  }

  /**
   * 終局まで進める。
   *
   * @param steps ステップ数の上限（超えたら {@code IllegalStateException} を投げる）
   */
  void finishGame(int steps) {
    for (int step = 0; step < steps; step++) {
      if (step() == Boundary.GAME_END) {
        return;
      }
    }
    throw new IllegalStateException("The game did not end within " + steps + " steps");
  }

  Engine engine() {
    return engine;
  }

  /** 終えた半荘の数。 */
  int games() {
    return games;
  }

  /** 手を選んだ判断の数（席ごとに数える）。 */
  int decisions() {
    return decisions;
  }

  /** 終えた局の数。 */
  int rounds() {
    return rounds;
  }

  /** 和了で終えた局の数。 */
  int wins() {
    return wins;
  }

  /** 終えた局で立直した席の数を、全ての局で足したもの。 */
  int riichi() {
    return riichi;
  }

  /** 選んだ北抜きの数。 */
  int nuki() {
    return nuki;
  }

  /** 選んだ手（席と手）を順に混ぜたダイジェスト。 */
  long digest() {
    return digest;
  }
}
