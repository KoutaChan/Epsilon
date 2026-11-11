package com.epsilon.prism.luck;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import java.util.Random;

/**
 * 裏ドラの期待値のテストで使う自己対局。立直・和了・槓・北抜きをよく選ぶ打ち手で、種で決まる半荘を続けて打ち、局末ごとに受け取り手に渡す。
 *
 * <p>和了は 8 回に 7 回、立直は 4 回に 3 回、槓と北抜きは 2 回に 1 回、チー・ポン・大明槓は 6 回に 1 回選ぶ。ほかは見送りか、向聴を最も
 * 下げる打牌。8 回に 1 回は合法手から一様に選ぶ。こうするのは、槓のある立直の和了とダブロンを十分に起こすため。
 */
final class UraGames {

  /** 局末を受け取る。エンジンの状態を変えてはならない。 */
  @FunctionalInterface
  interface Visitor {

    /**
     * @param engine 局末で止まっているエンジン
     * @param kan この局で槓が成立したか、加槓を宣言したなら true（打ち手の選んだ手から数える）
     */
    void handEnd(Engine engine, boolean kan);
  }

  private final Variant variant;
  private final ShapeAnalyzer analyzer;
  private final HandShape shape = new HandShape();

  UraGames(Variant variant) {
    this.variant = variant;
    analyzer = new ShapeAnalyzer(variant);
  }

  /**
   * 種から半荘を続けて打つ。
   *
   * @param seed 半荘の牌山と、選ぶ手を決める乱数の種
   * @param handEnds 渡す局末の数
   * @param visitor 局末を受け取る
   */
  void play(long seed, int handEnds, Visitor visitor) {
    Random random = new Random(seed);
    Engine engine = new Engine(variant);
    RuleWorkspace ws = new RuleWorkspace(variant);
    int[] choices = new int[variant.seats() - 1];
    engine.startGame(random.nextLong(), DrawOrder.SEAT_STREAMS);
    Boundary at = engine.advance(ws);
    boolean kan = false;
    int seen = 0;
    while (seen < handEnds) {
      if (at == Boundary.DECISIONS) {
        Decisions decisions = engine.decisions();
        boolean closedOrAdded = false;
        boolean open = false;
        for (int slot = 0; slot < decisions.size(); slot++) {
          choices[slot] = choose(engine, slot, random);
          switch (Action.of(decisions.legal(slot, choices[slot])).type()) {
            case ANKAN, KAKAN -> closedOrAdded = true;
            case DAIMINKAN -> open = true;
            default -> {}
          }
        }
        at = engine.commit(choices, ws);
        // 大明槓は、同じ打牌へのロンで局が終われば成立しない
        kan |= closedOrAdded || open && at == Boundary.DECISIONS;
        continue;
      }
      visitor.handEnd(engine, kan);
      seen++;
      kan = false;
      if (at == Boundary.GAME_END) {
        engine.startGame(random.nextLong(), DrawOrder.SEAT_STREAMS);
      }
      at = engine.advance(ws);
    }
  }

  private int choose(Engine engine, int slot, Random random) {
    Decisions decisions = engine.decisions();
    int count = decisions.legalCount(slot);
    if (random.nextInt(8) == 0) {
      return random.nextInt(count);
    }
    int win = -1;
    int kan = -1;
    int call = -1;
    int pass = -1;
    int riichi = -1;
    int riichis = 0;
    for (int index = 0; index < count; index++) {
      switch (Action.of(decisions.legal(slot, index)).type()) {
        case TSUMO_AGARI, RON_AGARI -> win = index;
        case ANKAN, KAKAN, NUKI -> kan = kan < 0 ? index : kan;
        case CHI, PON, DAIMINKAN -> call = call < 0 ? index : call;
        case PASS -> pass = index;
        case RIICHI_DAHAI -> {
          riichi = riichi < 0 ? index : riichi;
          riichis++;
        }
        default -> {}
      }
    }
    if (win >= 0 && random.nextInt(8) > 0) {
      return win;
    }
    if (kan >= 0 && random.nextBoolean()) {
      return kan;
    }
    if (call >= 0 && random.nextInt(6) == 0) {
      return call;
    }
    if (pass >= 0) {
      return pass;
    }
    if (riichis > 0 && random.nextInt(4) > 0) {
      return riichi + random.nextInt(riichis);
    }
    engine.full().copyHand(decisions.seat(slot), shape);
    int best = Integer.MAX_VALUE;
    int chosen = 0;
    int ties = 0;
    for (int index = 0; index < count; index++) {
      Action action = Action.of(decisions.legal(slot, index));
      if (action.type() != Action.Type.DAHAI) {
        continue;
      }
      shape.remove(action.tile());
      int shanten = analyzer.shanten(shape);
      shape.add(action.tile());
      if (shanten < best) {
        best = shanten;
        chosen = index;
        ties = 1;
      } else if (shanten == best && random.nextInt(++ties) == 0) {
        chosen = index;
      }
    }
    return chosen;
  }
}
