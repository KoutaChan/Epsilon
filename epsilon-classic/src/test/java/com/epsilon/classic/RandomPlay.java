package com.epsilon.classic;

import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.Variant;
import java.util.Random;

/**
 * テスト用のランダムな自己対局。判断する席はどれも、半分の確率で最初の合法手（和了・槓・鳴き・立直の打牌が先に並ぶ）を選び、残りの半分は
 * 合法手から一様に選ぶ。そのため、和了・槓・槍槓の応答が、一様に選ぶよりも多く起きる。局末では次の局を配り、終局では次の半荘を始める。
 * 手の選び方と半荘のシードは、作るときに渡したシードで決まる。ウォームアップの後はメモリを確保しない。
 */
public final class RandomPlay {

  public final Engine engine;
  public final RuleWorkspace ws;
  private final Random random;
  private final int[] choices;
  private Boundary boundary;

  public RandomPlay(Variant variant, long seed) {
    engine = new Engine(variant);
    ws = new RuleWorkspace(variant);
    random = new Random(seed);
    choices = new int[variant.seats() - 1];
    engine.startGame(random.nextLong(), random.nextBoolean() ? DrawOrder.SEAT_STREAMS : DrawOrder.WALL_ORDER);
    boundary = engine.advance(ws);
  }

  public Boundary boundary() {
    return boundary;
  }

  /** 1 ステップ進めて、止まった時点を返す。 */
  public Boundary step() {
    switch (boundary) {
      case DECISIONS -> {
        Decisions decisions = engine.decisions();
        for (int slot = 0; slot < decisions.size(); slot++) {
          choices[slot] = random.nextBoolean() ? 0 : random.nextInt(decisions.legalCount(slot));
        }
        boundary = engine.commit(choices, ws);
      }
      case ROUND_END -> boundary = engine.advance(ws);
      case GAME_END -> {
        engine.startGame(random.nextLong(), random.nextBoolean() ? DrawOrder.SEAT_STREAMS : DrawOrder.WALL_ORDER);
        boundary = engine.advance(ws);
      }
    }
    return boundary;
  }
}
