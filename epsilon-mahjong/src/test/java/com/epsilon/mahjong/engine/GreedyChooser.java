package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import java.util.Random;

/**
 * 和了と立直をよく選ぶ、テスト用の打ち手。和了は 8 回に 7 回、九種九牌と立直は 2 回に 1 回選び、槓・北抜き・鳴きも
 * ときどき選ぶ。それ以外は、見送るか、向聴数がいちばん小さくなる打牌を選ぶ（同じ向聴数の打牌からは一様に選ぶ）。また、
 * 8 回に 1 回は合法手から一様に選ぶ。立直での和了・ダブロン・流局・途中流局を十分な数だけ起こすためである。
 */
final class GreedyChooser implements RandomGames.Chooser {

  private final ShapeAnalyzer analyzer;
  private final HandShape shape = new HandShape();

  GreedyChooser(Variant variant) {
    analyzer = new ShapeAnalyzer(variant);
  }

  @Override
  public int choose(Engine engine, int slot, Random random) {
    Decisions decisions = engine.decisions();
    int count = decisions.legalCount(slot);
    if (random.nextInt(8) == 0) {
      return random.nextInt(count);
    }
    int win = -1;
    int kyushu = -1;
    int kan = -1;
    int call = -1;
    int pass = -1;
    int riichi = -1;
    int riichis = 0;
    for (int index = 0; index < count; index++) {
      switch (Action.of(decisions.legal(slot, index)).type()) {
        case TSUMO_AGARI, RON_AGARI -> win = index;
        case KYUSHU_KYUHAI -> kyushu = index;
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
    if (kyushu >= 0 && random.nextBoolean()) {
      return kyushu;
    }
    if (kan >= 0 && random.nextInt(3) == 0) {
      return kan;
    }
    if (call >= 0 && random.nextInt(6) == 0) {
      return call;
    }
    if (pass >= 0) {
      return pass;
    }
    if (riichis > 0 && random.nextBoolean()) {
      return riichi + random.nextInt(riichis);
    }
    return bestDiscard(engine, slot, random);
  }

  private int bestDiscard(Engine engine, int slot, Random random) {
    Decisions decisions = engine.decisions();
    engine.full().copyHand(decisions.seat(slot), shape);
    int best = Integer.MAX_VALUE;
    int chosen = 0;
    int ties = 0;
    for (int index = 0; index < decisions.legalCount(slot); index++) {
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
