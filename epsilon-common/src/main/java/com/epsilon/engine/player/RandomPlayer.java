package com.epsilon.engine.player;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.Player;
import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;

/** ランダムに合法アクションを選択するプレイヤー（デバッグ・ベースライン用）。 */
public final class RandomPlayer implements Player {

  private final SplittableRandom random;

  /** デバッグ用のランダムプレイヤーを生成する。 */
  public RandomPlayer() {
    this(new Random().nextLong());
  }

  /** 固定シードから再現可能なランダムプレイヤーを生成する。 */
  public RandomPlayer(long seed) {
    random = new SplittableRandom(seed);
  }

  @Override
  public Action selectAction(GameState state, int playerIndex, List<Action> legalActions) {
    return legalActions.get(random.nextInt(legalActions.size()));
  }
}
