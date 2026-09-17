package com.epsilon.ai.belief;

import com.epsilon.core.GameState;
import com.epsilon.core.Tile;

/** 各系列で共有するBelief教師データと予測値の配列配置。相対席1〜3を他家インデックス0〜2へ並べる。WAITという名前の区分も、テンパイ前は受け入れ牌、テンパイ時は待ち牌を表す。 */
public final class EpsilonBeliefLayout {

  public static final int OPPONENT_COUNT = GameState.NUM_PLAYERS - 1;
  public static final int OPPONENT_HAND_SIZE = OPPONENT_COUNT * Tile.NUM_TILE_TYPES;
  public static final int OPPONENT_WAIT_SIZE = OPPONENT_COUNT * Tile.NUM_TILE_TYPES;
  public static final int OPPONENT_SCALAR_COUNT = 2;
  public static final int SCALAR_SIZE = OPPONENT_COUNT * OPPONENT_SCALAR_COUNT;
  public static final int OUTPUT_SIZE = OPPONENT_HAND_SIZE + OPPONENT_WAIT_SIZE + SCALAR_SIZE;
  public static final int SCALAR_SHANTEN = 0;
  public static final int SCALAR_TENPAI = 1;

  private EpsilonBeliefLayout() {}
}
