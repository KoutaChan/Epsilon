package com.epsilon.ai.belief;

import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Tile;

/**
 * 全席の手牌を使い、観測者以外の3人についてBeliefモデルの教師データを生成する。
 *
 * <p>手牌分布の学習に使う残数は、各牌種の4枚から公開牌と観測者の手牌を引く。受け入れ牌の教師では、公開牌と対象の他家自身の手牌を引いた残数を使い、シャンテン数を改善する牌だけを選ぶ。テンパイ時は待ち牌に対応する。役の有無やフリテンは調べず、副露・槓を各3枚と数えて14枚の手牌では受け入れ牌を空にする。
 */
public final class EpsilonBeliefTargetBuilder {

  private final HandShapeAnalyzer shapes = new HandShapeAnalyzer();

  public EpsilonBeliefTargetBuilder() {}

  /**
   * 完全情報局面から観測者以外3家のBelief教師を構築する。
   *
   * @param state 全手牌を参照できる牌譜再生または自己対戦局面
   * @param observer 公開観測を行う絶対席
   * @return 他家手牌、観測者から見えない牌の残数、シャンテン数、テンパイ判定、受け入れ牌マスク
   */
  public EpsilonBeliefTarget build(GameState state, int observer) {
    float[] opponentHands = new float[EpsilonBeliefLayout.OPPONENT_HAND_SIZE];
    float[] opponentShanten = new float[EpsilonBeliefLayout.OPPONENT_COUNT];
    float[] opponentTenpai = new float[EpsilonBeliefLayout.OPPONENT_COUNT];
    float[] opponentWaits = new float[EpsilonBeliefLayout.OPPONENT_WAIT_SIZE];

    float[] hiddenTiles = new float[Tile.NUM_TILE_TYPES];
    Hand ownHand = state.hand(observer);
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      int remaining =
          Tile.TILES_PER_TYPE - state.publicState().visibleTileCount(tile) - ownHand.count(tile);
      if (remaining < 0)
        throw new IllegalStateException("visible + own hand exceeds four copies for tile " + tile);
      hiddenTiles[tile] = remaining;
    }

    for (int rel = 1; rel < GameState.NUM_PLAYERS; rel++) {
      int seat = (observer + rel) % GameState.NUM_PLAYERS;
      int opponent = rel - 1;
      Hand hand = state.hand(seat);
      int shanten = shapes.calculateMinimum(hand);
      opponentShanten[opponent] = shanten;
      opponentTenpai[opponent] = shanten <= 0 ? 1.0f : 0.0f;

      for (long waits = shapes.improvingTileTypeMask(hand, shanten);
          waits != 0L;
          waits &= waits - 1) {
        int tile = Long.numberOfTrailingZeros(waits);
        if (Tile.TILES_PER_TYPE - state.publicState().visibleTileCount(tile) - hand.count(tile) > 0)
          opponentWaits[opponent * Tile.NUM_TILE_TYPES + tile] = 1.0f;
      }

      for (int t = 0; t < Tile.NUM_TILE_TYPES; t++) {
        opponentHands[opponent * Tile.NUM_TILE_TYPES + t] = hand.count(t);
      }
    }

    return new EpsilonBeliefTarget(
        opponentHands, hiddenTiles, opponentShanten, opponentTenpai, opponentWaits);
  }
}
