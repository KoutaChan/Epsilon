package com.epsilon.prism.auxiliary;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.WinLegality;
import com.epsilon.engine.WinPointDeltas;
import java.util.Arrays;
import java.util.Objects;

/**
 * 判断の時点の全ての状態（commit の前、4 人の手）から神託（{@link OracleCodec}）を作る。行に残す判断の record の中で呼ぶ。進行ワーカーに
 * 1 つ持ち、作業域（解析器・採点器・下書き・打牌の出来事）は中に持つ。
 *
 * <p>相手ごとに、最小の向聴と形の待ちを出し、聴牌なら待ちの牌ごとに「判断の席が今その牌を捨てたらロンできるか」をエンジンの判定
 * （{@link WinLegality#canRon}）で確かめる。ロンできる牌は待ちの牌に限られるので、待ちの外は確かめない。ロンできる牌は、そのロンの和了の
 * 点（裏ドラなし。条件は捨てた時点の局況で、一発・河底を含む）を基本点から {@link WinPointDeltas} で作る。打牌の出来事は席 × 牌の
 * 136 個を前もって作り、捕捉のたびに割り当てない（割り当てるのは返す神託の配列だけ）。
 */
public final class OracleCapture {

  /** 前もって作る打牌の出来事の数（判断の席 × 牌）。 */
  private static final int DISCARDS = GameState.NUM_PLAYERS * Tile.NUM_TILE_TYPES;

  private final HandShapeAnalyzer shapes = new HandShapeAnalyzer();
  private final HandScoreEvaluator scores = new HandScoreEvaluator();
  private final HandScoreBuffer score = new HandScoreBuffer();
  private final int[] deltas = new int[GameState.NUM_PLAYERS];
  private final long[] draft = new long[OracleCodec.MAX_LONGS];
  private final TurnEvent.Discard[] discards = new TurnEvent.Discard[DISCARDS];

  public OracleCapture() {
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
        discards[seat * Tile.NUM_TILE_TYPES + tile] = new TurnEvent.Discard(seat, tile, false);
      }
    }
  }

  /**
   * 神託を作る。
   *
   * @param state 判断の時点の局面（全員の手を読む。変えない）
   * @param seat 判断の席（絶対席）
   * @return 神託（{@link OracleCodec#longs} の長さ）
   * @throws IllegalStateException 相手の手が 13 枚でないか、ロンできる牌に役がないとき
   */
  public long[] capture(GameState state, int seat) {
    Objects.checkIndex(seat, GameState.NUM_PLAYERS);
    Arrays.fill(draft, 0L);
    int dangers = 0;
    for (int opponent = 1; opponent < GameState.NUM_PLAYERS; opponent++) {
      int player = (seat + opponent) % GameState.NUM_PLAYERS;
      Hand hand = state.hand(player);
      if (hand.effectiveTileCount() != 13) {
        throw new IllegalStateException(
            "opponent " + player + " holds " + hand.effectiveTileCount() + " tiles at a decision");
      }
      int shanten = shapes.calculateMinimum(hand);
      long waits = shanten == 0 ? shapes.agariTileTypeMask(hand) : 0L;
      long danger = 0L;
      for (long rest = waits; rest != 0L; rest &= rest - 1) {
        int tile = Long.numberOfTrailingZeros(rest);
        TurnEvent.Discard discard = discards[seat * Tile.NUM_TILE_TYPES + tile];
        if (WinLegality.canRon(state, player, discard, scores, shapes)) {
          danger |= 1L << tile;
          OracleCodec.writePoints(draft, 0, dangers++, ronPoints(state, player, seat, discard));
        }
      }
      OracleCodec.writeOpponent(draft, 0, opponent, shanten, waits, danger);
    }
    return Arrays.copyOf(draft, OracleCodec.longs(draft, 0));
  }

  /** 席 {@code winner} が {@code discard} をロンしたときの和了の点（裏ドラ・本場・供託なし）。 */
  private int ronPoints(GameState state, int winner, int seat, TurnEvent.Discard discard) {
    Hand hand = state.hand(winner);
    boolean scored =
        scores.scoreAfterAdding(
            hand,
            discard.tileType(),
            WinMethod.RON,
            WinConditions.fromGameState(state, winner, discard),
            state.doraState(),
            false,
            hand.ownedAkaMask(),
            score);
    if (!scored) {
      throw new IllegalStateException("legal ron of seat " + winner + " has no yaku");
    }
    WinPointDeltas.settleInto(score.basePoints(), winner, seat, state.getOya(), 0, 0, deltas);
    return deltas[winner];
  }
}
