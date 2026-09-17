package com.epsilon.engine.player;

import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.calculate.shape.HandShapeCursor;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Tile;
import com.epsilon.engine.Player;
import java.util.List;

/**
 * シャンテン数最小化＋リーチ宣言型プレイヤー。
 *
 * <p>門前でシャンテン数を改善し、テンパイ時にリーチを宣言する。 鳴き（ポン・チー・大明槓）は行わず門前を維持する。 打牌はシャンテン数最小→受け入れ枚数最大の基準で選択する。
 */
public final class ShantenPlayer implements Player {

  private final HandShapeCursor cursor = new HandShapeCursor();
  private final int[] remainingTileCounts = new int[Tile.NUM_TILE_TYPES];
  private final HandShapeAnalyzer shapes = new HandShapeAnalyzer();

  /** シャンテン数と受け入れ枚数だけで打牌を選ぶベースラインプレイヤーを生成する。 */
  public ShantenPlayer() {}

  @Override
  public Action selectAction(GameState state, int playerIndex, List<Action> legalActions) {
    // 和了: ツモ・ロンは常に選択
    for (Action a : legalActions) {
      Action.Type type = a.type();
      if (type == Action.Type.TSUMO_AGARI || type == Action.Type.RON_AGARI) {
        return a;
      }
    }

    // リーチ: テンパイ時は最適打牌でリーチ宣言
    Action bestRiichi = chooseBestDahai(state, playerIndex, legalActions, Action.Type.RIICHI_DAHAI);
    if (bestRiichi != null) {
      return bestRiichi;
    }

    // 暗槓・加槓（門前維持のまま実行可能）
    for (Action a : legalActions) {
      Action.Type type = a.type();
      if (type.isTurnKan()) {
        return a;
      }
    }

    // 打牌: シャンテン数最小 → 受け入れ枚数最大
    Action bestDahai = chooseBestDahai(state, playerIndex, legalActions, Action.Type.DAHAI);
    if (bestDahai != null) {
      return bestDahai;
    }

    // パス（鳴き拒否）
    for (Action a : legalActions) {
      if (a.type() == Action.Type.PASS) {
        return a;
      }
    }

    return legalActions.getFirst();
  }

  /**
   * 指定タイプのアクションから最適な打牌を選択する。 シャンテン数が最小、同値なら受け入れ枚数が最大のアクションを返す。
   *
   * @return 最適アクション、該当なしならnull
   */
  private Action chooseBestDahai(
      GameState state, int playerIndex, List<Action> actions, Action.Type targetType) {
    Hand hand = state.hand(playerIndex);
    cursor.load(hand);
    int bestShanten = Integer.MAX_VALUE;
    int bestUkeire = -1;
    Action best = null;
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      remainingTileCounts[tile] =
          Math.max(
              0,
              Tile.TILES_PER_TYPE - state.publicState().visibleTileCount(tile) - hand.count(tile));
    }
    int previousTile = -1, previousShanten = 0, previousUkeire = 0;
    for (Action action : actions) {
      if (action.type() != targetType) continue;
      int tile = action.tileType();
      int shanten, ukeire;
      if (tile == previousTile) {
        shanten = previousShanten;
        ukeire = previousUkeire;
      } else {
        cursor.remove(tile);
        try {
          shanten = shapes.calculateMinimum(cursor);
          ukeire =
              shanten <= bestShanten
                  ? shapes.remainingImprovementTileCount(cursor, remainingTileCounts, shanten)
                  : 0;
        } finally {
          cursor.add(tile);
        }
        previousTile = tile;
        previousShanten = shanten;
        previousUkeire = ukeire;
      }
      if (shanten < bestShanten || shanten == bestShanten && ukeire > bestUkeire) {
        best = action;
        bestShanten = shanten;
        bestUkeire = ukeire;
      }
    }
    return best;
  }
}
