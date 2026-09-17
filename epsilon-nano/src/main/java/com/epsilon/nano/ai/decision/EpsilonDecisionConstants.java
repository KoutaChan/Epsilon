package com.epsilon.nano.ai.decision;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.nano.ai.model.EpsilonMahjongStateEncoder;
import com.epsilon.nano.ai.model.EpsilonTileRelationEncoder;

/** Decision モデルの構造を識別する定数と出力次元を定義する。 */
public final class EpsilonDecisionConstants {

  private EpsilonDecisionConstants() {}

  /** 重みを固定したGRP 効用事前予測とHL-Gauss スカラー価値を持つ現行構造。 */
  public static final String ARCHITECTURE_ID =
      "mahjong-entity-action-transition-policy-graph-shared-discard-"
          + "direct-route-sparse-wait-canonical-tile-player-memory1-rel2h4-key-value-"
          + EpsilonMahjongStateEncoder.STRATEGIC_CONTEXT_FINGERPRINT
          + "-shared-policy-memory-policy-c64-f384-value-c64-f256-detached-value-"
          + "candidate-context-single-player-tile-context-residual-c64-hl-gauss-utility-value-v45-"
          + DecisionInputSchema.fingerprint()
          + "-"
          + EpsilonTileRelationEncoder.FINGERPRINT;

  /** 4人麻雀の固定プレイヤー数。 */
  public static final int PLAYERS = GameState.NUM_PLAYERS;

  /** 絶対席×最終順位の4x4分布幅。 */
  public static final int PLAYERS_SQUARED = PLAYERS * PLAYERS;

  /** 萬子・筒子・索子・字牌を合わせた牌種数。 */
  public static final int TILE_TYPES = Tile.NUM_TILE_TYPES;

  /** チェックポイントが出力する効用の定義数。 */
  public static final int UTILITY_PROFILE_COUNT = EpsilonUtilityProfile.values().length;
}
