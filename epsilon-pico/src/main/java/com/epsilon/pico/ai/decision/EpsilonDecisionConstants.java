package com.epsilon.pico.ai.decision;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.pico.ai.decision.input.DecisionInputSchema;
import com.epsilon.pico.ai.model.EpsilonMahjongStateEncoder;
import com.epsilon.pico.ai.model.EpsilonTileRelationEncoder;

/** Decision 構造の識別子と出力次元を定義する。 */
public final class EpsilonDecisionConstants {

  private EpsilonDecisionConstants() {}

  /** 幅64の二択の判定と直接選択肢オフセットを持つEpsilon Pico 構造。 */
  public static final String ARCHITECTURE_ID =
      "epsilon-pico-v1-mahjong-entity-action-transition-policy-graph-shared-discard-"
          + "direct-route-sparse-wait-canonical-tile-player-memory1-rel2h4-key-value-"
          + EpsilonMahjongStateEncoder.STRATEGIC_CONTEXT_FINGERPRINT
          + "-shared-policy-memory-policy-c64-maxf384-value-c64-maxf128-detached-value-"
          + "candidate-context-single-player-tile-context-residual-c64-hl-gauss-utility-value-"
          + "binary-gates-c64-direct-alternative-offsets-"
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
