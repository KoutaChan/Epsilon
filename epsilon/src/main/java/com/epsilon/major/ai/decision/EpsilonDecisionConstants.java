package com.epsilon.major.ai.decision;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import com.epsilon.major.ai.model.EpsilonMahjongStateEncoder;
import com.epsilon.major.ai.model.EpsilonTileRelationEncoder;

/** Decision 構造の識別子と出力次元を定義する。 */
public final class EpsilonDecisionConstants {

  private EpsilonDecisionConstants() {}

  /** 公開時系列・共有K/V・独立幅二択分岐・価値残差を統合したmajor 構造。 */
  public static final String ARCHITECTURE_ID =
      "epsilon-major-v47-point-projection-point-gate6-shared-discard-public-history-shared-kv-"
          + EpsilonMahjongStateEncoder.STRATEGIC_CONTEXT_FINGERPRINT
          + "-policy-context-c128-readout-c128-f384-g256-folded-offset-transition-bridge-"
          + "value-readout-c128-f256-detached-preactivation-residual-hl101-"
          + DecisionInputSchema.fingerprint()
          + "-"
          + EpsilonTileRelationEncoder.FINGERPRINT;

  /** 4人麻雀の固定プレイヤー数。 */
  public static final int PLAYERS = GameState.NUM_PLAYERS;

  /** 絶対席×最終順位の4x4分布幅。 */
  public static final int PLAYERS_SQUARED = PLAYERS * PLAYERS;

  /** 萬子・筒子・索子・字牌を合わせた牌種数。 */
  public static final int TILE_TYPES = Tile.NUM_TILE_TYPES;

  /** チェックポイントが出力する効用定義の種類数。 */
  public static final int UTILITY_PROFILE_COUNT = EpsilonUtilityProfile.values().length;
}
