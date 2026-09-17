package com.epsilon.pico.ai.decision.audit;

/** 候補モデルを本番用に採用するための、独立した固定牌山による検証基準。 */
public final class EpsilonDecisionProductionAuditProtocol {

  public static final String ID = "decision-production-fixed-wall-v1";
  public static final int WALL_SEEDS = 100_000;
  public static final double ALPHA = 0.01;
  public static final double PROMOTION_MARGIN = 0.0;
  public static final double HARMFUL_MARGIN = 0.0;

  private EpsilonDecisionProductionAuditProtocol() {}

  /** 乱数シード系列と対戦評価番号から連続学習内で再利用しない監査IDを作る。 */
  public static String auditId(long seedBase, long duelSequence) {
    if (duelSequence <= 0L) {
      throw new IllegalArgumentException("duelSequence must be positive");
    }
    return ID + "-" + seedBase + "-" + duelSequence;
  }
}
