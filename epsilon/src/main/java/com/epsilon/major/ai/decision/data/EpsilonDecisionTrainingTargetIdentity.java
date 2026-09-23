package com.epsilon.major.ai.decision.data;

/**
 * Decision の学習データに用いた教師値の計算方式と係数を識別する。
 *
 * <p>現行規約は直後の局の GRP 予測で局内トレースを閉じ、Value と Actor の遷移列を分けて、現在の行動の補正係数を
 * 局所 TD 誤差と後続トレースに適用するスカラー V-trace を受け付ける。
 *
 * @param targetProtocolVersion 教師値構築規則を識別する規約バージョン
 * @param causalTraceLambda 価値計算用と方策更新用の二つの遷移列のスカラー値のトレース係数
 * @param explorationCreditMix 探索行動へ残す選択行動学習への寄与の線形混合率
 */
public record EpsilonDecisionTrainingTargetIdentity(
    int targetProtocolVersion, float causalTraceLambda, float explorationCreditMix) {

  /** 直後の局の GRP 予測で閉じる、重なり方策に対するスカラー V-trace 教師値規約。 */
  public static final int NEXT_BOUNDARY_GRP_SCALAR_OVERLAP_VTRACE_V8 = 8;

  /** 規約バージョンと二つの実効係数を検証する。 */
  public EpsilonDecisionTrainingTargetIdentity {
    if (targetProtocolVersion != NEXT_BOUNDARY_GRP_SCALAR_OVERLAP_VTRACE_V8) {
      throw new IllegalArgumentException(
          "Unsupported Decision training target protocol: " + targetProtocolVersion);
    }
    if (!Float.isFinite(causalTraceLambda)
        || causalTraceLambda < 0.0f
        || causalTraceLambda > 1.0f) {
      throw new IllegalArgumentException(
          "causalTraceLambda must be finite and in [0,1]: " + causalTraceLambda);
    }
    if (!Float.isFinite(explorationCreditMix)
        || explorationCreditMix <= 0.0f
        || explorationCreditMix > 1.0f) {
      throw new IllegalArgumentException(
          "explorationCreditMix must be finite and in (0,1]: " + explorationCreditMix);
    }
  }

  /**
   * 現行選択行動の方策勾配学習教師値規約の識別情報を作る。
   *
   * @param causalTraceLambda 価値/方策価値計算用と方策更新用の二つの遷移列に使う実効lambda
   * @param explorationCreditMix 重なり方策に使う探索分の学習への寄与混合率
   * @return 学習データ片ヘッダーへ保存する教師値識別情報
   */
  public static EpsilonDecisionTrainingTargetIdentity selectedPg(
      float causalTraceLambda, float explorationCreditMix) {
    return new EpsilonDecisionTrainingTargetIdentity(
        NEXT_BOUNDARY_GRP_SCALAR_OVERLAP_VTRACE_V8, causalTraceLambda, explorationCreditMix);
  }
}
