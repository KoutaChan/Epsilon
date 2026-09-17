package com.epsilon.pico.ai.decision.data;

/**
 * Decision の学習データに用いた教師値の計算方式と係数を識別する。
 *
 * <p>現行規約は直後の次局GRPで局内トレースを閉じ、価値計算用と方策更新用の判断列を分離したスカラー Retrace（重要度比を補正する収益推定）だけを受け付ける。
 *
 * @param targetProtocolVersion 教師値構築規則を識別する規約バージョン
 * @param causalTraceLambda 価値学習と方策学習で分けた判断列のスカラー値のトレース係数
 * @param explorationCreditMix 探索行動へ残す選択行動学習への寄与の線形混合率
 */
public record EpsilonDecisionTrainingTargetIdentity(
    int targetProtocolVersion, float causalTraceLambda, float explorationCreditMix) {

  /** 直後の次局GRPを初期化にし、選択行動のq-ret係数で局内を補正する教師値規約。 */
  public static final int NEXT_BOUNDARY_GRP_SCALAR_Q_RET_RETRACE_V7 = 7;

  /** 規約バージョンと二つの実効係数を検証する。 */
  public EpsilonDecisionTrainingTargetIdentity {
    if (targetProtocolVersion != NEXT_BOUNDARY_GRP_SCALAR_Q_RET_RETRACE_V7) {
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
   * @param causalTraceLambda 価値/方策価値学習と方策学習で分けた判断列に使う実効lambda
   * @param explorationCreditMix q-retに使う探索による選択の学習への寄与混合率
   * @return 対局単位の学習データヘッダーへ保存する教師値識別情報
   */
  public static EpsilonDecisionTrainingTargetIdentity selectedPg(
      float causalTraceLambda, float explorationCreditMix) {
    return new EpsilonDecisionTrainingTargetIdentity(
        NEXT_BOUNDARY_GRP_SCALAR_Q_RET_RETRACE_V7, causalTraceLambda, explorationCreditMix);
  }
}
