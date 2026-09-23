package com.epsilon.pico.ai.decision.data;

/**
 * Decision の学習データに用いた教師値の計算方式と係数を識別する。
 *
 * <p>現行規約は直後の次局GRPで局内トレースを閉じ、価値計算用と方策更新用の判断列を分離する。
 * 選択行動の補正係数は、探索前後の方策が重なる部分から作り、Value の局所 TD 差分と後続トレースの両方に適用する。
 *
 * @param targetProtocolVersion 教師値構築規則を識別する規約バージョン
 * @param causalTraceLambda 価値学習と方策学習で分けた判断列の Value トレース係数
 * @param explorationCreditMix 探索前後の方策差を補正する係数の混合率
 */
public record EpsilonDecisionTrainingTargetIdentity(
    int targetProtocolVersion, float causalTraceLambda, float explorationCreditMix) {

  /** 直後の次局GRPを初期化にし、補正済み Value トレースで局内を評価する教師値規約。 */
  public static final int NEXT_BOUNDARY_GRP_OVERLAP_V_TRACE_V8 = 8;

  /** 規約バージョンと二つの実効係数を検証する。 */
  public EpsilonDecisionTrainingTargetIdentity {
    if (targetProtocolVersion != NEXT_BOUNDARY_GRP_OVERLAP_V_TRACE_V8) {
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
   * @param causalTraceLambda 価値計算用と方策更新用で分けた判断列に使う実効lambda
   * @param explorationCreditMix 探索前後の方策差を補正する係数の混合率
   * @return 対局単位の学習データヘッダーへ保存する教師値識別情報
   */
  public static EpsilonDecisionTrainingTargetIdentity selectedPg(
      float causalTraceLambda, float explorationCreditMix) {
    return new EpsilonDecisionTrainingTargetIdentity(
        NEXT_BOUNDARY_GRP_OVERLAP_V_TRACE_V8, causalTraceLambda, explorationCreditMix);
  }
}
