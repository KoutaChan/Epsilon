package com.epsilon.nano.ai.decision.data;

/**
 * Decision の学習データに用いた教師値の計算方式と係数を識別する。
 *
 * <p>現行規約は直後の次局GRPで局内トレースを閉じ、全判断の価値学習と因果関係のある行動の方策学習で遡る手順を分ける。
 * 局所TD誤差と後続トレースへ同じ選択行動の補正係数を掛け、探索後方策と探索前方策の重なりを評価する。
 *
 * @param targetProtocolVersion 教師値構築規則を識別する規約バージョン
 * @param causalTraceLambda 価値学習と方策学習で遡る際の減衰係数
 * @param explorationCreditMix 探索行動へ残す選択行動学習への寄与の線形混合率
 */
public record EpsilonDecisionTrainingTargetIdentity(
    int targetProtocolVersion, float causalTraceLambda, float explorationCreditMix) {

  /** 直後の次局GRPで閉じる、探索補正付きのスカラー価値教師値規約。 */
  public static final int NEXT_BOUNDARY_GRP_SCALAR_V_TRACE_V8 = 8;

  /** 規約バージョンと二つの実効係数を検証する。 */
  public EpsilonDecisionTrainingTargetIdentity {
    if (targetProtocolVersion != NEXT_BOUNDARY_GRP_SCALAR_V_TRACE_V8) {
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
   * @param causalTraceLambda 価値計算用と方策更新用の遷移列に使う実効トレース係数
   * @param explorationCreditMix 探索補正係数に使う探索学習への寄与混合率
   * @return 学習データファイルヘッダーへ保存する教師値識別情報
   */
  public static EpsilonDecisionTrainingTargetIdentity selectedPg(
      float causalTraceLambda, float explorationCreditMix) {
    return new EpsilonDecisionTrainingTargetIdentity(
        NEXT_BOUNDARY_GRP_SCALAR_V_TRACE_V8, causalTraceLambda, explorationCreditMix);
  }
}
