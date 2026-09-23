package com.epsilon.major.ai.decision.data;

import com.epsilon.ai.decision.DecisionBranchTarget;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import com.epsilon.core.DecisionLearningRole;
import com.epsilon.major.ai.decision.EpsilonDecisionConstants;
import com.epsilon.major.ai.decision.EpsilonDecisionReturns;
import com.epsilon.major.ai.decision.EpsilonUtilityTargets;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.ai.decision.training.EpsilonDecisionPretrainTargets;

/**
 * 一回の判断で得た入力・選択行動・確率分布と、その判断に対する教師値を保持する。
 *
 * <p>{@code valueTarget}は直後の局の GRP 予測で閉じるスカラー V-trace、{@code advantage}は行動価値教師値と
 * 対局生成予測の効用差である。探索後と探索前の方策は合法手候補の位置順の正規化済み確率である。
 *
 * @param input 1 判断行の型付きホスト入力
 * @param chosenLegalSlot 動的合法候補列で実際に選択した格納位置
 * @param chosenActionId 選択格納位置に対応する安定行動 ID
 * @param behaviorLogProb 探索を含む探索適用後の方策での選択行動対数確率
 * @param valueTarget 期待効用のスカラー教師
 * @param advantage 選択行動のスカラー効用アドバンテージ
 * @param finalRank 終局時の自家順位。0が1着、3が4着
 * @param behaviorPolicy 探索適用後の合法候補確率分布
 * @param rolloutPolicy 探索適用前のモデル合法候補確率分布
 * @param actorSnapshotId サンプルを生成した学習側モデルのスナップショット ID
 * @param ruleProfile 対局で使用した効用の定義インデックス
 * @param gameId サンプルを生成した対局 ID
 * @param grpFeatureSequence 局境界で固定するGRP 入力に使う事前予測の特徴量列
 * @param grpFinalRanksCode GRP 用に符号化した終局順位
 * @param learningRole CAUSAL/FORCED/PREEMPTEDの明示的な学習役割
 * @param branchTarget 終了ゲートの分岐比較教師値
 */
public record EpsilonDecisionSample(
    DecisionHostBatch input,
    int chosenLegalSlot,
    int chosenActionId,
    float behaviorLogProb,
    float valueTarget,
    float advantage,
    int finalRank,
    float[] behaviorPolicy,
    float[] rolloutPolicy,
    long actorSnapshotId,
    int ruleProfile,
    long gameId,
    int boundaryIndex,
    int seatDecisionOrdinal,
    float[] grpFeatureSequence,
    int grpFinalRanksCode,
    DecisionLearningRole learningRole,
    DecisionBranchTarget branchTarget)
    implements EpsilonDecisionSampleRecord {

  private static final float POLICY_SUM_EPS = 1.0e-4f;

  /**
   * 完了牌譜の一行を、事前学習の開始用教師値生成前の事前学習サンプルとして作る。
   *
   * <p>選択方策はone-hot、Decision 価値は終局効用、席通し番号は未割当である。{@link
   * EpsilonDecisionPretrainTargets#prepare}がゲーム全体を見て局境界を一意化し、標準形式の教師値へ置換する。
   */
  public static EpsilonDecisionSample fromPretrainingLog(
      DecisionHostBatch input,
      int chosenLegalSlot,
      int chosenActionId,
      int finalRank,
      int ruleProfile,
      long gameId,
      float[] grpFeatureSequence,
      int grpFinalRanksCode) {
    int legalActionCount = input.legalActionCount(0);
    float[] selectedPolicy = oneHotPolicyForSlot(legalActionCount, chosenLegalSlot);
    return new EpsilonDecisionSample(
        input,
        chosenLegalSlot,
        chosenActionId,
        0.0f,
        EpsilonUtilityTargets.profile(ruleProfile).utilityForRank(finalRank),
        0.0f,
        finalRank,
        selectedPolicy,
        selectedPolicy,
        0L,
        ruleProfile,
        gameId,
        Math.max(0, EpsilonGrpFeature.steps(grpFeatureSequence) - 1),
        -1,
        grpFeatureSequence,
        grpFinalRanksCode,
        legalActionCount == 1 ? DecisionLearningRole.FORCED : DecisionLearningRole.CAUSAL,
        DecisionBranchTarget.NONE);
  }

  /** 形状、識別情報、確率分布、教師値のサンプル契約を検証し、不整合があれば直ちに例外を送出する。 */
  public EpsilonDecisionSample {
    if (input.size() != 1 || input.hasTrainingTargets()) {
      throw new IllegalArgumentException("sample input must be one inference row");
    }
    if (chosenLegalSlot < 0 || chosenLegalSlot >= input.legalActionCount(0)) {
      throw new IllegalArgumentException("chosenLegalSlot outside legal action range");
    }
    if (chosenActionId != input.legalActionId(0, chosenLegalSlot)) {
      throw new IllegalArgumentException("chosenActionId does not match chosenLegalSlot");
    }
    if (finalRank < 0 || finalRank >= EpsilonDecisionConstants.PLAYERS) {
      throw new IllegalArgumentException("finalRank must be 0-3");
    }
    if (ruleProfile < 0 || ruleProfile >= EpsilonDecisionConstants.UTILITY_PROFILE_COUNT) {
      throw new IllegalArgumentException("ruleProfile outside utility profile range");
    }
    EpsilonDecisionDataChecks.requireFinite(behaviorLogProb, "behaviorLogProb");
    EpsilonDecisionReturns.requireValueTarget(
        valueTarget, EpsilonUtilityTargets.profile(ruleProfile));
    EpsilonDecisionDataChecks.requireFinite(advantage, "advantage");
    if (boundaryIndex < 0) {
      throw new IllegalArgumentException("boundaryIndex must be non-negative");
    }
    if (seatDecisionOrdinal < -1) {
      throw new IllegalArgumentException("seatDecisionOrdinal must be -1 or non-negative");
    }
    int legalCount = input.legalActionCount(0);
    if (learningRole == null) {
      throw new IllegalArgumentException("learningRole must not be null");
    }
    if (learningRole == DecisionLearningRole.FORCED && legalCount != 1) {
      throw new IllegalArgumentException("FORCED sample must have exactly one legal action");
    }
    if (learningRole == DecisionLearningRole.CAUSAL && legalCount == 1) {
      throw new IllegalArgumentException("single-action sample must be FORCED or PREEMPTED");
    }
    behaviorPolicy =
        requireNormalizedSlotPolicy(
            behaviorPolicy, legalCount, "behaviorPolicy", actorSnapshotId > 0L);
    float selectedBehaviorProb = behaviorPolicy[chosenLegalSlot];
    if (!(selectedBehaviorProb > 0.0f)) {
      throw new EpsilonDecisionDataException("selected behavior probability must be positive");
    }
    float expectedBehaviorLogProb = (float) Math.log(selectedBehaviorProb);
    if (Math.abs(behaviorLogProb - expectedBehaviorLogProb) > 1.0e-4f) {
      throw new EpsilonDecisionDataException(
          "behaviorLogProb does not match selected behaviorPolicy: logged="
              + behaviorLogProb
              + " expected="
              + expectedBehaviorLogProb);
    }
    rolloutPolicy = requireNormalizedSlotPolicy(rolloutPolicy, legalCount, "rolloutPolicy");
    grpFeatureSequence = fixedGrpFeatureSequence(grpFeatureSequence);
    grpFinalRanksCode = requireGrpFinalRanksCode(grpFinalRanksCode);
    if (EpsilonGrpRanks.decode(grpFinalRanksCode)[input.playerSeat(0)] != finalRank) {
      throw new IllegalArgumentException("finalRank does not match grpFinalRanksCode");
    }
  }

  @Override
  public EpsilonDecisionSample withTrainingTargets(float nextValueTarget, float nextAdvantage) {
    return new EpsilonDecisionSample(
        input,
        chosenLegalSlot,
        chosenActionId,
        behaviorLogProb,
        nextValueTarget,
        nextAdvantage,
        finalRank,
        behaviorPolicy,
        rolloutPolicy,
        actorSnapshotId,
        ruleProfile,
        gameId,
        boundaryIndex,
        seatDecisionOrdinal,
        grpFeatureSequence,
        grpFinalRanksCode,
        learningRole,
        branchTarget);
  }

  /** 固定調査処理で使う参照方策だけを置き換え、収集時の探索後の方策と学習教師値は保持する。 */
  EpsilonDecisionSample withRolloutPolicy(float[] nextRolloutPolicy) {
    return new EpsilonDecisionSample(
        input,
        chosenLegalSlot,
        chosenActionId,
        behaviorLogProb,
        valueTarget,
        advantage,
        finalRank,
        behaviorPolicy,
        nextRolloutPolicy,
        actorSnapshotId,
        ruleProfile,
        gameId,
        boundaryIndex,
        seatDecisionOrdinal,
        grpFeatureSequence,
        grpFinalRanksCode,
        learningRole,
        branchTarget);
  }

  @Override
  public EpsilonDecisionSample materialize() {
    return this;
  }

  @Override
  public float[] grpFeatureSequence() {
    return grpFeatureSequence.clone();
  }

  /** 教師値／監査向け共有読み取り専用のビュー。配列を変更してはならない。 */
  public float[] grpFeatureSequenceView() {
    return grpFeatureSequence;
  }

  /**
   * 合法手候補の位置順の one-hot 方策を作る。
   *
   * @param legalActionCount 動的合法候補数
   * @param chosenLegalSlot 1にするコンパクト合法格納位置
   * @return 長さ {@code legalActionCount} の one-hot 配列
   */
  public static float[] oneHotPolicyForSlot(int legalActionCount, int chosenLegalSlot) {
    if (legalActionCount <= 0 || chosenLegalSlot < 0 || chosenLegalSlot >= legalActionCount) {
      throw new IllegalArgumentException("chosenLegalSlot outside legal action range");
    }
    float[] policy = new float[legalActionCount];
    policy[chosenLegalSlot] = 1.0f;
    return policy;
  }

  static int requireGrpFinalRanksCode(int code) {
    EpsilonGrpRanks.requireCode(code);
    return code;
  }

  private static float[] fixedGrpFeatureSequence(float[] values) {
    if (values == null
        || values.length == 0
        || values.length % EpsilonGrpFeature.FEATURE_SIZE != 0) {
      throw new IllegalArgumentException(
          "grpFeatureSequence must contain complete non-empty boundary steps");
    }
    return EpsilonDecisionDataChecks.requireFinite(values, "grpFeatureSequence");
  }

  static float[] requireNormalizedSlotPolicy(float[] values, int width, String label) {
    return requireNormalizedSlotPolicy(values, width, label, false);
  }

  private static float[] requireNormalizedSlotPolicy(
      float[] values, int width, String label, boolean requireFullSupport) {
    return requireNormalizedNonNegativeFinite(
        values, width, label, POLICY_SUM_EPS, requireFullSupport);
  }

  private static float[] requireNormalizedNonNegativeFinite(
      float[] values, int width, String label, float sumTolerance, boolean requireFullSupport) {
    if (values.length != width) {
      throw new IllegalArgumentException(label + " length must be " + width + ": " + values.length);
    }
    float sum = 0.0f;
    for (float value : values) {
      if (!Float.isFinite(value) || value < 0.0f || (requireFullSupport && value == 0.0f)) {
        throw new EpsilonDecisionDataException(
            label
                + (requireFullSupport
                    ? " must be finite and positive: "
                    : " must be finite and non-negative: ")
                + value);
      }
      sum += value;
    }
    if (Math.abs(sum - 1.0f) > sumTolerance) {
      throw new EpsilonDecisionDataException(label + " sum must be 1: " + sum);
    }
    return values;
  }
}
