package com.epsilon.major.ai.decision.data;

import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import java.util.Arrays;

/** 各局の配牌前に取得した、重みを固定した GRP の事前予測を保持する補助データ。 */
public final class EpsilonDecisionGameBoundary {

  private final int index;
  private final float[] grpFeatureSequence;
  private final float[] grpRankProbabilities;

  /** 境界インデックス、GRPの局履歴、4x4 行和・列和が1の順位分布を検証する。 */
  public EpsilonDecisionGameBoundary(
      int index, float[] grpFeatureSequence, float[] grpRankProbabilities) {
    this(index, grpFeatureSequence, grpRankProbabilities, false);
  }

  /** 符号化・復号処理が新規生成して所有する配列を、再複製せずに引き取る。 */
  static EpsilonDecisionGameBoundary fromOwnedArrays(
      int index, float[] grpFeatureSequence, float[] grpRankProbabilities) {
    return new EpsilonDecisionGameBoundary(index, grpFeatureSequence, grpRankProbabilities, true);
  }

  private EpsilonDecisionGameBoundary(
      int index, float[] grpFeatureSequence, float[] grpRankProbabilities, boolean takeOwnership) {
    if (index < 0) {
      throw new IllegalArgumentException("boundary index must be non-negative");
    }
    if (grpFeatureSequence == null || EpsilonGrpFeature.steps(grpFeatureSequence) <= 0) {
      throw new IllegalArgumentException("boundary requires a non-empty GRP feature prefix");
    }
    this.index = index;
    this.grpFeatureSequence = takeOwnership ? grpFeatureSequence : grpFeatureSequence.clone();
    this.grpRankProbabilities =
        takeOwnership
            ? EpsilonGrpRanks.requirePositiveMarginalsInPlace(
                grpRankProbabilities, "grpRankProbabilities")
            : EpsilonGrpRanks.requirePositiveMarginals(
                grpRankProbabilities, "grpRankProbabilities");
  }

  public int index() {
    return index;
  }

  public float[] grpFeatureSequence() {
    return grpFeatureSequence.clone();
  }

  public float[] grpRankProbabilities() {
    return grpRankProbabilities.clone();
  }

  /** パッケージ内の符号化・復号処理向け。呼び出し側は配列を変更してはならない。 */
  float[] grpFeatureSequenceView() {
    return grpFeatureSequence;
  }

  /** パッケージ内の符号化・復号処理向け。呼び出し側は配列を変更してはならない。 */
  float[] grpRankProbabilitiesView() {
    return grpRankProbabilities;
  }

  /** 防御的な複製を作らずにサンプルのGRPの局履歴と照合する。 */
  boolean hasSameFeatureSequence(float[] featureSequence) {
    return Arrays.equals(grpFeatureSequence, featureSequence);
  }
}
