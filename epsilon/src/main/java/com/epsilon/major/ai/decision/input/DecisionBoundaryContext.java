package com.epsilon.major.ai.decision.input;

import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import java.util.Arrays;

/**
 * Decision の価値予測だけが参照する、配牌前に確定した公開情報と GRP の予測分布。
 *
 * <p>方策入力とは別テンソルに保持し、方策がGRP 教師モデルの出力を特徴量として利用することを防ぐ。先頭は絶対席×順位の席優先 4x4
 * 周辺分布、後半は配牌前に確定している局番号・本場・供託・4家得点と、その有効マスクである。手牌、山、配牌乱数シードはこの型へ入れられない。
 */
public record DecisionBoundaryContext(
    float[] rankProbabilities, float[] publicFeatures, boolean present) {

  /** 4席×4順位のGRP 周辺分布幅。 */
  public static final int GRP_RANK_SIZE = EpsilonGrpRanks.MATRIX_SIZE;

  /** 配牌前に公開済みの局進行・得点特徴量幅。 */
  public static final int PUBLIC_FEATURE_SIZE = EpsilonGrpFeature.FEATURE_SIZE;

  /** GRP 周辺分布の先頭オフセット。 */
  public static final int GRP_RANK_OFFSET = 0;

  /** 公開局境界特徴量の先頭オフセット。 */
  public static final int PUBLIC_FEATURE_OFFSET = GRP_RANK_OFFSET + GRP_RANK_SIZE;

  /** 公開局境界が利用可能かを示すマスクインデックス。 */
  public static final int PRESENT_INDEX = PUBLIC_FEATURE_OFFSET + PUBLIC_FEATURE_SIZE;

  /** ネットワークへ渡す一次元のテンソル幅。 */
  public static final int INPUT_SIZE = PRESENT_INDEX + 1;

  private static final DecisionBoundaryContext UNIFORM = createUniform();

  /** 形状、有限性、4席×4順位の周辺制約を検証し、入力配列を独立所有する。 */
  public DecisionBoundaryContext {
    rankProbabilities =
        EpsilonGrpRanks.requirePositiveMarginals(rankProbabilities, "GRP boundary marginals");
    if (publicFeatures == null || publicFeatures.length != PUBLIC_FEATURE_SIZE) {
      throw new IllegalArgumentException(
          "boundary public feature length must be " + PUBLIC_FEATURE_SIZE);
    }
    publicFeatures = publicFeatures.clone();
    for (float value : publicFeatures) {
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException("boundary public feature must be finite");
      }
    }
  }

  /** GRPが利用できない方策のみ経路向けの無情報事前予測を作る。 */
  public static DecisionBoundaryContext uniform() {
    return UNIFORM;
  }

  private static DecisionBoundaryContext createUniform() {
    float[] probabilities = new float[GRP_RANK_SIZE];
    Arrays.fill(probabilities, 1.0f / EpsilonGrpRanks.RANK_COUNT);
    return new DecisionBoundaryContext(probabilities, new float[PUBLIC_FEATURE_SIZE], false);
  }

  /** 公開特徴量がない互換代替処理用に、重みを固定したGRPの境界周辺分布だけを保持する。 */
  public static DecisionBoundaryContext fromGrpPrior(float[] rankProbabilities) {
    return new DecisionBoundaryContext(rankProbabilities, new float[PUBLIC_FEATURE_SIZE], false);
  }

  /** 重みを固定したGRPと、その推論に使った配牌前の公開局履歴から価値事前予測入力を作る。 */
  public static DecisionBoundaryContext fromGrpBoundary(
      float[] grpFeatureSequence, float[] rankProbabilities) {
    int steps = EpsilonGrpFeature.steps(grpFeatureSequence);
    if (steps <= 0) {
      throw new IllegalArgumentException("boundary requires a non-empty GRP feature prefix");
    }
    int offset = (steps - 1) * EpsilonGrpFeature.FEATURE_SIZE;
    float[] publicFeatures =
        Arrays.copyOfRange(grpFeatureSequence, offset, offset + EpsilonGrpFeature.FEATURE_SIZE);
    return new DecisionBoundaryContext(rankProbabilities, publicFeatures, true);
  }

  @Override
  public float[] rankProbabilities() {
    return rankProbabilities.clone();
  }

  @Override
  public float[] publicFeatures() {
    return publicFeatures.clone();
  }

  /** 標準形式のネットワークテンソル順へ連結する。 */
  public float[] toNetworkInput() {
    float[] input = new float[INPUT_SIZE];
    copyTo(input, 0);
    return input;
  }

  /** 標準形式のテンソル記憶領域へ中間配列を作らず直接書き込む。 */
  void copyTo(float[] destination, int offset) {
    System.arraycopy(rankProbabilities, 0, destination, offset + GRP_RANK_OFFSET, GRP_RANK_SIZE);
    System.arraycopy(
        publicFeatures, 0, destination, offset + PUBLIC_FEATURE_OFFSET, PUBLIC_FEATURE_SIZE);
    destination[offset + PRESENT_INDEX] = present ? 1.0f : 0.0f;
  }
}
