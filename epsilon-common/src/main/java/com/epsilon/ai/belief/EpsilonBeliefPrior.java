package com.epsilon.ai.belief;

/**
 * 他家の非公開状態を推定した出力値。手牌のロジットは、牌の残数に掛ける重みの対数として使う。
 *
 * <p>相対席1〜3を配列の他家インデックス0〜2に対応させる。牌山の分布は保持しない。
 *
 * @param opponentHandLogits 他家ごと・牌種ごとの手牌の重みの対数。抽選の重みは残数とロジットの指数関数の積
 * @param opponentWaitLogits 他家ごと・牌種ごとの受け入れ牌のロジット。テンパイ時は待ち牌に対応する
 * @param opponentScalars 他家ごとのシャンテン数の予測値とテンパイのロジット
 */
public record EpsilonBeliefPrior(
    float[] opponentHandLogits, float[] opponentWaitLogits, float[] opponentScalars) {

  /** 新規生成した出力配列の所有権を受け取り、長さと有限性を検証する。 */
  public EpsilonBeliefPrior {
    require(opponentHandLogits, EpsilonBeliefLayout.OPPONENT_HAND_SIZE, "opponentHandLogits");
    require(opponentWaitLogits, EpsilonBeliefLayout.OPPONENT_WAIT_SIZE, "opponentWaitLogits");
    require(opponentScalars, EpsilonBeliefLayout.SCALAR_SIZE, "opponentScalars");
  }

  /**
   * 手牌・受け入れ牌・テンパイのロジットを0、シャンテン数の予測値を3とした比較基準を作る。
   *
   * @return 全ロジットが0で、シャンテン数の予測値が3の出力。テンパイ確率は0.5
   */
  public static EpsilonBeliefPrior uniform() {
    return new EpsilonBeliefPrior(
        new float[EpsilonBeliefLayout.OPPONENT_HAND_SIZE],
        new float[EpsilonBeliefLayout.OPPONENT_WAIT_SIZE],
        defaultOpponentScalars());
  }

  /**
   * 教師データから比較用の出力値を作る。
   *
   * <p>手牌枚数に小さい正値を加えて対数を取り、受け入れ牌マスクにも下限を設けて対数を取る。シャンテン数はそのまま使い、テンパイ判定は確率の上下限を設けて対数オッズに変換する。受け入れ牌の出力は対数オッズではない。
   *
   * @param target 完全情報局面から構築した教師
   * @return 教師値から作った参照用の出力
   */
  public static EpsilonBeliefPrior fromTarget(EpsilonBeliefTarget target) {
    return new EpsilonBeliefPrior(
        logCounts(target.opponentHandCounts()),
        logProbabilities(target.opponentWaitMask()),
        targetScalars(target));
  }

  /**
   * バッチ出力の指定行を取り出し、手牌・受け入れ牌・スカラー値の独立した配列を作る。
   *
   * @param beliefOutput Beliefの予測値を行順に格納した配列
   * @param offset 行の開始位置
   * @return 各区分の配列を独立して所有する出力
   */
  public static EpsilonBeliefPrior fromFlatOutput(float[] beliefOutput, int offset) {
    float[] handLogits = new float[EpsilonBeliefLayout.OPPONENT_HAND_SIZE];
    System.arraycopy(beliefOutput, offset, handLogits, 0, handLogits.length);
    offset += handLogits.length;
    float[] waitLogits = new float[EpsilonBeliefLayout.OPPONENT_WAIT_SIZE];
    System.arraycopy(beliefOutput, offset, waitLogits, 0, waitLogits.length);
    offset += waitLogits.length;
    float[] scalars = new float[EpsilonBeliefLayout.SCALAR_SIZE];
    System.arraycopy(beliefOutput, offset, scalars, 0, scalars.length);
    return new EpsilonBeliefPrior(handLogits, waitLogits, scalars);
  }

  /**
   * 他家別スカラーを返す。
   *
   * @param opponent 相対席1-3を0-2へ詰めたインデックス
   * @param metric スカラー区分内の指標インデックス
   * @return シャンテン実数値またはテンパイロジット
   */
  public float opponentScalar(int opponent, int metric) {
    return opponentScalars[opponent * EpsilonBeliefLayout.OPPONENT_SCALAR_COUNT + metric];
  }

  private static float[] logCounts(float[] counts) {
    float[] logits = new float[counts.length];
    for (int i = 0; i < counts.length; i++) {
      logits[i] = (float) Math.log(Math.max(1.0e-3f, counts[i] + 1.0e-3f));
    }
    return logits;
  }

  private static float[] logProbabilities(float[] probabilities) {
    float[] logits = new float[probabilities.length];
    for (int i = 0; i < probabilities.length; i++) {
      logits[i] = (float) Math.log(Math.max(1.0e-4f, probabilities[i]));
    }
    return logits;
  }

  private static float[] targetScalars(EpsilonBeliefTarget target) {
    float[] scalars = new float[EpsilonBeliefLayout.SCALAR_SIZE];
    for (int i = 0; i < EpsilonBeliefLayout.OPPONENT_COUNT; i++) {
      int base = i * EpsilonBeliefLayout.OPPONENT_SCALAR_COUNT;
      scalars[base + EpsilonBeliefLayout.SCALAR_SHANTEN] = target.opponentShanten()[i];
      scalars[base + EpsilonBeliefLayout.SCALAR_TENPAI] = logit(target.opponentTenpai()[i]);
    }
    return scalars;
  }

  private static float[] defaultOpponentScalars() {
    float[] scalars = new float[EpsilonBeliefLayout.SCALAR_SIZE];
    for (int i = 0; i < EpsilonBeliefLayout.OPPONENT_COUNT; i++) {
      scalars[i * EpsilonBeliefLayout.OPPONENT_SCALAR_COUNT + EpsilonBeliefLayout.SCALAR_SHANTEN] =
          3.0f;
    }
    return scalars;
  }

  private static float logit(float probability) {
    float p = Math.max(1.0e-4f, Math.min(1.0f - 1.0e-4f, probability));
    return (float) Math.log(p / (1.0f - p));
  }

  private static void require(float[] values, int expected, String label) {
    if (values.length != expected) {
      throw new IllegalArgumentException(
          label + " length must be " + expected + ", got " + values.length);
    }
    for (float value : values) {
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException(label + " contains non-finite value");
      }
    }
  }
}
