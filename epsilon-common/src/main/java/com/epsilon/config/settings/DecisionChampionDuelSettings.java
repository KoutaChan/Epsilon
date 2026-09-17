package com.epsilon.config.settings;

/**
 * 学習区間の終了時に、現在採用している固定モデルと比較対局を行うための設定。
 *
 * <p>事前に決めた2回の判定時点に有意水準を配分する。効用差の信頼区間の下限が0を超えれば採用、上限が0未満なら不採用とし、それ以外は判定を保留する。
 * 本番モデルへの採用を判断する独立した検証は、この比較対局とは別に実施する。
 *
 * @param interimWallSeeds 最初の判定までに使用する牌山数
 * @param maximumWallSeeds 最終判定までに使用する最大牌山数
 * @param alpha 比較対局全体に割り当てる両側の有意水準
 * @param interimAlphaFraction 最初の判定に割り当てる有意水準の割合
 */
@SettingsPrefix("epsilon.decision.championDuel")
public record DecisionChampionDuelSettings(
    @Setting("interimWallSeeds") @Default("50000") @Positive int interimWallSeeds,
    @Setting("maximumWallSeeds") @Default("100000") @Positive int maximumWallSeeds,
    @Setting("alpha")
        @Default("0.01")
        @Range(min = 0.0, minInclusive = false, max = 1.0, maxInclusive = false)
        double alpha,
    @Setting("interimAlphaFraction")
        @Default("0.25")
        @Range(min = 0.0, minInclusive = false, max = 1.0, maxInclusive = false)
        double interimAlphaFraction) {

  /** 二段階比較対局の牌山数が正しく増加することを検証する。 */
  public DecisionChampionDuelSettings {
    if (interimWallSeeds >= maximumWallSeeds) {
      throw new IllegalArgumentException(
          "champion duel wall seeds must satisfy interimWallSeeds < maximumWallSeeds");
    }
  }

  /** 二段階比較対局の最初の判定時点へ割り当てる両側誤差率。 */
  public double interimAlpha() {
    return alpha * interimAlphaFraction;
  }

  /** 二段階比較対局の最終判定時点へ割り当てる両側誤差率。 */
  public double finalLookAlpha() {
    return alpha - interimAlpha();
  }
}
