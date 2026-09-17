package com.epsilon.config.settings;

/**
 * 行動選択モデルの推論処理ごとに、通常のDJL演算とdjl-rocmの融合演算を選択する設定。
 *
 * <p>{@link DecisionInferenceFusionMode#EAGER}は通常演算、{@link
 * DecisionInferenceFusionMode#FUSION}は再利用する融合演算の実行計画を使う。
 * デバイス・パラメーター・精度の互換性は実行環境の構築時に検証する。候補全体のアフィン変換とその前処理を同時に融合すると処理が重複するため、その組み合わせはここで拒否する。
 *
 * @param playerMemory プレイヤーの要約・河・面子を処理するTransformer
 * @param tileRelation 34牌種の関係を処理するTransformer
 * @param strategicContext 局・手牌・4人の要約を処理するTransformer
 * @param stateReadout 方策と価値の計算に使う局面特徴の集約処理
 * @param policyAffine 候補・行動後の状態・分岐ごとのアフィン変換を加算する処理
 * @param policyIndexedAffine リーチ・鳴き・ロン・槓・九種九牌・ツモの二値分岐の計算
 * @param candidatePrefix 候補のアフィン変換に使う特徴の前処理。policyAffineの融合演算とは同時に指定できない
 * @param playerTileContext 4人の牌種別特徴を合成する残差付き多層パーセプトロン
 * @param candidateContext 行動の種類ごとに候補の文脈を集約する処理
 * @param binaryContext 鳴き・槓・九種九牌の2つの分岐の文脈を混合する処理
 * @param outputPacking 複数の出力スコアをホスト転送用のFLOAT32テンソルへ変換・連結する処理
 */
@SettingsPrefix("epsilon.decision.inference.fusion")
public record DecisionInferenceFusionSettings(
    @Setting("playerMemory") @Default("FUSION") DecisionInferenceFusionMode playerMemory,
    @Setting("tileRelation") @Default("FUSION") DecisionInferenceFusionMode tileRelation,
    @Setting("strategicContext") @Default("FUSION") DecisionInferenceFusionMode strategicContext,
    @Setting("stateReadout") @Default("FUSION") DecisionInferenceFusionMode stateReadout,
    @Setting("policyAffine") @Default("EAGER") DecisionInferenceFusionMode policyAffine,
    @Setting("policyIndexedAffine") @Default("FUSION")
        DecisionInferenceFusionMode policyIndexedAffine,
    @Setting("candidatePrefix") @Default("FUSION") DecisionInferenceFusionMode candidatePrefix,
    @Setting("playerTileContext") @Default("FUSION") DecisionInferenceFusionMode playerTileContext,
    @Setting("candidateContext") @Default("FUSION") DecisionInferenceFusionMode candidateContext,
    @Setting("binaryContext") @Default("FUSION") DecisionInferenceFusionMode binaryContext,
    @Setting("outputPacking") @Default("FUSION") DecisionInferenceFusionMode outputPacking) {

  private static final DecisionInferenceFusionSettings EAGER =
      new DecisionInferenceFusionSettings(
          DecisionInferenceFusionMode.EAGER,
          DecisionInferenceFusionMode.EAGER,
          DecisionInferenceFusionMode.EAGER,
          DecisionInferenceFusionMode.EAGER,
          DecisionInferenceFusionMode.EAGER,
          DecisionInferenceFusionMode.EAGER,
          DecisionInferenceFusionMode.EAGER,
          DecisionInferenceFusionMode.EAGER,
          DecisionInferenceFusionMode.EAGER,
          DecisionInferenceFusionMode.EAGER,
          DecisionInferenceFusionMode.EAGER);

  /** CPUおよび更新可能な借用モデルで使う、パラメーターを定数化しない基準経路。 */
  public static DecisionInferenceFusionSettings eager() {
    return EAGER;
  }

  /** 候補アフィン変換を二つのFusion 実行計画が同時に所有する不正構成を拒否する。 */
  public DecisionInferenceFusionSettings {
    if (policyAffine == DecisionInferenceFusionMode.FUSION
        && candidatePrefix == DecisionInferenceFusionMode.FUSION) {
      throw new IllegalArgumentException(
          "epsilon.decision.inference.fusion.policyAffine=FUSION cannot be combined with "
              + "epsilon.decision.inference.fusion.candidatePrefix=FUSION");
    }
  }

  /**
   * 凍結パラメーターへ関連付けするFusion 構成要素が一つでも有効かを返す。
   *
   * @return 凍結パラメーターを定数として関連付けする構成要素にFUSIONがあれば{@code true}
   */
  public boolean requiresFrozenParameters() {
    return playerMemory == DecisionInferenceFusionMode.FUSION
        || tileRelation == DecisionInferenceFusionMode.FUSION
        || strategicContext == DecisionInferenceFusionMode.FUSION
        || stateReadout == DecisionInferenceFusionMode.FUSION
        || policyAffine == DecisionInferenceFusionMode.FUSION
        || policyIndexedAffine == DecisionInferenceFusionMode.FUSION
        || candidatePrefix == DecisionInferenceFusionMode.FUSION
        || playerTileContext == DecisionInferenceFusionMode.FUSION;
  }

  /**
   * パラメーターを持たない文脈演算と出力の連結も含め、融合演算が一つでも有効かを返す。
   *
   * @return 一つ以上の構成要素がFUSIONなら{@code true}
   */
  public boolean hasFusion() {
    return requiresFrozenParameters()
        || candidateContext == DecisionInferenceFusionMode.FUSION
        || binaryContext == DecisionInferenceFusionMode.FUSION
        || outputPacking == DecisionInferenceFusionMode.FUSION;
  }
}
