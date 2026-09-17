package com.epsilon.pico.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

/**
 * 方策出力層が生成し、方策グラフが正規化する未正規化スコア群。
 *
 * @param alternativeScores 行動の種類のスコア。形状は {@code [batch, DecisionAlternative.NETWORK_SIZE]}
 * @param actionCandidateScores 同一行動の種類内の具体的行動、および物理牌の打牌の識別情報のスコア。形状は {@code [batch,
 *     legalActionCapacity]}
 * @param riichiGateScores 同じ物理牌の打牌に条件付けた RIICHI/DAMA 二択の判定のロジット。形状は {@code [batch,
 *     legalActionCapacity]}
 */
public record DecisionPolicyScores(
    NDArray alternativeScores, NDArray actionCandidateScores, NDArray riichiGateScores) {

  /** {@link #toNDList()}が返すテンソル数。 */
  public static final int TENSOR_COUNT = 3;

  /**
   * 指定した行動の種類のバッチスコアを返す。
   *
   * @param alternative 取得する行動の種類
   * @return 形状 {@code [batch]} のビュー
   */
  public NDArray alternative(DecisionAlternative alternative) {
    return alternativeScores.get(":,{}", alternative.networkIndex());
  }

  /**
   * 指定した合法行動候補の位置に対応する候補依存二択の判定スコアを返す。
   *
   * @param actionSlot 行動容量区分内の位置
   * @return 形状 {@code [batch]} のビュー
   */
  public NDArray riichiGate(int actionSlot) {
    return riichiGateScores.get(":,{}", actionSlot);
  }

  /**
   * encoded 行動メタデータとスコアを条件付き最終的な行動の確率分布へ合成する。
   *
   * @param actionCategories 形状 {@code [batch, action, actionField]}
   * @param actionRoutes 形状 {@code [batch, action, routeField]}
   * @return 節点別診断を含む条件付き分布
   */
  public EpsilonDecisionPolicyGraph.Distribution distribution(
      NDArray actionCategories, NDArray actionRoutes) {
    return EpsilonDecisionPolicyGraph.composeDistribution(this, actionCategories, actionRoutes);
  }

  /**
   * 推論向けに、確率テンソルを追加生成せず最終末端の行動の対数確率だけを返す。
   *
   * @param actionCategories 形状 {@code [batch, action, actionField]}
   * @param actionRoutes 形状 {@code [batch, action, routeField]}
   * @return 形状 {@code [batch, action]} の末端の行動対数確率
   */
  public NDArray logProbabilities(NDArray actionCategories, NDArray actionRoutes) {
    return EpsilonDecisionPolicyGraph.composeLogProbabilities(this, actionCategories, actionRoutes);
  }

  /**
   * チェックポイント・汎用Block APIで使う固定順のテンソルリストへ変換する。
   *
   * @return 選択肢、候補、riichi 二択の判定の順のNDList
   */
  public NDList toNDList() {
    return new NDList(alternativeScores, actionCandidateScores, riichiGateScores);
  }
}
