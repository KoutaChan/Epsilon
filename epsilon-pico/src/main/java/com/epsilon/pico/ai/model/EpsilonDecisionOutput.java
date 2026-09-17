package com.epsilon.pico.ai.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import com.epsilon.pico.ai.decision.policy.DecisionPolicyScores;
import com.epsilon.pico.ai.decision.policy.EpsilonDecisionPolicyGraph;

/**
 * 動的合法候補に対する Decision ネットワーク出力。
 *
 * @param policyScores 条件付き方策グラフが消費する未正規化スコア
 * @param valueLogits 自家の効用-区間ロジット [バッチ,101]。順位確率ではなく区間 center の期待値へ復号する
 */
public record EpsilonDecisionOutput(DecisionPolicyScores policyScores, NDArray valueLogits) {

  /** 汎用 {@link NDList} 表現に含むテンソル数。 */
  public static final int TENSOR_COUNT = DecisionPolicyScores.TENSOR_COUNT + 1;

  /**
   * 行動メタデータを使って最終的な行動の確率分布を構築する。
   *
   * @param actionCategories 行動種類・グループを持つカテゴリ値テンソル
   * @param actionRoutes 同一打牌と RIICHI・DAMA 対応を持つ参照先の対応テンソル
   * @return 合法末端の行動上の対数確率と確率
   */
  public EpsilonDecisionPolicyGraph.Distribution policyDistribution(
      NDArray actionCategories, NDArray actionRoutes) {
    return policyScores.distribution(actionCategories, actionRoutes);
  }

  /**
   * 推論向けに最終末端の行動の対数確率だけを構築する。
   *
   * @param actionCategories 行動種類・グループを持つカテゴリ値テンソル
   * @param actionRoutes 同一打牌と RIICHI・DAMA 対応を持つ参照先の対応テンソル
   * @return パディングに有限な大負値を持つ末端の行動対数確率
   */
  public NDArray policyLogProbabilities(NDArray actionCategories, NDArray actionRoutes) {
    return policyScores.logProbabilities(actionCategories, actionRoutes);
  }

  /**
   * 汎用 Block・チェックポイント API 用の固定順テンソルリストへ変換する。
   *
   * @return 方策スコア群の後ろに価値ロジットを置いたテンソルリスト
   */
  public NDList toNDList() {
    NDList arrays = policyScores.toNDList();
    arrays.add(valueLogits);
    return arrays;
  }
}
