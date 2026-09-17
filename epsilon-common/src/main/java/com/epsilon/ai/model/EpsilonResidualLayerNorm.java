package com.epsilon.ai.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.training.ParameterStore;

/** 入力を変更せずに残差を加算し、既存のパラメーターで層正規化を行う。 */
public final class EpsilonResidualLayerNorm {

  private static final float EPSILON = 1.0e-5f;

  private EpsilonResidualLayerNorm() {}

  /**
   * 残差と更新を加算し、その和と正規化結果を同じ自動微分の計算グラフ上で返す。
   *
   * <p>既存{@link LayerNorm}のgamma/betaを学習モードで取得するため、パラメーター名とチェックポイントスキーマは変わらない。
   * 入力テンソルは変更も複製もせず、返却順は{@code [normalized, summedResidual]}である。
   *
   * @param parameterStore 学習パラメーターの取得先
   * @param residual 残差本体
   * @param update 加算する更新量
   * @param layerNorm 加算後に適用する既存LayerNorm ブロック
   * @return 正規化結果と加算後残差
   */
  public static NDList applyTraining(
      ParameterStore parameterStore, NDArray residual, NDArray update, LayerNorm layerNorm) {
    NDArray gamma =
        parameterStore.getValue(layerNorm.getParameters().get("gamma"), residual.getDevice(), true);
    NDArray beta =
        parameterStore.getValue(layerNorm.getParameters().get("beta"), residual.getDevice(), true);
    return LayerNorm.residualAddLayerNorm(residual, update, gamma.getShape(), gamma, beta, EPSILON);
  }
}
