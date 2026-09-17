package com.epsilon.nano.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.nano.ai.model.EpsilonMahjongStateEncoder;

/**
 * 通常の打牌と鳴き直後の打牌を、同じ特徴量・パラメーター・尺度で採点する。
 *
 * <p>入力は最初の行動の種類ではなく、完全に投影済みの打牌遷移表現である。したがって同じ物理打牌候補を通常手番で比較するときも、
 * チー・ポン後の合法打牌集合を比較するときも、同じ関数と同じ尺度を使う。最初の行動の効用は別スコア計算処理が求め、このブロックのロジットは条件付き マスク付きソフトマックスの重みにだけ使われる。
 */
public final class EpsilonDiscardCandidateScorer extends AbstractBlock {

  private final int hiddenSize;
  private final Linear hiddenProjection;
  private final Linear scoreHead;

  /**
   * 指定した隠れ層幅の共有打牌スコア計算処理を構築する。
   *
   * @param hiddenSize 遷移表現と中間射影の幅
   */
  public EpsilonDiscardCandidateScorer(int hiddenSize) {
    this.hiddenSize = hiddenSize;
    hiddenProjection =
        addChildBlock("hiddenProjection", Linear.builder().setUnits(hiddenSize).build());
    scoreHead = addChildBlock("scoreHead", Linear.builder().setUnits(1).optBias(false).build());
  }

  /**
   * 任意の先行形状を保ったまま、末尾隠れ層次元をスカラーロジットへ写す。
   *
   * @param parameterStore 順伝播に使うパラメーター保存先
   * @param transitionEmbeddings 採点する投影済み打牌遷移表現
   * @param training 学習時の順伝播なら {@code true}
   * @param runtimeParameters DJL ブロックへ渡す実行時パラメーター
   * @return 入力の末尾次元だけを 1 に置き換えた条件付き打牌ロジット
   */
  public NDArray score(
      ParameterStore parameterStore,
      NDArray transitionEmbeddings,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    NDArray hidden =
        hiddenProjection
            .forward(parameterStore, new NDList(transitionEmbeddings), training, runtimeParameters)
            .singletonOrThrow();
    return scoreHead
        .forward(
            parameterStore,
            new NDList(EpsilonMahjongStateEncoder.silu(hidden)),
            training,
            runtimeParameters)
        .singletonOrThrow();
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    hiddenProjection.initialize(manager, dataType, new Shape(-1, hiddenSize));
    scoreHead.initialize(manager, dataType, new Shape(-1, hiddenSize));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return new NDList(
        score(parameterStore, inputs.singletonOrThrow(), training, runtimeParameters));
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    Shape input = inputShapes[0];
    long[] dimensions = input.getShape().clone();
    dimensions[dimensions.length - 1] = 1;
    return new Shape[] {new Shape(dimensions)};
  }
}
