package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * 打牌の遷移を採点する。手番の打牌も、チー・ポンの後の打牌も、同じ射影と尺度で採点する。チー・ポンの手では、このロジットを、
 * 鳴いた後の打牌の遷移を 1 つにまとめるときの重みにだけ使う。
 */
final class DiscardScorer extends AbstractBlock {

  private final int hidden;
  private final Linear hiddenProjection;
  private final Linear scoreHead;

  DiscardScorer(int hidden) {
    this.hidden = hidden;
    hiddenProjection = addChildBlock("hiddenProjection", Linear.builder().setUnits(hidden).build());
    scoreHead = addChildBlock("scoreHead", Linear.builder().setUnits(1).optBias(false).build());
  }

  /**
   * @param transitions 遷移の表現 [..., 幅]
   * @return ロジット [..., 1]
   */
  NDArray score(Pass pass, NDArray transitions) {
    return pass.linear(scoreHead, Pass.silu(pass.linear(hiddenProjection, transitions)));
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    hiddenProjection.initialize(manager, dataType, new Shape(-1, hidden));
    scoreHead.initialize(manager, dataType, new Shape(-1, hidden));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use score");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, 1)};
  }
}
