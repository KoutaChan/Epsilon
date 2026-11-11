package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.util.PairList;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 4 席それぞれから読んだ牌の文脈を、牌の種類ごとに 1 つにまとめる残差の層。席の軸を特徴の軸へ並べて [行, 34, 4 × 文脈] にし、
 * [残差 | 隠れ] = Linear(入力)、出力 = 残差 ＋ Linear(SiLU(隠れ)) を求める。後の射影は重みを 0 で初期化し、バイアスを持たない。
 */
final class PlayerTileMixer extends AbstractBlock {

  private final int context;
  private final Linear skipAndHiddenProjection;
  private final Linear residualProjection;

  PlayerTileMixer(int context) {
    this.context = context;
    skipAndHiddenProjection =
        addChildBlock("skipAndHiddenProjection", Linear.builder().setUnits(2L * context).build());
    Linear residual = Linear.builder().setUnits(context).optBias(false).build();
    residual.getDirectParameters().get("weight").setInitializer(new ConstantInitializer(0f));
    residualProjection = addChildBlock("residualProjection", residual);
  }

  /**
   * @param byPlayer 席ごとに読んだ牌の文脈 [行, 4, 34, 文脈]
   * @return 4 席を合わせた牌の文脈 [行, 34, 文脈]
   */
  NDArray mix(Pass pass, NDArray byPlayer) {
    long rows = byPlayer.getShape().get(0);
    NDArray input = byPlayer.swapAxes(1, 2).reshape(rows, Tiles.KINDS, (long) ClassicSchema.SEATS * context);
    NDArray combined = pass.linear(skipAndHiddenProjection, input);
    NDArray skip = combined.get("..., 0:{}", context);
    NDArray hidden = combined.get("..., {}:", context);
    return skip.add(pass.linear(residualProjection, Pass.silu(hidden)));
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    skipAndHiddenProjection.initialize(
        manager, dataType, new Shape(-1, Tiles.KINDS, (long) ClassicSchema.SEATS * context));
    residualProjection.initialize(manager, dataType, new Shape(-1, Tiles.KINDS, context));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use mix");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, Tiles.KINDS, context)};
  }
}
