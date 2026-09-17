package com.epsilon.pico.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.util.PairList;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;

/**
 * 同じ牌種に対する4家の局所文脈を、牌種ごとに独立して非線形に合成する残差ブロック。
 *
 * <p>入力 {@code [batch,4,34,context]} を {@code [batch,34,4*context]} へ並べ替え、最初の射影で
 * 残差接続と隠れ層を同時に作る。出力は次式の {@code [batch,34,context]} である。
 *
 * <pre>{@code
 * combined = Linear(4 * context, 2 * context, input)
 * skip     = combined[..., 0:context]
 * hidden   = combined[..., context:2 * context]
 * output   = skip + Linear(context, context, SiLU(hidden))
 * }</pre>
 *
 * <p>残差射影の重みは0で初期化する。そのため残差接続側へ既存の線形射影を移した時点では数学的に同じ出力となり、最初の
 * 逆伝播では残差射影だけが新しい勾配を受ける。隠れ層側へ勾配が流れ始めるのは、残差射影が更新された次の順伝播からである。 残差射影は残差接続側バイアスと重複するバイアスを持たない。
 */
public final class EpsilonPlayerTileContextMixer extends AbstractBlock {

  private final int contextWidth;
  private final Linear skipAndHiddenProjection;
  private final Linear residualProjection;

  EpsilonPlayerTileContextMixer(int contextWidth) {
    if (contextWidth <= 0) {
      throw new IllegalArgumentException("player tile context width must be positive");
    }
    this.contextWidth = contextWidth;
    skipAndHiddenProjection =
        addChildBlock(
            "skipAndHiddenProjection", Linear.builder().setUnits(contextWidth * 2L).build());
    Linear zeroInitializedResidual = Linear.builder().setUnits(contextWidth).optBias(false).build();
    zeroInitializedResidual
        .getDirectParameters()
        .get("weight")
        .setInitializer(new ConstantInitializer(0.0f));
    residualProjection = addChildBlock("residualProjection", zeroInitializedResidual);
  }

  /** 重みを固定したモデルの推論実行計画へ結び付ける残差接続・隠れ層同時射影を返す。 */
  public Linear skipAndHiddenProjection() {
    return skipAndHiddenProjection;
  }

  /** 重みを固定したモデルの推論実行計画へ結び付けるゼロ初期化済み残差射影を返す。 */
  public Linear residualProjection() {
    return residualProjection;
  }

  /**
   * プレイヤー軸を各牌種の末尾特徴量へ移し、4家を一つにに合成した牌別文脈を返す。
   *
   * @param parameterStore 射影パラメーターの取得先
   * @param byPlayer プレイヤー軸を保持した {@code [batch,4,34,context]} 文脈
   * @param training 学習用パラメーターを取得するなら{@code true}
   * @return 全プレイヤーを合成した {@code [batch,34,context]} 文脈
   */
  NDArray mix(ParameterStore parameterStore, NDArray byPlayer, boolean training) {
    NDArray input = flatten(byPlayer);
    if (!input.getDevice().isGpu()) {
      return mixCpu(parameterStore, input, training);
    }
    NDArray combinedWeight =
        parameterStore.getValue(
            skipAndHiddenProjection.getDirectParameters().get("weight"),
            input.getDevice(),
            training);
    NDArray combinedBias =
        parameterStore.getValue(
            skipAndHiddenProjection.getDirectParameters().get("bias"), input.getDevice(), training);
    NDArray outputWeight =
        parameterStore.getValue(
            residualProjection.getDirectParameters().get("weight"), input.getDevice(), training);
    return NDArrays.projectedResidualMlp(input, combinedWeight, combinedBias, outputWeight);
  }

  /**
   * CPU の通常 Linear は自動混合精度を適用した入力と FLOAT32 master 重みを受け付ける。 custom MLP の CPU 代替処理
   * は計算前に同データ型を要求するため、同じ式を標準演算で評価する。 パラメーター/入力の toType は使わず、元の学習グラフへの勾配を保つ。
   */
  private NDArray mixCpu(ParameterStore parameterStore, NDArray input, boolean training) {
    NDArray combined =
        skipAndHiddenProjection
            .forward(parameterStore, new NDList(input), training, new PairList<>())
            .singletonOrThrow();
    NDArray skip = combined.get("...,0:{}", contextWidth);
    NDArray hidden = combined.get("...,{}:", contextWidth);
    NDArray residual =
        residualProjection
            .forward(
                parameterStore,
                new NDList(Activation.swish(hidden, 1.0f)),
                training,
                new PairList<>())
            .singletonOrThrow();
    return skip.add(residual);
  }

  /** Fusion 実行計画へ渡すため、プレイヤー軸を同じ牌種の連続特徴量へ畳み込む。 */
  public NDArray flatten(NDArray byPlayer) {
    long rowCount = byPlayer.getShape().get(0);
    return byPlayer
        .swapAxes(1, 2)
        .reshape(rowCount, Tile.NUM_TILE_TYPES, GameState.NUM_PLAYERS * (long) contextWidth);
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    skipAndHiddenProjection.initialize(
        manager,
        dataType,
        new Shape(-1, Tile.NUM_TILE_TYPES, GameState.NUM_PLAYERS * (long) contextWidth));
    residualProjection.initialize(
        manager, dataType, new Shape(-1, Tile.NUM_TILE_TYPES, contextWidth));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return new NDList(mix(parameterStore, inputs.singletonOrThrow(), training));
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(inputShapes[0].get(0), Tile.NUM_TILE_TYPES, contextWidth)};
  }
}
