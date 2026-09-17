package com.epsilon.ai.grp;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.GRU;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/** 局の進行と得点の時系列から、各席の最終順位に対応する4行4列のロジットを予測するGRUモデル。 */
public final class EpsilonGrpNetwork extends AbstractBlock {

  /** 標準構成の GRU 隠れ表現幅。 */
  public static final int DEFAULT_HIDDEN = 64;

  /** 標準構成の GRU 層数。 */
  public static final int DEFAULT_LAYERS = 2;

  private final int hiddenSize;
  private final int layers;
  private final GRU gru;
  private final Linear projection;
  private final Linear logitsHead;

  /** 標準構成の隠れ表現幅と層数で GRP ネットワークを構築する。 */
  public EpsilonGrpNetwork() {
    this(DEFAULT_HIDDEN, DEFAULT_LAYERS);
  }

  /**
   * GRU の隠れ表現の幅と層数を指定して GRP ネットワークを構築する。
   *
   * @param hiddenSize GRU と投影の隠れ表現幅
   * @param layers GRU 層数
   */
  public EpsilonGrpNetwork(int hiddenSize, int layers) {
    if (hiddenSize <= 0) {
      throw new IllegalArgumentException("hiddenSize must be positive");
    }
    if (layers <= 0) {
      throw new IllegalArgumentException("layers must be positive");
    }
    this.hiddenSize = hiddenSize;
    this.layers = layers;
    this.gru =
        addChildBlock(
            "grpGru",
            GRU.builder()
                .setStateSize(hiddenSize)
                .setNumLayers(layers)
                .optBatchFirst(true)
                .optHasBiases(true)
                .optReturnState(false)
                .build());
    this.projection = addChildBlock("projection", Linear.builder().setUnits(hiddenSize).build());
    this.logitsHead =
        addChildBlock(
            "rankMarginalLogitsHead",
            Linear.builder().setUnits(EpsilonGrpRanks.MATRIX_SIZE).build());
  }

  /** GRU と投影層の隠れ表現の幅を返す。 */
  public int hiddenSize() {
    return hiddenSize;
  }

  /** GRU の層数を返す。 */
  public int layers() {
    return layers;
  }

  /** 標準構成のチェックポイント互換性を判定するための識別子を返す。 */
  public static String architectureSummary() {
    return architectureSummary(DEFAULT_HIDDEN, DEFAULT_LAYERS);
  }

  /**
   * 指定したモデル構成の互換性識別子を返す。
   *
   * @param hidden GRU と投影層の隠れ表現の幅
   * @param layers GRU の層数
   * @return 入力幅、GRU の構成、出力形式、正規化方式を含む識別子
   */
  public static String architectureSummary(int hidden, int layers) {
    // v3: 24通りの順位順列を予測する出力層を 4x4 Sinkhorn 周辺確率出力層へ置換。
    return "grp-epsilon-v3 input=[B,T,"
        + EpsilonGrpFeature.FEATURE_SIZE
        + "] gruHidden="
        + hidden
        + " layers="
        + layers
        + " output=rankMarginal4x4 sinkhornIterations="
        + EpsilonGrpSinkhorn.ITERATIONS
        + " interactionClip="
        + EpsilonGrpSinkhorn.INTERACTION_CLIP
        + " marginalRepair=affine optimizerState=adamw-v1";
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    Shape sequenceShape =
        inputShapes.length > 0 ? inputShapes[0] : new Shape(-1, -1, EpsilonGrpFeature.FEATURE_SIZE);
    gru.initialize(manager, dataType, sequenceShape);
    Shape hiddenShape = new Shape(-1, hiddenSize);
    projection.initialize(manager, dataType, hiddenShape);
    logitsHead.initialize(manager, dataType, hiddenShape);
  }

  /**
   * 長さの異なる GRP 特徴量系列から、席と最終順位の組合せに対応するロジットを計算する。
   *
   * @param ps 順伝播に使うパラメーターの管理オブジェクト
   * @param sequence {@code [batch, time, feature]} のパディング済み系列
   * @param lengths 各入力系列の有効ステップ数
   * @param training 学習時の順伝播なら {@code true}
   * @param params DJL ブロックへ渡す実行時パラメーター
   * @return {@code [batch, 16]} の Sinkhorn 前ロジット
   */
  public NDArray forwardLogits(
      ParameterStore ps,
      NDArray sequence,
      NDArray lengths,
      boolean training,
      PairList<String, Object> params) {
    return forwardInternal(ps, new NDList(sequence, lengths), training, params).singletonOrThrow();
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
    NDArray sequence = inputs.get(0);
    NDArray lengths = inputs.size() > 1 ? inputs.get(1) : null;
    NDArray outputs = gru.forward(ps, new NDList(sequence), training, params).singletonOrThrow();
    NDArray finalState = selectLastValid(outputs, lengths);
    NDArray hidden =
        activation(
            projection.forward(ps, new NDList(finalState), training, params).singletonOrThrow());
    return new NDList(
        logitsHead.forward(ps, new NDList(hidden), training, params).singletonOrThrow());
  }

  private static NDArray selectLastValid(NDArray outputs, NDArray lengths) {
    long batch = outputs.getShape().get(0);
    long time = outputs.getShape().get(1);
    if (lengths == null) {
      return outputs.get(":, {}, :", time - 1);
    }
    NDArray timeIndex =
        outputs
            .getManager()
            .arange((int) time)
            .toType(outputs.getDataType(), false)
            .reshape(1, time);
    NDArray lastIndex =
        lengths.toType(outputs.getDataType(), false).sub(1.0f).maximum(0.0f).reshape(batch, 1);
    NDArray mask =
        timeIndex.eq(lastIndex).toType(outputs.getDataType(), false).reshape(batch, time, 1);
    return outputs.mul(mask).sum(new int[] {1});
  }

  private static NDArray activation(NDArray x) {
    return Activation.relu(x);
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    long batch = inputShapes[0].get(0);
    return new Shape[] {new Shape(batch, EpsilonGrpRanks.MATRIX_SIZE)};
  }
}
