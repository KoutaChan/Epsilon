package com.epsilon.nano.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * 幅の異なる複数表現を共通出力幅へ投影し、固有形状のままブロードキャスト加算するブロック。
 *
 * <p>状態のように全候補で共通な表現は {@code [B,1,H]} または {@code [B,1,1,H]} のまま一度だけ変換する。 牌・プレイヤー
 * メモリとの通信表現は小さいコンテキスト幅のまま受け取れるため、隠れ層幅へ事前展開する行列積と一時テンソルを作らない。
 * 先頭構成要素だけがバイアスを持ち、加算後のバイアスが構成要素数に依存しないようにする。
 *
 * <p>ここでのFusionは学習モデル上の特徴合成を意味し、djl-rocmのFusion 実行計画ではない。凍結推論用のネイティブ実装は {@code
 * com.epsilon.nano.ai.decision.policy.fusion} が所有する。
 */
public final class EpsilonFeatureFusion extends AbstractBlock {

  private final int outputSize;
  private final int[] inputSizes;
  private final int fusedPrefixComponents;
  private final Linear[] projections;

  EpsilonFeatureFusion(int outputSize, int fusedPrefixComponents, int... inputSizes) {
    if (outputSize < 1
        || inputSizes.length < 2
        || fusedPrefixComponents < 1
        || fusedPrefixComponents > inputSizes.length) {
      throw new IllegalArgumentException("feature fusion requires positive widths and components");
    }
    this.outputSize = outputSize;
    this.inputSizes = inputSizes.clone();
    this.fusedPrefixComponents = fusedPrefixComponents;
    projections = new Linear[inputSizes.length];
    for (int component = 0; component < inputSizes.length; component++) {
      if (inputSizes[component] < 1) {
        throw new IllegalArgumentException("feature fusion input width must be positive");
      }
      projections[component] =
          addChildBlock(
              "component" + component,
              Linear.builder().setUnits(outputSize).optBias(component == 0).build());
    }
  }

  /** 各構成要素を宣言済み入力幅から共通幅へ投影し、ブロードキャスト可能な軸を保って加算する。 */
  NDArray fuse(
      ParameterStore parameterStore,
      boolean training,
      PairList<String, Object> runtimeParameters,
      NDArray... components) {
    if (components.length != projections.length) {
      throw new IllegalArgumentException(
          "fusion component count must be " + projections.length + ": " + components.length);
    }
    NDArray result = fusePrefix(parameterStore, components, training);
    for (int component = fusedPrefixComponents; component < projections.length; component++) {
      result.addi(
          applyLinear(
              projections[component],
              parameterStore,
              components[component],
              training,
              runtimeParameters));
    }
    return result;
  }

  /**
   * 宣言済み構成要素数を返す。
   *
   * <p>推論専用の永続実行計画が既存Linearを直接関連付けるためのパッケージ内境界であり、パラメーター自体は複製しない。
   */
  public int componentCount() {
    return projections.length;
  }

  /** 同じ候補軸を持ち、一つのGEMMへ連結できる先頭構成要素数を返す。 */
  public int fusedPrefixComponents() {
    return fusedPrefixComponents;
  }

  /** 指定構成要素の入力幅を返す。 */
  public int inputSize(int component) {
    return inputSizes[component];
  }

  /** 共通の出力幅を返す。 */
  public int outputSize() {
    return outputSize;
  }

  /** チェックポイント名を変えず既存射影を関連付けるため、指定構成要素のLinearを返す。 */
  public Linear projection(int component) {
    return projections[component];
  }

  /**
   * 同じ候補軸を持つ先頭構成要素を一つの大きなGEMMへまとめる。
   *
   * <p>個別の {@code x_i W_i^T} の和は、末尾軸で連結した入力と重みの積に厳密に書き換えられる。状態のように
   * 候補軸へブロードキャストする構成要素は先頭部分へ含めず、候補数分の不要なFLOPsを増やさない。
   */
  private NDArray fusePrefix(
      ParameterStore parameterStore, NDArray[] components, boolean training) {
    NDList inputs = new NDList(fusedPrefixComponents);
    for (int component = 0; component < fusedPrefixComponents; component++) {
      inputs.add(components[component]);
    }
    Parameter bias = projections[0].getDirectParameters().get("bias");
    // Autograd retains its saved tensors independently of these temporary handles.
    try (NDArray weight = prefixWeight(parameterStore, components[0], training);
        NDArray packed =
            training
                ? NDArrays.concat(inputs, components[0].getShape().dimension() - 1)
                : NDArrays.concatToType(
                    inputs, components[0].getShape().dimension() - 1, weight.getDataType())) {
      return Linear.linear(
              packed, weight, parameterStore.getValue(bias, components[0].getDevice(), training))
          .singletonOrThrow();
    }
  }

  private NDArray prefixWeight(ParameterStore parameterStore, NDArray reference, boolean training) {
    NDList weights = new NDList(fusedPrefixComponents);
    for (int component = 0; component < fusedPrefixComponents; component++) {
      Parameter weight = projections[component].getDirectParameters().get("weight");
      weights.add(parameterStore.getValue(weight, reference.getDevice(), training));
    }
    NDArray fusedWeight = NDArrays.concat(weights, 1);
    // Parameterはモデル管理元が所有するが、連結重みはこの順伝播だけの一時値である。
    // 入力の作業用管理元へ移し、小バッチ終了時に確実に解放する。
    fusedWeight.attach(reference.getManager());
    return fusedWeight;
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    for (int component = 0; component < projections.length; component++) {
      projections[component].initialize(manager, dataType, new Shape(-1, inputSizes[component]));
    }
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return new NDList(
        fuse(parameterStore, training, runtimeParameters, inputs.toArray(new NDArray[0])));
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    long[] outputShape = inputShapes[0].getShape().clone();
    outputShape[outputShape.length - 1] = outputSize;
    return new Shape[] {new Shape(outputShape)};
  }

  private static NDArray applyLinear(
      Linear block,
      ParameterStore parameterStore,
      NDArray input,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return block
        .forward(parameterStore, new NDList(input), training, runtimeParameters)
        .singletonOrThrow();
  }
}
