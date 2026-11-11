package com.epsilon.classic.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Embedding;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.nn.transformer.IdEmbedding;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.Precision;

/**
 * 1 回の順伝播の文脈。パラメーターの値の取り出し方と、計算の型への丸め方をここにまとめる。
 *
 * <p>パラメーターは FP32 で持ち、使う所で計算の型（FP32 か BF16）にする。正規化の統計は FP32 で求め、結果を計算の型に丸める。
 *
 * <p>推論の ParameterStore は、モデルのマネージャーに付いたパラメーターの配列そのものを返す。そのため、そこから作った配列（型を変えた
 * 配列や取り出した行）は、全て作業のマネージャーに付け直す。入力から作った配列は、もともと入力のマネージャー（作業のマネージャー）に付く。
 */
final class Pass {

  private static final float NORM_EPSILON = 1e-5f;
  private static final String WEIGHT = "weight";
  private static final String BIAS = "bias";
  private static final String GAMMA = "gamma";
  private static final String BETA = "beta";
  private static final String EMBEDDING = "embedding";

  private final ParameterStore store;
  private final NDManager working;
  private final Device device;
  private final DataType compute;
  private final boolean training;

  /**
   * @param store パラメーターの値を返すストア
   * @param working 作業のマネージャー（入力のマネージャー）
   * @param precision 計算の型
   * @param training 学習の順伝播なら true
   */
  Pass(ParameterStore store, NDManager working, Precision precision, boolean training) {
    this.store = store;
    this.working = working;
    this.device = working.getDevice();
    this.compute = precision.compute();
    this.training = training;
  }

  boolean training() {
    return training;
  }

  DataType compute() {
    return compute;
  }

  NDManager working() {
    return working;
  }

  /** パラメーターの値を計算の型で返す。 */
  NDArray value(Parameter parameter) {
    return cast(store.getValue(parameter, device, training), compute);
  }

  /** パラメーターの値を FP32 で返す。 */
  NDArray fp32(Parameter parameter) {
    return cast(store.getValue(parameter, device, training), DataType.FLOAT32);
  }

  /** 勾配が流れるように型を変える。型が同じならそのまま返す。新しく作った配列は作業のマネージャーに付ける。 */
  NDArray cast(NDArray array, DataType type) {
    if (array.getDataType() == type) {
      return array;
    }
    return own(array.getNDArrayInternal().differentiableCast(type));
  }

  NDArray toCompute(NDArray array) {
    return cast(array, compute);
  }

  NDArray toFloat(NDArray array) {
    return cast(array, DataType.FLOAT32);
  }

  /** 作業のマネージャーに付け直して返す。 */
  NDArray own(NDArray array) {
    array.attach(working);
    return array;
  }

  /**
   * 層が持つ定数の表を、作業のマネージャーに付くコピーにする。コピーはデバイスの中で作り、ホストからは送らない。定数の表をそのまま
   * 演算の左に置くと、結果が定数のマネージャーに付き、順伝播の後も残る。そのため、演算の左に置く定数はこのコピーを使う。
   *
   * @param constant 層の初期化で作った定数の表
   * @return 同じ値の配列（作業のマネージャーに付く）
   */
  NDArray copy(NDArray constant) {
    return own(constant.duplicate());
  }

  /**
   * 線形の層を計算する。入力は、末尾以外の軸を 1 つにまとめて 2 階にしてから掛ける。積とバイアスの和を 1 回の演算で求めるので、丸めは
   * 1 回で済む。
   *
   * @param block 重み（とバイアス）を持つ層
   * @param input 入力 [..., 入力の幅]（計算の型）
   * @return 出力 [..., 出力の幅]（計算の型）
   */
  NDArray linear(Linear block, NDArray input) {
    NDArray weight = value(parameter(block, WEIGHT));
    Parameter biasParameter = block.getDirectParameters().get(BIAS);
    NDArray bias = biasParameter == null ? null : value(biasParameter);
    Shape shape = input.getShape();
    int last = shape.dimension() - 1;
    if (last <= 1) {
      return Linear.linear(input, weight, bias).singletonOrThrow();
    }
    long[] dimensions = shape.getShape().clone();
    NDArray flat = input.reshape(shape.size() / dimensions[last], dimensions[last]);
    NDArray output = Linear.linear(flat, weight, bias).singletonOrThrow();
    dimensions[last] = weight.getShape().get(0);
    return output.reshape(new Shape(dimensions));
  }

  /**
   * 末尾の軸で LayerNorm を取る。統計とゲイン・バイアスは FP32 で、出力は計算の型。FP32 の途中の配列は返す前に解放する。
   *
   * @param block ゲインとバイアスを持つ層
   * @param input 入力 [..., 幅]
   * @return 正規化した値
   */
  NDArray norm(LayerNorm block, NDArray input) {
    try (ArrayScope scope = new ArrayScope()) {
      NDArray values = toFloat(input);
      long width = values.getShape().get(values.getShape().dimension() - 1);
      NDArray normalized =
          LayerNorm.layerNorm(
                  values,
                  new Shape(width),
                  fp32(parameter(block, GAMMA)),
                  fp32(parameter(block, BETA)),
                  NORM_EPSILON)
              .singletonOrThrow();
      return scope.keep(toCompute(normalized));
    }
  }

  /** 埋め込みの表 [行, 幅] を計算の型で返す。 */
  NDArray table(IdEmbedding block) {
    return value(parameter(block, EMBEDDING));
  }

  /**
   * 埋め込みの表 [行, 幅] を計算の型で、作業のマネージャーに付く配列として返す。型を変えない FP32 では、パラメーターと値を共有するビューに
   * する。DJL の融合した Attention は入力を一時的に自分のマネージャーへ付け替えるので、表はこれで渡す。パラメーターの配列そのものを渡すと、
   * 同じネットワークを同時に推論するスレッドの付け替えがぶつかり、パラメーターの配列が閉じられる。
   *
   * @param block 表を持つ層
   * @return 表と同じ値の配列
   */
  NDArray ownedTable(IdEmbedding block) {
    NDArray stored = store.getValue(parameter(block, EMBEDDING), device, training);
    NDArray table = cast(stored, compute);
    return table != stored ? table : own(stored.reshape(stored.getShape()));
  }

  /**
   * 埋め込みの表から行を引く。
   *
   * @param block 表を持つ層
   * @param ids 行の番号（INT32 か INT64）
   * @return 行の番号の形の末尾に幅の軸を足した配列（計算の型、作業のマネージャーに付く）
   */
  NDArray embed(IdEmbedding block, NDArray ids) {
    return own(Embedding.embedding(ids, table(block), SparseFormat.DENSE).singletonOrThrow());
  }

  /** x × sigmoid(x)。 */
  static NDArray silu(NDArray value) {
    return Activation.swish(value, 1f);
  }

  /**
   * スケール付き内積の Attention を、ヘッドごとに求める。スケールは 1/√（ヘッドの幅）。
   *
   * @param queries [..., ヘッド, クエリ, 幅]
   * @param keys [..., ヘッド, キー, 幅]
   * @param values [..., ヘッド, キー, バリューの幅]
   * @param mask 足すバイアス（[..., クエリ, キー] に広げられる形で、使わないキーは −∞）。無ければ null
   * @return [..., ヘッド, クエリ, バリューの幅]
   */
  static NDArray attention(NDArray queries, NDArray keys, NDArray values, NDArray mask) {
    return queries.getNDArrayInternal().scaledDotProductAttention(keys, values, mask, 0.0, false);
  }

  private static Parameter parameter(Block block, String name) {
    Parameter parameter = block.getDirectParameters().get(name);
    if (parameter == null) {
      throw new IllegalStateException("Block has no parameter " + name);
    }
    return parameter;
  }
}
