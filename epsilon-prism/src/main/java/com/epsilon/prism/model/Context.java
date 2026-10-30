package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;

/**
 * 1 回の順伝播の文脈。パラメーターの値の取り方と、丸めの所を 1 か所に置く。
 *
 * <p>推論の ParameterStore はパラメーターの配列そのもの（管理元はモデル）を返すので、そこから作った配列（型の変換・切り出し・行の
 * 集め）は全て作業の管理元に付け直す。そうしないとモデルの管理元に付いたまま閉じられずに残る。入力から作る配列は入力の管理元に付く。
 */
public final class Context {

  private final ParameterStore store;
  private final NDManager working;
  private final Device device;
  private final Precision precision;
  private final boolean training;
  private final AttentionTape tape;
  private final FitProbe probe;

  /**
   * 文脈を作る。
   *
   * @param store パラメーターの値の取り元
   * @param working 作業の管理元（入力の管理元）
   * @param precision 計算の型
   * @param training 学習の順伝播なら true
   */
  public Context(ParameterStore store, NDManager working, Precision precision, boolean training) {
    this(store, working, precision, training, null, FitProbe.NONE);
  }

  /**
   * 背骨の注意を自前の HIP の演算にし、段に時計の印を付ける文脈を作る（{@link Pass} の網の順伝播）。
   *
   * @param store パラメーターの値の取り元
   * @param working 作業の管理元（入力の管理元）
   * @param precision 計算の型
   * @param training 学習の順伝播なら true
   * @param tape 注意の記録（null なら djl-rocm の演算）
   * @param probe 段の時計の印（計測しないなら {@link FitProbe#NONE}）
   */
  public Context(
      ParameterStore store,
      NDManager working,
      Precision precision,
      boolean training,
      AttentionTape tape,
      FitProbe probe) {
    this.store = store;
    this.working = working;
    this.device = working.getDevice();
    this.precision = precision;
    this.training = training;
    this.tape = tape;
    this.probe = probe;
  }

  /** 背骨の注意の自前の演算の記録（なければ null）。 */
  AttentionTape tape() {
    return tape;
  }

  /** 段の時計の印（計測しないなら {@link FitProbe#NONE}）。 */
  FitProbe probe() {
    return probe;
  }

  public NDManager working() {
    return working;
  }

  public DataType compute() {
    return precision.compute();
  }

  /**
   * パラメーターの FP32 の値を返す（写さない）。
   *
   * @param parameter パラメーター
   * @return 値
   */
  public NDArray fp32(Parameter parameter) {
    return store.getValue(parameter, device, training);
  }

  /**
   * パラメーターの値を計算の型で返す（変換で作った配列は作業の管理元に付ける）。
   *
   * @param parameter パラメーター
   * @return 値
   */
  public NDArray value(Parameter parameter) {
    return cast(fp32(parameter), compute());
  }

  /**
   * 微分できる型の変換。型が同じならそのまま返す。
   *
   * @param array 配列
   * @param type 型
   * @return 変換した配列
   */
  public NDArray cast(NDArray array, DataType type) {
    if (array.getDataType() == type) {
      return array;
    }
    return own(array.getNDArrayInternal().differentiableCast(type));
  }

  /** 計算の型にする（丸め）。 */
  public NDArray toCompute(NDArray array) {
    return cast(array, compute());
  }

  public NDArray toFloat(NDArray array) {
    return cast(array, DataType.FLOAT32);
  }

  /**
   * 作業の管理元に付け直す。
   *
   * @param array 配列
   * @return 同じ配列
   */
  public NDArray own(NDArray array) {
    array.attach(working);
    return array;
  }

  /**
   * Linear（計算の型の積、FP32 で累積し、偏りを足して 1 回丸める）。
   *
   * @param input 入力（計算の型）
   * @param weight 重み [出力, 入力]
   * @param bias 偏り（null は偏りなし）
   * @return 出力（計算の型）
   */
  public NDArray linear(NDArray input, Parameter weight, Parameter bias) {
    return linear(input, value(weight), bias == null ? null : value(bias));
  }

  /**
   * 計算の型の重みで Linear を取る。
   *
   * @param input 入力
   * @param weight 重み（計算の型）
   * @param bias 偏り（計算の型、null は偏りなし）
   * @return 出力
   */
  public NDArray linear(NDArray input, NDArray weight, NDArray bias) {
    Shape shape = input.getShape();
    int last = shape.dimension() - 1;
    if (last <= 1) {
      return Linear.linear(input, weight, bias).singletonOrThrow();
    }
    // 3 階以上の入力では Linear が積と偏りの和を別々に丸める（2 回丸める）。先頭の軸を 1 つにまとめた 2 階の入力にして、HIP と同じく
    // FP32 で累積して偏りを足してから 1 回で丸める。
    // Shape.getShape は中の配列をそのまま返すので、写しに書く（入力の形を壊さない）
    long[] dimensions = shape.getShape().clone();
    NDArray flat = input.reshape(shape.size() / dimensions[last], dimensions[last]);
    NDArray output = Linear.linear(flat, weight, bias).singletonOrThrow();
    dimensions[last] = weight.getShape().get(0);
    return output.reshape(new Shape(dimensions));
  }

  /**
   * RMSNorm（入力・統計・利得は FP32、出力は計算の型）。
   *
   * @param input 入力 [..., H]
   * @param gain 利得 [H]
   * @param epsilon ε
   * @return 正規化した値
   */
  public NDArray norm(NDArray input, Parameter gain, double epsilon) {
    NDArray values = toFloat(input);
    long width = values.getShape().get(values.getShape().dimension() - 1);
    NDArray normalized =
        values.getNDArrayInternal().rmsNorm(new long[] {width}, fp32(gain), epsilon);
    return toCompute(normalized);
  }

  /** SiLU（FP32 で計算して入力の型に丸める）。 */
  public static NDArray silu(NDArray input) {
    return Activation.swish(input, 1f);
  }

  /**
   * パラメーターの行を集める（先頭の軸、FP32）。
   *
   * @param parameter パラメーター
   * @param rows 行の番号（1 次元）
   * @return 集めた行
   */
  public NDArray rows(Parameter parameter, NDArray rows) {
    return own(NDArrays.gatherRows(fp32(parameter), rows));
  }

  /**
   * 先頭の軸の [from, to) を切り出す（作業の管理元に付ける）。
   *
   * @param array 配列
   * @param from 始め
   * @param to 終わり
   * @return 切り出し
   */
  public NDArray slice(NDArray array, long from, long to) {
    return own(array.get(new NDIndex("{}:{}", from, to)));
  }
}
