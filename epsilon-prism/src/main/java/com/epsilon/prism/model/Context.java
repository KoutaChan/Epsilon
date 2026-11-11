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
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.model.FitProbe.Stage;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;
import java.util.function.Function;

/**
 * 1 回の順伝播の文脈。パラメーターの値の取り出し方と、どこで丸めるかを 1 か所にまとめる。
 *
 * <p>推論の ParameterStore は、パラメーターの配列そのもの（マネージャーはモデル）を返す。そこから作った配列（型の変換、切り出し、
 * 行の gather）は、全て作業用のマネージャーに付け直す。付け直さないと、モデルのマネージャーに付いたまま閉じられずに残る。入力から
 * 作る配列は、入力のマネージャーに付く。
 *
 * <p>層や、層の中の区切り（Attention・FFN・正規化）で作った途中の配列は、区切りを出るときに閉じる（{@link #scoped}）。閉じないと、
 * バッチの全ての層の途中の配列がバッチの終わりまで残り、使用メモリのピークが 1 つの層の何倍にもなる。学習でも閉じてよい。逆伝播に
 * 要る値は自動微分のグラフが配列とは別に持ち、Attention の記録が逆伝播で読む配列は、記録が区切りから引き取る（{@link AttentionTape#adopt}）。
 */
public final class Context {

  /** 区切り（{@link #scoped}）の中で作った途中の配列の扱い。 */
  enum Intermediates {
    /** 区切りを出るときに閉じる。 */
    CLOSE,
    /** 区切りを作らず、全てを順伝播の作業用のマネージャーに残す。閉じても値と勾配が変わらないことを確かめる比較に使う。 */
    KEEP
  }

  private final ParameterStore store;
  private final NDManager working;
  private final Device device;
  private final Precision precision;
  private final boolean training;
  private final AttentionTape tape;
  private final FitProbe probe;
  private final Intermediates intermediates;

  /**
   * 文脈を作る。
   *
   * @param store パラメーターの値の取り出し元
   * @param working 作業用のマネージャー（入力のマネージャー）
   * @param precision 計算の型
   * @param training 学習の順伝播なら true
   */
  public Context(ParameterStore store, NDManager working, Precision precision, boolean training) {
    this(store, working, precision, training, null, FitProbe.NONE);
  }

  /**
   * バックボーンの Attention を自前の HIP の演算で行い、段ごとに時間を測る文脈を作る（{@link Pass} のネットワークの順伝播で使う）。
   *
   * @param store パラメーターの値の取り出し元
   * @param working 作業用のマネージャー（入力のマネージャー）
   * @param precision 計算の型
   * @param training 学習の順伝播なら true
   * @param tape Attention の記録（null なら djl-rocm の演算を使う。記録するときは、先に {@link AttentionTape#open} を呼んでおく）
   * @param probe 段ごとの時間の計測点（計測しないなら {@link FitProbe#NONE}）
   */
  public Context(
      ParameterStore store,
      NDManager working,
      Precision precision,
      boolean training,
      AttentionTape tape,
      FitProbe probe) {
    this(store, working, precision, training, tape, probe, Intermediates.CLOSE);
  }

  /**
   * 区切りの中で作った途中の配列の扱いを選んで、文脈を作る。
   *
   * @param store パラメーターの値の取り出し元
   * @param working 作業用のマネージャー（入力のマネージャー）
   * @param precision 計算の型
   * @param training 学習の順伝播なら true
   * @param tape Attention の記録（null なら djl-rocm の演算を使う）
   * @param probe 段ごとの時間の計測点
   * @param intermediates 区切りの中で作った途中の配列の扱い
   */
  Context(
      ParameterStore store,
      NDManager working,
      Precision precision,
      boolean training,
      AttentionTape tape,
      FitProbe probe,
      Intermediates intermediates) {
    this.store = store;
    this.working = working;
    this.device = working.getDevice();
    this.precision = precision;
    this.training = training;
    this.tape = tape;
    this.probe = probe;
    this.intermediates = intermediates;
  }

  /** バックボーンの Attention を自前の演算で行うときの記録（なければ null）。 */
  AttentionTape tape() {
    return tape;
  }

  /** 段ごとの時間の計測点（計測しないなら {@link FitProbe#NONE}）。 */
  FitProbe probe() {
    return probe;
  }

  /**
   * 記録があれば、値を葉にする（{@link AttentionTape#cut}。葉は作業用のマネージャーに付く）。記録がなければ値をそのまま返す。
   *
   * @param source 値（勾配を持つグラフの出力）
   * @param stage 葉への勾配の注入の時間を計上する段
   * @param layer 層ごとの段なら層の番号、ほかは 0
   * @return 葉（同じ値）か {@code source}
   */
  NDArray cut(NDArray source, Stage stage, int layer) {
    return tape == null ? source : tape.cut(working, source, stage, layer);
  }

  /**
   * 計測するときだけ、値を葉にする（{@link AttentionTape#cut}）。計測するときとは、{@link FitProbe#cuts} が true で、Attention の
   * 記録がある学習のこと。葉への勾配の注入は {@code stage} の段に計上するので、逆伝播を段に分けて測れる。注入する勾配は葉への寄与の
   * 和そのものなので、勾配の式は葉にしないときと変わらない。注入はグラフを残さない。そのため {@code source} のグラフ（前の葉まで）は、
   * 値を保存したノードを、ほかの葉のグラフと共有してはならない。
   *
   * @param source 値（勾配を持つグラフの出力）
   * @param stage 葉への勾配の注入の時間を計上する段
   * @param layer 層ごとの段なら層の番号、ほかは 0
   * @return 計測するときは葉（同じ値）、ほかは {@code source}
   */
  NDArray probeCut(NDArray source, Stage stage, int layer) {
    return tape == null || !probe.cuts() ? source : tape.cut(working, source, stage, layer);
  }

  public NDManager working() {
    return working;
  }

  /**
   * 区切りの中で計算し、途中の配列は区切りを出るときに閉じる。この文脈の作業用のマネージャーには、計算の出力だけを残す。計算には、
   * 区切りのマネージャーを作業用にした文脈を渡す。演算の結果は最初の引数の配列のマネージャーに付く。そのため、計算が演算に使う区切りの
   * 外の配列（{@code carried}）は、区切りの間だけ区切りのマネージャーに付け、出るときに元のマネージャーへ戻す（閉じない）。
   * {@code carried} に入れない配列から作った配列は、その配列のマネージャーに残る。
   *
   * <p>Attention の記録があれば、区切りを閉じる前に、区切りの中で作って記録した配列を記録のマネージャーへ移す
   * （{@link AttentionTape#adopt}）。出力がその配列なら、外のマネージャーへは付け直さない。
   *
   * <p>区切りで作る配列は、全てこの順伝播のストリームで作って読む。そのため、計算の終わりを待たずに閉じてよい（ストリームが取っておく
   * 領域を使い回すのは、同じストリームで後から確保するときだけ）。ほかのストリームで送った入力は、{@code carried} に入れても閉じない。
   *
   * @param block 区切りの計算（新しく作った配列を返すこと。パラメーターの配列そのものを返してはならない）
   * @param carried 計算が演算に使う、区切りの外の配列（null は飛ばす）
   * @return 計算の出力
   */
  NDArray scoped(Function<Context, NDArray> block, NDArray... carried) {
    if (intermediates == Intermediates.KEEP) {
      return block.apply(this);
    }
    try (NDManager scope = working.newSubManager()) {
      Set<NDArray> attached = Collections.newSetFromMap(new IdentityHashMap<>());
      for (NDArray array : carried) {
        // 同じ配列を 2 回付けると、2 回目は区切りを元のマネージャーとして覚え、区切りを閉じるときに配列も閉じてしまう
        if (array != null && attached.add(array)) {
          array.tempAttach(scope);
        }
      }
      NDArray output =
          block.apply(new Context(store, scope, precision, training, tape, probe, intermediates));
      if (tape != null) {
        tape.adopt(scope, attached);
      }
      if (tape == null || !tape.owns(output)) {
        output.attach(working);
      }
      return output;
    }
  }

  public DataType compute() {
    return precision.compute();
  }

  /**
   * パラメーターの FP32 の値を、コピーせずに返す。
   *
   * @param parameter パラメーター
   * @return 値
   */
  public NDArray fp32(Parameter parameter) {
    return store.getValue(parameter, device, training);
  }

  /**
   * パラメーターの値を計算の型で返す（変換で作った配列は作業用のマネージャーに付ける）。
   *
   * @param parameter パラメーター
   * @return 値
   */
  public NDArray value(Parameter parameter) {
    return cast(fp32(parameter), compute());
  }

  /**
   * 微分できる形で型を変換する。型が同じならそのまま返す。
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

  /** 計算の型に丸める。 */
  public NDArray toCompute(NDArray array) {
    return cast(array, compute());
  }

  public NDArray toFloat(NDArray array) {
    return cast(array, DataType.FLOAT32);
  }

  /**
   * 作業用のマネージャーに付け直す。
   *
   * @param array 配列
   * @return 同じ配列
   */
  public NDArray own(NDArray array) {
    array.attach(working);
    return array;
  }

  /**
   * Linear を計算する。計算の型で積を取って FP32 で累積し、バイアスを足してから 1 回だけ丸める。
   *
   * @param input 入力（計算の型）
   * @param weight 重み [出力, 入力]
   * @param bias バイアス（null ならバイアスなし）
   * @return 出力（計算の型）
   */
  public NDArray linear(NDArray input, Parameter weight, Parameter bias) {
    return linear(input, value(weight), bias == null ? null : value(bias));
  }

  /**
   * 計算の型の重みで Linear を計算する。
   *
   * @param input 入力
   * @param weight 重み（計算の型）
   * @param bias バイアス（計算の型。null ならバイアスなし）
   * @return 出力
   */
  public NDArray linear(NDArray input, NDArray weight, NDArray bias) {
    Shape shape = input.getShape();
    int last = shape.dimension() - 1;
    if (last <= 1) {
      return Linear.linear(input, weight, bias).singletonOrThrow();
    }
    // 3 階以上の入力では、Linear が積とバイアスの和を別々に丸める（2 回丸める）。先頭の軸を 1 つにまとめた 2 階の入力にして、HIP と同じく
    // FP32 で累積し、バイアスを足してから 1 回で丸める。
    // Shape.getShape は中の配列をそのまま返すので、写しに書く（入力の形を壊さない）
    long[] dimensions = shape.getShape().clone();
    NDArray flat = input.reshape(shape.size() / dimensions[last], dimensions[last]);
    NDArray output = Linear.linear(flat, weight, bias).singletonOrThrow();
    dimensions[last] = weight.getShape().get(0);
    return output.reshape(new Shape(dimensions));
  }

  /**
   * RMSNorm を計算する。入力・統計・ゲインは FP32、出力は計算の型。FP32 にした入力と出力は、丸めた出力を作った所で閉じる（{@link #scoped}）。
   *
   * @param input 入力 [..., H]
   * @param gain ゲイン [H]
   * @param epsilon ε
   * @return 正規化した値
   */
  public NDArray norm(NDArray input, Parameter gain, double epsilon) {
    return scoped(
        scope -> {
          NDArray values = scope.toFloat(input);
          long width = values.getShape().get(values.getShape().dimension() - 1);
          NDArray normalized =
              values.getNDArrayInternal().rmsNorm(new long[] {width}, scope.fp32(gain), epsilon);
          return scope.toCompute(normalized);
        },
        input);
  }

  /**
   * バックボーンの RMSNorm（{@link #norm} と同じ式）。Attention の記録があれば、記録の自前の演算（{@link AttentionTape#norm}。
   * 入出力は計算の型、統計・ゲインは FP32）で計算し、記録するときは出力の葉を返す。{@code input} は連続した配列であること。また、
   * そのグラフを自動微分の中でこの正規化のほかから読まないこと（ほかで読むときは、先に葉にした値を渡す）。入力の勾配は記録の逆伝播が
   * {@code input} を根にして流すので、損失の逆伝播が先に同じグラフをたどってはならない。
   *
   * @param input 入力 [..., H]（計算の型）
   * @param gain ゲイン [H]
   * @param epsilon ε
   * @param stage 記録の逆伝播（入力とゲインへの勾配の注入）の時間を計上する段
   * @param layer 層ごとの段の、層の番号
   * @return 正規化した値（計算の型）
   */
  NDArray backboneNorm(NDArray input, Parameter gain, double epsilon, Stage stage, int layer) {
    return tape == null
        ? norm(input, gain, epsilon)
        : tape.norm(working, input, fp32(gain), (float) epsilon, stage, layer);
  }

  /** SiLU を計算する（FP32 で計算し、入力の型に丸める）。 */
  public static NDArray silu(NDArray input) {
    return Activation.swish(input, 1f);
  }

  /**
   * パラメーターから、先頭の軸の行を FP32 で集める。
   *
   * @param parameter パラメーター
   * @param rows 行の番号（1 次元）
   * @return 集めた行
   */
  public NDArray rows(Parameter parameter, NDArray rows) {
    return own(NDArrays.gatherRows(fp32(parameter), rows));
  }

  /**
   * 先頭の軸の [from, to) を切り出す（作業用のマネージャーに付ける）。
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
