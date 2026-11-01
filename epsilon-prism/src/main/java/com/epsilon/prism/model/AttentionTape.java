package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;
import com.epsilon.prism.hip.RelationAttention;
import com.epsilon.prism.hip.RelationCodes;
import com.epsilon.prism.hip.RmsNormKernels;
import com.epsilon.prism.hip.TrainingKernels;
import com.epsilon.prism.model.FitProbe.Stage;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Objects;

/**
 * 学習で、背骨の注意と RMSNorm を自動微分の外の演算（{@link TrainingKernels}。GPU は自前の HIP の演算）で計算し、その逆伝播を DJL の
 * 自動微分につなぐ記録（1 回の順伝播と逆伝播に 1 つ。スレッド安全ではない）。演算が関係の符号も作るなら（GPU の HIP）、背骨はその符号を
 * 使う（{@link #relationCodes}）。
 *
 * <p>DJL の自動微分は外の演算の逆伝播を持てないので、計算の木を葉で切る。順伝播では値の葉（{@link #cut}）と、注意（{@link #attend}:
 * 入力・出力の葉・保存した値）と正規化（{@link #norm}: 入力・利得・出力の葉・保存した r）を積む。{@link #backward} は損失の逆伝播の後、
 * 積んだ順の逆に、根と勾配の VJP（{@code JniUtils.backward}。根の勾配をそのまま根の木へ流す）で勾配を注入する:
 *
 * <ul>
 *   <li>注意: 出力の葉の勾配から自前の逆伝播で dq・dkv（q・kv の型）と dρ を作り、q・kv・ρ をそれぞれ根にして流す。
 *   <li>正規化: 出力の葉の勾配から自前の逆伝播で dx（入力の型）と利得の勾配を作り、入力と利得をそれぞれ根にして流す。
 *   <li>葉: その勾配 g を、葉の元の値を根にして前の木へ流す。
 * </ul>
 *
 * 注入する値は根の勾配そのものなので、Σ 根·勾配 を 1 回逆伝播するのと同じ勾配になる。葉の勾配は全ての寄与が揃ってから流すので、どの木も
 * 1 回だけたどる。q と kv が共に読む正規化の出力が葉なのはこのためで（{@link EncoderLayer}）、dq・dkv の寄与はその葉の勾配で足されてから
 * 1 回だけ正規化の逆伝播を通る。同じ理由で、正規化の入力の木は正規化のほかに自動微分の中から読まれてはならない（読み手がほかにあれば、
 * 呼び手が先に葉にしてその葉を全ての読み手に渡す）。q と kv が共にする重みと偏りの型の変換は、切り出す行が重ならないので、根を分けて
 * 溜めても値は同じ。
 *
 * <p>逆伝播は区切りごとに段の時計の印を付ける（{@link FitProbe}）: 損失の逆伝播は {@link Stage#HEADS_BACKWARD}、注意は自前の逆伝播
 * の演算だけを {@link Stage#ATTENTION_BACKWARD}、正規化と葉はその注入を、積むときに受け取った段にする。
 *
 * <p>{@link #forwardOnly} は記録しない形（推論。錨の s を学習と同じ演算で出す）で、木を切らず、注意と正規化の出力をそのまま返す。
 */
public final class AttentionTape {

  private sealed interface Step permits Cut, Attention, Norm {}

  /** 葉の元の値（source）と、その葉。stage・layer は葉の注入を計る段。 */
  private record Cut(NDArray source, NDArray leaf, Stage stage, int layer) implements Step {}

  /** 層 layer の注意。output は出力の葉、outFloat・logSumExp は逆伝播のために保存した値。 */
  private record Attention(
      RelationAttention.Dimensions dimensions,
      NDArray query,
      NDArray keyValue,
      NDArray mask,
      NDArray codes,
      NDArray table,
      NDArray output,
      NDArray outFloat,
      NDArray logSumExp,
      int layer)
      implements Step {}

  /** 正規化の入力 input・利得 gain（FP32 のパラメーターの値）・出力の葉 output・保存した r。stage・layer は注入を計る段。 */
  private record Norm(
      RmsNormKernels.NormShape shape,
      NDArray input,
      NDArray gain,
      NDArray output,
      NDArray inverseRms,
      Stage stage,
      int layer)
      implements Step {}

  private final TrainingKernels kernels;
  private final FitProbe probe;
  private final boolean recording;
  private final Deque<Step> steps = new ArrayDeque<>();

  /**
   * 記録を作る（段の時計の印は付けない）。
   *
   * @param kernels 注意と正規化の演算（GPU は学習のレーンのストリームに積む HIP の演算）
   */
  public AttentionTape(TrainingKernels kernels) {
    this(kernels, FitProbe.NONE);
  }

  /**
   * 逆伝播の区切りに段の時計の印を付ける記録を作る（当てはめの学習器。順伝播の印は網が {@link Pass} の同じ印で付ける）。
   *
   * @param kernels 注意と正規化の演算（GPU は学習のレーンのストリームに積む HIP の演算）
   * @param probe 段の時計の印
   */
  public AttentionTape(TrainingKernels kernels, FitProbe probe) {
    this(kernels, probe, true);
  }

  private AttentionTape(TrainingKernels kernels, FitProbe probe, boolean recording) {
    this.kernels = Objects.requireNonNull(kernels, "kernels");
    this.probe = Objects.requireNonNull(probe, "probe");
    this.recording = recording;
  }

  /**
   * 記録しない形を作る（推論の順伝播で注意と正規化を自前の演算にする）。
   *
   * @param kernels 注意と正規化の演算
   * @return 記録しない形
   */
  public static AttentionTape forwardOnly(TrainingKernels kernels) {
    return new AttentionTape(kernels, FitProbe.NONE, false);
  }

  /** 逆伝播のために記録するなら true（学習）。 */
  public boolean recording() {
    return recording;
  }

  /**
   * 関係の符号を作る自前の演算を返す（記録する形で、注意の演算が符号も作るときだけ。GPU の学習の HIP の演算）。
   *
   * @return 符号の演算（なければ null で、呼び手は eager の {@link RelationClass} で作る）
   */
  RelationCodes relationCodes() {
    return recording && kernels instanceof RelationCodes codes ? codes : null;
  }

  /**
   * 値を葉にする（層の入力と、q・kv が共に読む正規化の出力。計測のときは背骨の出力も）。
   *
   * @param source 値（前の木の出力）
   * @param stage 葉の注入（元の値の木の逆伝播）を計る段
   * @param layer 段の層の番号（層の段でなければ 0）
   * @return 葉（同じ値）
   */
  NDArray cut(NDArray source, Stage stage, int layer) {
    if (!recording) {
      return source;
    }
    NDArray leaf = source.stopGradient();
    leaf.setRequiresGradient(true);
    steps.addLast(new Cut(source, leaf, stage, layer));
    return leaf;
  }

  /**
   * 注意を自前の演算で計算し、出力の葉を返す。
   *
   * @param query 問い合わせ [行, 問い合わせ, A]（計算の型）
   * @param keyValue 鍵と値 [行, S, 2A]（計算の型）
   * @param mask トークンの在否 [行, S]
   * @param codes 関係の符号 [行, S, S, 語]（問い合わせは先頭の行だけを読む）
   * @param table ρ のパラメーターの値 [区画 × 値, 頭]（FP32）
   * @param heads 頭の数
   * @param entries 区画あたりの値の数
   * @param scale 内積の倍率（native には float で渡す。djl-rocm の演算と同じ）
   * @param layer 背骨の層の番号（逆伝播の段の印）
   * @return 出力 [行, 問い合わせ, A]（計算の型）
   */
  NDArray attend(
      NDArray query,
      NDArray keyValue,
      NDArray mask,
      NDArray codes,
      NDArray table,
      int heads,
      int entries,
      double scale,
      int layer) {
    Shape queries = query.getShape();
    Shape pairs = codes.getShape();
    RelationAttention.Dimensions dimensions =
        new RelationAttention.Dimensions(
            Math.toIntExact(queries.get(0)),
            Math.toIntExact(queries.get(1)),
            Math.toIntExact(keyValue.getShape().get(1)),
            Math.toIntExact(pairs.get(1)),
            heads,
            Math.toIntExact(pairs.get(3)),
            Math.toIntExact(table.getShape().get(0) / entries),
            entries,
            (float) scale);
    NDManager manager = query.getManager();
    NDArray output = manager.create(queries, query.getDataType());
    NDArray outFloat = manager.create(queries, DataType.FLOAT32);
    NDArray logSumExp =
        manager.create(
            new Shape(dimensions.rows(), dimensions.heads(), dimensions.queries()), DataType.FLOAT32);
    kernels.forward(dimensions, query, keyValue, mask, codes, table, output, outFloat, logSumExp);
    if (recording) {
      output.setRequiresGradient(true);
      steps.addLast(
          new Attention(
              dimensions, query, keyValue, mask, codes, table, output, outFloat, logSumExp, layer));
    }
    return output;
  }

  /**
   * RMSNorm を自前の演算で計算し、出力の葉を返す（記録しない形では出力をそのまま返す）。入力の木は、この正規化のほかに自動微分の中から
   * 読まれないこと（クラスの説明）。
   *
   * @param input 入力 [..., H]（計算の型）
   * @param gain 利得のパラメーターの値 [H]（FP32）
   * @param epsilon ε
   * @param stage 逆伝播（入力と利得への注入）を計る段
   * @param layer 段の層の番号（層の段でなければ 0）
   * @return 正規化した値（入力と同じ型・形）
   */
  NDArray norm(NDArray input, NDArray gain, float epsilon, Stage stage, int layer) {
    Shape shape = input.getShape();
    long width = shape.get(shape.dimension() - 1);
    RmsNormKernels.NormShape dimensions =
        new RmsNormKernels.NormShape(
            Math.toIntExact(shape.size() / width), Math.toIntExact(width), epsilon);
    NDManager manager = input.getManager();
    NDArray output = manager.create(shape, input.getDataType());
    NDArray inverseRms = manager.create(new Shape(dimensions.rows()), DataType.FLOAT32);
    kernels.forward(dimensions, input, gain, output, inverseRms);
    if (recording) {
      output.setRequiresGradient(true);
      steps.addLast(new Norm(dimensions, input, gain, output, inverseRms, stage, layer));
    }
    return output;
  }

  /**
   * 損失から全ての木へ逆伝播する（{@link GradientCollector#backward} の代わり）。集め手は葉の勾配に溜める既定の形
   * （{@code Engine#newGradientCollector}）であること。注入の VJP は同じ自動微分に直に積む。
   *
   * @param collector 勾配の集め手
   * @param loss 損失（スカラー）
   */
  public void backward(GradientCollector collector, NDArray loss) {
    if (!recording) {
      throw new IllegalStateException("a forward-only attention tape has nothing to back-propagate");
    }
    collector.backward(loss);
    probe.mark(Stage.HEADS_BACKWARD);
    while (!steps.isEmpty()) {
      switch (steps.removeLast()) {
        case Attention attention -> inject(attention);
        case Norm norm -> {
          inject(norm);
          probe.mark(norm.stage(), norm.layer());
        }
        case Cut cut -> {
          inject(cut.source(), cut.leaf().getGradient(), false);
          probe.mark(cut.stage(), cut.layer());
        }
      }
    }
  }

  /**
   * 自前の逆伝播で q・kv・ρ の勾配を作り、それぞれを根にして流す。q と kv は重みと偏りの型の変換の節を共にするので、先に流す q は木を
   * 残す。ρ はパラメーターの葉なので、その勾配に直に溜まる。段の印は自前の逆伝播の演算の後に付け、ρ の和と注入は次の葉（注意の前の
   * 正規化の出力）の段に入る。
   */
  private void inject(Attention a) {
    RelationAttention.Dimensions d = a.dimensions();
    NDManager manager = a.output().getManager();
    NDArray gradQuery = manager.create(a.query().getShape(), a.query().getDataType());
    NDArray gradKeyValue = manager.create(a.keyValue().getShape(), a.keyValue().getDataType());
    NDArray gradTable =
        manager.create(new Shape((long) d.rows() * d.heads(), d.tableRows()), DataType.FLOAT32);
    kernels.backward(
        d,
        a.query(),
        a.keyValue(),
        a.mask(),
        a.codes(),
        a.table(),
        a.outFloat(),
        a.logSumExp(),
        a.output().getGradient(),
        gradQuery,
        gradKeyValue,
        gradTable);
    probe.mark(Stage.ATTENTION_BACKWARD, a.layer());
    // 作業群（行 × 頭）ごとの部分和を行で足して [区画 × 値, 頭] にする
    NDArray rho =
        gradTable.reshape(d.rows(), d.heads(), d.tableRows()).sum(new int[] {0}).transpose();
    inject(a.query(), gradQuery, true);
    inject(a.keyValue(), gradKeyValue, false);
    inject(a.table(), rho, false);
  }

  /** 自前の逆伝播で dx と利得の勾配の部分和を作り、部分和を塊の順に足して、入力と利得をそれぞれ根にして流す。 */
  private void inject(Norm n) {
    RmsNormKernels.NormShape d = n.shape();
    NDManager manager = n.output().getManager();
    NDArray gradInput = manager.create(n.input().getShape(), n.input().getDataType());
    NDArray gradGain = manager.create(new Shape(d.partials(), d.width()), DataType.FLOAT32);
    kernels.backward(
        d, n.input(), n.gain(), n.inverseRms(), n.output().getGradient(), gradInput, gradGain);
    inject(n.input(), gradInput, false);
    inject(n.gain(), gradGain.sum(new int[] {0}), false);
  }

  /** 根 root の勾配を gradient として逆伝播する（VJP。keep なら後の根のために木を残す）。 */
  private static void inject(NDArray root, NDArray gradient, boolean keep) {
    JniUtils.backward((PtNDArray) root, (PtNDArray) gradient, keep, false);
  }
}
