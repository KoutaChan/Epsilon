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
import java.util.Collections;
import java.util.Deque;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.Objects;
import java.util.Set;

/**
 * 学習で、バックボーンの Attention と RMSNorm を自動微分の外の演算（{@link TrainingKernels}。GPU では自前の HIP の演算）で計算し、
 * その逆伝播を DJL の自動微分につなぐ記録。1 回の順伝播と逆伝播に 1 つ使う。スレッド安全ではない。演算が関係のコードも作るなら（GPU の
 * HIP）、バックボーンはそのコードを使う（{@link #relationCodes}）。
 *
 * <p>DJL の自動微分は外の演算の逆伝播を扱えないので、計算の木を葉で切る。順伝播では、値の葉（{@link #cut}）、Attention（{@link #attend}:
 * 入力、出力の葉、保存した値）、正規化（{@link #norm}: 入力、ゲイン、出力の葉、保存した r）を積む。{@link #backward} は損失を逆伝播した後、
 * 積んだのと逆の順に勾配を注入する。注入には根と勾配を渡す VJP（{@code JniUtils.backward}）を使い、根の勾配をそのまま根の木へ流す。
 *
 * <ul>
 *   <li>Attention: 出力の葉の勾配から、自前の逆伝播で dq・dkv（q・kv と同じ型）と dρ を作り、q・kv・ρ をそれぞれ根にして流す。
 *   <li>正規化: 出力の葉の勾配から、自前の逆伝播で dx（入力と同じ型）とゲインの勾配を作り、入力とゲインをそれぞれ根にして流す。
 *   <li>葉: 葉の勾配 g を、葉の元の値を根にして前の木へ流す。
 * </ul>
 *
 * 注入する値は根の勾配そのものなので、Σ 根·勾配 を 1 回逆伝播するのと同じ勾配になる。葉の勾配は全ての寄与がそろってから流すので、どの木も
 * 1 回だけたどる。q と kv の両方が読む正規化の出力を葉にするのはこのためで（{@link EncoderLayer}）、dq と dkv の寄与はその葉の勾配で足されて
 * から、1 回だけ正規化の逆伝播を通る。同じ理由で、自動微分の中では、正規化の入力をこの正規化のほかの演算が読んではならない。ほかにも読む演算
 * があれば、呼び出し元が先に入力を葉にして、読む演算の全てにその葉を渡す。q と kv が共有する重みとバイアスの型変換は、切り出す行が重ならない
 * ので、根を分けて溜めても値は同じ。
 *
 * <p>記録した配列（葉と葉の元、Attention の q・kv・出力・保存した値、正規化の入力・出力・保存した r）は逆伝播まで残す。順伝播のスコープ
 * （{@link Context#scoped}）の中で作った配列は、スコープを閉じる前に記録のマネージャーへ移す（{@link #adopt}）ので、スコープを閉じても
 * 配列は閉じない。記録のマネージャーは順伝播の作業用マネージャーの下に作り（{@link #open}）、それと一緒に閉じる。パラメーター（ρ・ゲイン）
 * と、順伝播の全体で共有する配列（トークンの有無・関係のコード）は移さない。自前の演算は配列を連続したメモリとして読むので、q・kv と正規化の
 * 入力は連続していなければならない。
 *
 * <p>逆伝播では、切れ目ごとに段の時間を計る印を付ける（{@link FitProbe}）。損失の逆伝播は {@link Stage#HEADS_BACKWARD} に入れる。
 * Attention は、自前の逆伝播の演算だけを {@link Stage#ATTENTION_BACKWARD} に入れる。正規化と葉は、注入にかかった時間を、積むときに
 * 受け取った段に入れる。
 *
 * <p>{@link #forwardOnly} は記録しない形で、推論に使う（アンカーの s を学習と同じ演算で出す）。木を切らず、Attention と正規化の出力をそのまま
 * 返す。
 */
public final class AttentionTape {

  private sealed interface Step permits Cut, Attention, Norm {}

  /** 葉の元の値と、その葉。{@code stage} と {@code layer} は、葉の注入の時間を計る段。 */
  private record Cut(NDArray source, NDArray leaf, Stage stage, int layer) implements Step {}

  /** 1 つの層の Attention。{@code output} は出力の葉、{@code outFloat} と {@code logSumExp} は逆伝播のために保存した値。 */
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

  /**
   * 正規化の記録。入力、ゲイン（FP32 のパラメーターの値）、出力の葉、保存した r を持つ。{@code stage} と {@code layer} は、注入の時間を
   * 計る段。
   */
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

  /** 記録のマネージャー。{@link #open} で作り、記録しない形では作らない。 */
  private NDManager arrays;

  /** 記録したが、まだ記録のマネージャーへ移していない配列。 */
  private final Set<NDArray> loose = Collections.newSetFromMap(new IdentityHashMap<>());

  /**
   * 記録を作る。段の時間を計る印は付けない。
   *
   * @param kernels Attention と正規化の演算（GPU では、学習のレーンのストリームに積む HIP の演算）
   */
  public AttentionTape(TrainingKernels kernels) {
    this(kernels, FitProbe.NONE);
  }

  /**
   * 逆伝播の切れ目ごとに、段の時間を計る印を付ける記録を作る。学習器が使う。順伝播の印は、ネットワークが {@link Pass} の同じ印で付ける。
   *
   * @param kernels Attention と正規化の演算（GPU では、学習のレーンのストリームに積む HIP の演算）
   * @param probe 段の時間を計る印
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
   * 記録しない形を作る。推論の順伝播で、Attention と正規化を自前の演算で計算するときに使う。
   *
   * @param kernels Attention と正規化の演算
   * @return 記録しない形
   */
  public static AttentionTape forwardOnly(TrainingKernels kernels) {
    return new AttentionTape(kernels, FitProbe.NONE, false);
  }

  /** 逆伝播のために記録する形（学習）なら true。 */
  public boolean recording() {
    return recording;
  }

  /**
   * 順伝播の作業用マネージャーの下に、記録のマネージャーを作る。ネットワークの順伝播の始めに呼ぶ。記録しない形では何もしない。
   *
   * @param forward 順伝播の作業用マネージャー（逆伝播が終わるまで閉じないこと）
   * @throws IllegalStateException この記録で既に順伝播を 1 回記録したとき
   */
  void open(NDManager forward) {
    if (!recording) {
      return;
    }
    if (arrays != null) {
      throw new IllegalStateException("an attention tape records one forward pass");
    }
    arrays = forward.newSubManager();
  }

  /**
   * 配列が記録のマネージャーに付いているかを返す。
   *
   * @param array 配列
   * @return 記録のマネージャーに付いていれば true
   */
  boolean owns(NDArray array) {
    return arrays != null && array.getManager() == arrays;
  }

  /**
   * スコープを閉じる前に、スコープの中で作って記録した配列を、記録のマネージャーへ移す。スコープが借りた配列は、スコープが元のマネージャーへ
   * 返すので移さない。
   *
   * @param scope 閉じるスコープのマネージャー
   * @param carried スコープが借りた配列
   */
  void adopt(NDManager scope, Set<NDArray> carried) {
    for (Iterator<NDArray> each = loose.iterator(); each.hasNext(); ) {
      NDArray array = each.next();
      if (owns(array)) {
        // 借りたスコープが記録のマネージャーへ返した配列
        each.remove();
      } else if (array.getManager() == scope && !carried.contains(array)) {
        array.attach(arrays);
        each.remove();
      }
    }
  }

  /**
   * 関係のコードを作る自前の演算を返す。記録する形で、Attention の演算がコードも作るとき（GPU の学習の HIP の演算）だけ返す。
   *
   * @return コードを作る演算。無ければ null で、呼び出し元は eager の {@link RelationClass} で作る
   */
  RelationCodes relationCodes() {
    return recording && kernels instanceof RelationCodes codes ? codes : null;
  }

  /**
   * 値を葉にする。層の入力と、q と kv の両方が読む正規化の出力に使う。計測するときは {@link Context#probeCut} の葉にも使う。
   *
   * @param working 葉を付ける作業用マネージャー
   * @param source 値（前の木の出力）
   * @param stage 葉の注入（元の値の木の逆伝播）の時間を計る段
   * @param layer 段の層の番号（層ごとの段でなければ 0）
   * @return 葉（値は同じ）
   */
  NDArray cut(NDManager working, NDArray source, Stage stage, int layer) {
    if (!recording) {
      return source;
    }
    NDArray leaf = source.stopGradient();
    leaf.attach(working);
    leaf.setRequiresGradient(true);
    steps.addLast(new Cut(source, leaf, stage, layer));
    keep(source, leaf);
    return leaf;
  }

  /**
   * Attention を自前の演算で計算し、出力の葉を返す。
   *
   * @param working 出力と保存する値を付ける作業用マネージャー
   * @param query クエリ [行, クエリ, A]（計算の型、連続）
   * @param keyValue キーとバリュー [行, S, 2A]（計算の型、連続）
   * @param mask トークンの有無 [行, S]
   * @param codes 関係のコード [行, S, S, ワード]（クエリの側は先頭の行だけを読む）
   * @param table ρ のパラメーターの値 [領域 × 値, ヘッド]（FP32）
   * @param heads ヘッドの数
   * @param entries 領域あたりの値の数
   * @param scale 内積に掛ける倍率（native には float で渡す。djl-rocm の演算と同じ）
   * @param layer バックボーンの層の番号（逆伝播の段の印に使う）
   * @return 出力 [行, クエリ, A]（計算の型）
   * @throws IllegalArgumentException q か kv が連続していないとき
   */
  NDArray attend(
      NDManager working,
      NDArray query,
      NDArray keyValue,
      NDArray mask,
      NDArray codes,
      NDArray table,
      int heads,
      int entries,
      double scale,
      int layer) {
    requireContiguous(query, "attention query");
    requireContiguous(keyValue, "attention keys and values");
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
    NDArray output = working.create(queries, query.getDataType());
    NDArray outFloat = working.create(queries, DataType.FLOAT32);
    NDArray logSumExp =
        working.create(
            new Shape(dimensions.rows(), dimensions.heads(), dimensions.queries()), DataType.FLOAT32);
    kernels.forward(dimensions, query, keyValue, mask, codes, table, output, outFloat, logSumExp);
    if (recording) {
      output.setRequiresGradient(true);
      steps.addLast(
          new Attention(
              dimensions, query, keyValue, mask, codes, table, output, outFloat, logSumExp, layer));
      keep(query, keyValue, output, outFloat, logSumExp);
    }
    return output;
  }

  /**
   * RMSNorm を自前の演算で計算し、出力の葉を返す。記録しない形では出力をそのまま返す。自動微分の中では、入力をこの正規化のほかの演算が
   * 読んではならない（クラスの説明を参照）。
   *
   * @param working 出力と保存する r を付ける作業用マネージャー
   * @param input 入力 [..., H]（計算の型、連続）
   * @param gain ゲインのパラメーターの値 [H]（FP32）
   * @param epsilon ε
   * @param stage 逆伝播（入力とゲインへの注入）の時間を計る段
   * @param layer 段の層の番号（層ごとの段でなければ 0）
   * @return 正規化した値（入力と同じ型・形）
   * @throws IllegalArgumentException 入力が連続していないとき
   */
  NDArray norm(
      NDManager working, NDArray input, NDArray gain, float epsilon, Stage stage, int layer) {
    requireContiguous(input, "RMSNorm input");
    Shape shape = input.getShape();
    long width = shape.get(shape.dimension() - 1);
    RmsNormKernels.NormShape dimensions =
        new RmsNormKernels.NormShape(
            Math.toIntExact(shape.size() / width), Math.toIntExact(width), epsilon);
    NDArray output = working.create(shape, input.getDataType());
    NDArray inverseRms = working.create(new Shape(dimensions.rows()), DataType.FLOAT32);
    kernels.forward(dimensions, input, gain, output, inverseRms);
    if (recording) {
      output.setRequiresGradient(true);
      steps.addLast(new Norm(dimensions, input, gain, output, inverseRms, stage, layer));
      keep(input, output, inverseRms);
    }
    return output;
  }

  /**
   * 損失から全ての木へ逆伝播する。{@link GradientCollector#backward} の代わりに呼ぶ。コレクターは、葉の勾配に溜める既定の形
   * （{@code Engine#newGradientCollector}）であること。注入の VJP は同じ自動微分に直接積む。記録しない形で呼ぶと
   * {@link IllegalStateException} を投げる。
   *
   * @param collector 勾配のコレクター
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

  /** 記録した配列を、スコープを出るときに記録のマネージャーへ移せるように覚えておく。 */
  private void keep(NDArray... recorded) {
    if (arrays == null) {
      throw new IllegalStateException("the attention tape records only after the network opens it");
    }
    for (NDArray array : recorded) {
      if (!owns(array)) {
        loose.add(array);
      }
    }
  }

  /**
   * 自前の逆伝播で q・kv・ρ の勾配を作り、それぞれを根にして流す。q と kv は重みとバイアスの型変換のノードを共有するので、先に流す q では
   * 木を残す。ρ はパラメーターの葉なので、勾配はその葉に直接溜まる。段の印は自前の逆伝播の演算の後に付ける。ρ の勾配の和と注入の時間は、
   * 次の葉（Attention の前の正規化の出力）の段に入る。
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
    // ワークグループ（行 × ヘッド）ごとの部分和を行について足し、[領域 × 値, ヘッド] にする
    NDArray rho =
        gradTable.reshape(d.rows(), d.heads(), d.tableRows()).sum(new int[] {0}).transpose();
    inject(a.query(), gradQuery, true);
    inject(a.keyValue(), gradKeyValue, false);
    inject(a.table(), rho, false);
  }

  /** 自前の逆伝播で dx とゲインの勾配の部分和を作る。部分和をタイルの順に足し、入力とゲインをそれぞれ根にして流す。 */
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

  /** 根の勾配を与えて逆伝播する（VJP）。{@code keep} なら、後の根のために木を残す。 */
  private static void inject(NDArray root, NDArray gradient, boolean keep) {
    JniUtils.backward((PtNDArray) root, (PtNDArray) gradient, keep, false);
  }

  /** 自前の演算が読む配列が、連続したメモリに並んでいることを確かめる。 */
  private static void requireContiguous(NDArray array, String role) {
    if (!JniUtils.isContiguous((PtNDArray) array)) {
      throw new IllegalArgumentException(
          "the attention tape reads the " + role + " as contiguous memory, got a strided " + array.getShape());
    }
  }
}
