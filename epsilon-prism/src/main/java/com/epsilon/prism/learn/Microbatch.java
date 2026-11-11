package com.epsilon.prism.learn;

import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.model.AuxiliaryLabels;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.NetworkInputs;
import com.epsilon.prism.model.StagedBatch;
import com.epsilon.prism.model.ValueLabel;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * 学習のマイクロバッチ（1 回の順伝播の分）。入力（入力の形式の行のライターが書いた {@link HostInputs}）と、行の順に並べた差分整合の
 * 教師・C の印・補助のヘッドの教師を持つ。さらに、補助の項に掛ける割合と、補助のヘッドの平均の分母を持つ。
 *
 * <p>割合と分母は、{@link StepFeed} が詰めた 1 つのバッチ（分ける前のもの）についての値である。割合は 1 ステップの行のうち元のバッチの
 * 行が占める割合で、分母は元のバッチの中で教師の条件を満たす所の数。分けないバッチのマイクロバッチは、自分の行と教師から割合と分母が
 * 決まる。分けた部分は元のバッチの割合と分母を持つ（{@link #ofPiece}）。補助の項はバッチごとの平均に割合を掛けたものなので、部分の項を
 * 足すと元のバッチの項になる。
 *
 * <p>ステップを作る {@link StepFeed} は、割合と分母を付けた後にデバイスへの入力を前もって並べ（{@link #staged}）、レーンはそれを転送する
 * （{@link #send}）。
 *
 * @param batch 入力（計画を確定したバッチ）
 * @param labels 行の順の教師
 * @param measured C の印（バッチの中の手の順。{@link ValueLabel#measured}）
 * @param auxiliary 補助のヘッドの教師（補助のヘッドを計算しないなら null）
 * @param tokens パディングを含むトークン数（行数 × 最も長い行の S）
 * @param share 補助の項に掛ける割合（1 ステップの行のうち元のバッチの行が占める割合）
 * @param denominators 補助のヘッドの平均の分母（元のバッチで数えた数。補助のヘッドを計算しないなら null）
 * @param staged 前もって並べたデバイスへの入力（並べる前は null）
 */
record Microbatch(
    HostInputs batch,
    List<ValueLabel> labels,
    boolean[] measured,
    AuxiliaryLabels auxiliary,
    long tokens,
    float share,
    AuxiliaryLoss.Denominators denominators,
    StagedBatch staged) {

  static {
    // HostBatch はリトルエンディアンで書き、ネットワークの入口はネイティブのバイト順で各領域を読む
    if (ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN) {
      throw new IllegalStateException("PRISM host batches need a little-endian machine");
    }
  }

  /**
   * 書き終えたバッチと教師から作る。割合は 1 ステップがこのマイクロバッチだけのときの 1 にし、分母は自分の教師から数える。1 ステップの
   * 中での割合と分母は {@link #ofPiece} で付ける。
   *
   * @param batch 入力
   * @param labels 行の順の教師
   * @param auxiliary 補助のヘッドの教師（補助のヘッドを計算しないなら null）
   * @return マイクロバッチ
   */
  static Microbatch of(HostInputs batch, List<ValueLabel> labels, AuxiliaryLabels auxiliary) {
    BatchPlan plan = batch.plan();
    int widest = 0;
    for (int row = 0; row < plan.rows(); row++) {
      widest = Math.max(widest, plan.events(row));
    }
    long tokens = (long) plan.rows() * (plan.format().fixedTokens() + widest);
    return new Microbatch(
        batch,
        labels,
        ValueLabel.measured(labels),
        auxiliary,
        tokens,
        1f,
        auxiliary == null ? null : AuxiliaryLoss.Denominators.of(auxiliary),
        null);
  }

  /**
   * 詰めた 1 つのバッチを分けた部分に、元のバッチの割合と分母を付ける。割合は元のバッチの行数 / 1 ステップの行数で、分母は部分の分母の和。
   * 分けないバッチは部分が 1 つなので、自分の割合と分母になる。
   *
   * @param parts 元のバッチの部分（符号化したマイクロバッチ）
   * @param stepRows 1 ステップの行数
   * @return 割合と分母を付けた部分（同じ順）
   */
  static List<Microbatch> ofPiece(List<Microbatch> parts, int stepRows) {
    int rows = 0;
    AuxiliaryLoss.Denominators denominators = null;
    for (Microbatch part : parts) {
      rows += part.rows();
      if (part.denominators != null) {
        denominators = denominators == null ? part.denominators : denominators.plus(part.denominators);
      }
    }
    float share = (float) rows / stepRows;
    List<Microbatch> placed = new ArrayList<>(parts.size());
    for (Microbatch part : parts) {
      placed.add(
          new Microbatch(
              part.batch,
              part.labels,
              part.measured,
              part.auxiliary,
              part.tokens,
              share,
              denominators,
              null));
    }
    return placed;
  }

  int rows() {
    return labels.size();
  }

  /**
   * ネットワークの入力・インデックス・損失の入力を並べてバッファに書き、そのマイクロバッチを返す（作業スレッドで呼ぶ）。教師はアンカーを
   * 混ぜた後の値で、補助のヘッドの分母は元のバッチで数えた数。
   *
   * @param region 書き込むバッファ（{@link StepFeed#release} で返すまで、このマイクロバッチだけが使う）
   * @return 並べたマイクロバッチ
   */
  Microbatch staged(PinnedUploads region) {
    return new Microbatch(
        batch,
        labels,
        measured,
        auxiliary,
        tokens,
        share,
        denominators,
        StagedBatch.stage(region, batch.input(), measured, auxiliary, labels, denominators));
  }

  /**
   * 並べた入力を、転送用のストリームで非同期に転送する（{@link StagedBatch#send}）。計算ストリームは順伝播の前に
   * {@link StagedBatch.Upload#receive} で転送の終わりを待つ。
   *
   * @param manager 転送先の配列を作るマネージャー（計算を終えるまで閉じない）
   * @param stream 転送を積むストリーム（CPU のマネージャーなら null）
   * @return 積んだ転送
   * @throws IllegalStateException まだ並べていないとき
   */
  StagedBatch.Upload send(NDManager manager, PtStream stream) {
    if (staged == null) {
      throw new IllegalStateException("the microbatch has not been staged");
    }
    return staged.send(manager, stream);
  }

  /**
   * ネットワークの入力をその場で作る（{@link NetworkInputs#of(NDManager, PinnedUploads, HostBatch)}）。インデックスは
   * 順伝播がその場で作り、バッファから転送する。CPU では配列がバッチの領域を指すので、配列を閉じるまでバッチを返さない。
   *
   * @param manager 配列を作るマネージャー
   * @param uploads インデックスを転送するバッファ（呼び出し元のスレッドのもの）
   * @return 入力
   */
  NetworkInputs inputs(NDManager manager, PinnedUploads uploads) {
    return NetworkInputs.of(manager, uploads, batch.input());
  }
}
