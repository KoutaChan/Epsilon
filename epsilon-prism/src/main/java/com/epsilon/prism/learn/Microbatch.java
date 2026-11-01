package com.epsilon.prism.learn;

import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.FrontierBatch;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.model.AuxiliaryLabels;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.NetworkInputs;
import com.epsilon.prism.model.StagedBatch;
import com.epsilon.prism.model.UploadRegion;
import com.epsilon.prism.model.Uploads;
import com.epsilon.prism.model.ValueLabel;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * 学習の微小バッチ（1 回の順伝播）。入力（Tokenizer と、前線の腕なら前線の書き手が書いた {@link HostInputs}）と、行の順の差分整合の教師・
 * C の印・補助の頭の教師、補助の項に掛ける割合と補助の頭の平均の分母を持つ。
 *
 * <p>割合と分母は詰め方の 1 塊（{@link StepFeed} の参照の分け方の塊）のもの: 割合は 1 歩の行のうち元の塊の行の割合、分母は元の塊の教師の
 * 条件を満たす所の数。塊を分けない微小バッチは自分の行と教師から決まり、塊を分けた部分は元の塊の割合と分母を持つ（{@link #ofPiece}）。補助の
 * 項は塊ごとの平均に割合を掛けたものなので、部分の項の和が元の塊の項になる。
 *
 * <p>学習の歩の作り手は、割合と分母を付けた後にデバイスへの入力を前もって並べ（{@link #staged}）、レーンはそれを送る（{@link #send}）。
 *
 * @param batch 入力（封じた塊）
 * @param labels 行の順の教師
 * @param measured C の印（塊の手の順。{@link ValueLabel#measured}）
 * @param auxiliary 補助の頭の教師（補助の頭を計算しないなら null）
 * @param tokens 詰めたトークン（行の数 × 最も長い行の S）
 * @param share 補助の項に掛ける割合（1 歩の行のうち元の塊の行の割合）
 * @param denominators 補助の頭の平均の分母（元の塊の数。補助の頭を計算しないなら null）
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
    // HostBatch と FrontierBatch はリトルエンディアン、網の入口はネイティブの順で区画を読む
    if (ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN) {
      throw new IllegalStateException("PRISM host batches need a little-endian machine");
    }
  }

  /**
   * 書き終えた塊と教師から作る。割合は 1 歩がこの微小バッチだけのときの 1、分母は自分の教師から数える（1 歩の中の割合と分母は
   * {@link #ofPiece} で付ける）。
   *
   * @param batch 入力
   * @param labels 行の順の教師
   * @param auxiliary 補助の頭の教師（補助の頭を計算しないなら null）
   * @return 微小バッチ
   */
  static Microbatch of(HostInputs batch, List<ValueLabel> labels, AuxiliaryLabels auxiliary) {
    BatchPlan plan = batch.plan();
    int widest = 0;
    for (int row = 0; row < plan.rows(); row++) {
      widest = Math.max(widest, plan.events(row));
    }
    long tokens = (long) plan.rows() * (BatchPlan.FIXED_TOKENS + widest);
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
   * 詰め方の 1 塊の部分に、元の塊の割合と分母を付ける。割合は元の塊の行の数 / 1 歩の行の数、分母は部分の分母の和。分けない塊は部分 1 つで、
   * 自分の割合と分母になる。
   *
   * @param parts 元の塊の部分（符号化した微小バッチ）
   * @param stepRows 1 歩の行の数
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
   * 網の入力・索引・損失の入力（教師は錨を混ぜた後、補助の頭の分母は元の塊の数）を並べて領域に書いた微小バッチを返す（作業のスレッド）。
   *
   * @param region 領域（{@link StepFeed#release} で返すまでこの微小バッチだけが使う）
   * @return 並べた微小バッチ
   */
  Microbatch staged(UploadRegion region) {
    return new Microbatch(
        batch,
        labels,
        measured,
        auxiliary,
        tokens,
        share,
        denominators,
        StagedBatch.stage(
            region, batch.input(), batch.frontier(), measured, auxiliary, labels, denominators));
  }

  /**
   * 並べた入力を、呼び手のスレッドの今のストリームに非同期の写しで送る（{@link StagedBatch#send}。網の順伝播の前に呼ぶ）。
   *
   * @param manager 配列を作る管理元
   * @param stream 呼び手のスレッドの今のストリーム（CPU の管理元なら null）
   * @return 網の入力と損失の入力
   * @throws IllegalStateException 並べる前のとき
   */
  StagedBatch.Sent send(NDManager manager, PtStream stream) {
    if (staged == null) {
      throw new IllegalStateException("the microbatch has not been staged");
    }
    return staged.send(manager, stream);
  }

  /**
   * 網の入力をその場で作る（{@link NetworkInputs#of(NDManager, Uploads, HostBatch, FrontierBatch)}。索引は順伝播がその場で作って置き場から
   * 送る。CPU では配列が塊の領域を指すので、配列を閉じるまで塊を返さない）。
   *
   * @param manager 配列を作る管理元
   * @param uploads 索引を送る置き場（呼び手のスレッドの物）
   * @return 入力
   */
  NetworkInputs inputs(NDManager manager, Uploads uploads) {
    return NetworkInputs.of(manager, uploads, batch.input(), batch.frontier());
  }
}
