package com.epsilon.prism.backend;

import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.hip.HipBatch;
import com.epsilon.prism.hip.HipKernel;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.HipPlan;
import com.epsilon.prism.hip.HipSection;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.Tensor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * ネットワークの HIP の推論を、1 つのスロットで行う。ネットワークに依る部分（重みの並びと、入力の領域から HIP の領域への対応）を
 * ここに置き、ネットワークを知らない {@link HipPlan} を動かす。
 *
 * <p>1 バッチの処理は、HostBatch の blob を H2D で 1 回送り、形の記述の順にカーネル（{@link HipKernel}）を流し、s を D2H で 1 回読む。
 * s はバッチの手の順（要求の順）に並ぶ FP32。{@link #submit} はストリームに積むだけなので、終わりはストリームのイベントか
 * {@link #synchronize()} で待つ。スレッドセーフではない。
 */
public final class HipBackend implements AutoCloseable {

  private static final HipSection[] SECTIONS = sections();

  private final WeightLayout layout;
  private final HipPlan plan;
  private final ByteBuffer scores;

  private HipBackend(WeightLayout layout, HipPlan plan, ByteBuffer scores) {
    this.layout = layout;
    this.plan = plan;
    this.scores = scores;
  }

  /**
   * スロットの計画を作る。作業領域と重みのバッファは、ここで 1 回だけ確保する。
   *
   * @param library 読み込み済みの libprism_hip.so
   * @param stream スロットの計算ストリーム（GPU）
   * @param layout ネットワークの重みの並びを決める表
   * @param capacity バッチの容量
   * @return 推論のバックエンド
   */
  public static HipBackend open(
      HipLibrary library, PtStream stream, WeightLayout layout, BatchPlan.Capacity capacity) {
    Objects.requireNonNull(layout, "layout");
    HipPlan plan = HipPlan.create(library, stream, layout.hipShape(capacity));
    ByteBuffer scores =
        ByteBuffer.allocateDirect(Math.multiplyExact(capacity.actions(), Float.BYTES))
            .order(ByteOrder.LITTLE_ENDIAN);
    return new HipBackend(layout, plan, scores);
  }

  /** ネットワークを知らない計画を返す。カーネルごとのテストと計測で使う。 */
  public HipPlan plan() {
    return plan;
  }

  public WeightLayout layout() {
    return layout;
  }

  /** デバイスに送った重みのバージョンを返す。まだ送っていなければ −1。 */
  public long weightsVersion() {
    return plan.weightsVersion();
  }

  /**
   * 重みを公開する。デバイスへ送り、ストリームを同期する。別の表で作った重みを渡すと IllegalArgumentException を投げる。
   *
   * @param pack 同じ表で作った重みの束
   */
  public void publish(WeightPack pack) {
    if (pack.layout() != layout) {
      throw new IllegalArgumentException("the weight pack was built for another layout");
    }
    plan.uploadWeights(pack.buffer(), pack.version());
  }

  /**
   * 1 バッチの順伝播をストリームに積む。
   *
   * @param batch 確定したバッチ（blob はダイレクトバッファ。推論ではページ固定）
   * @param out s を書く先のダイレクトバッファ（位置 0 から手の数だけ FP32 を書く）
   * @param version 期待するネットワークのバージョン（送ったバージョンと違えば例外）
   */
  public void submit(HostBatch batch, ByteBuffer out, long version) {
    plan.forward(batch.blob().duplicate().position(0), batchOf(batch), out.duplicate().position(0), version);
  }

  /**
   * 1 バッチの s を計算して返す。ストリームに積み、終わるのを待ってから読む。
   *
   * @param batch 確定したバッチ
   * @param version 期待するネットワークのバージョン
   * @return s（バッチの手の順）
   */
  public float[] forward(HostBatch batch, long version) {
    submit(batch, scores, version);
    plan.synchronize();
    float[] result = new float[batch.plan().count(Axis.ACTION)];
    scores.duplicate().order(ByteOrder.LITTLE_ENDIAN).position(0).asFloatBuffer().get(result);
    return result;
  }

  /** ストリームに積んだ処理が全て終わるのを待つ。 */
  public void synchronize() {
    plan.synchronize();
  }

  @Override
  public void close() {
    plan.close();
  }

  /**
   * 確定したバッチの行・イベント・遷移・手の数と領域ごとの位置を、HIP のバッチの形にして返す。入力の領域と HIP の領域は名前で対応させる。
   *
   * @param batch 確定したバッチ
   * @return HIP のバッチ
   */
  public static HipBatch batchOf(HostBatch batch) {
    BatchPlan plan = batch.plan();
    long[] offsets = new long[SECTIONS.length];
    for (Tensor tensor : Tensor.values()) {
      offsets[SECTIONS[tensor.ordinal()].ordinal()] = plan.offset(tensor);
    }
    return new HipBatch(
        plan.rows(),
        plan.count(Axis.EVENT),
        plan.count(Axis.TRANSITION),
        plan.count(Axis.ACTION),
        plan.bytes(),
        offsets);
  }

  /** 入力の領域ごとに、同じ名前の HIP の領域を返す。軸の名前と要素のバイト数も同じでなければならない。 */
  private static HipSection[] sections() {
    Tensor[] tensors = Tensor.values();
    if (tensors.length != HipSection.values().length) {
      throw new IllegalStateException("HIP sections do not match the input tensors");
    }
    HipSection[] sections = new HipSection[tensors.length];
    for (Tensor tensor : tensors) {
      HipSection section = HipSection.valueOf(tensor.name());
      if (!section.axis().name().equals(tensor.axis().name())
          || section.element().bytes() != tensor.element().bytes()) {
        throw new IllegalStateException(tensor + " does not match the HIP section " + section);
      }
      sections[tensor.ordinal()] = section;
    }
    return sections;
  }
}
