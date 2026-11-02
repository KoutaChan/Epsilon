package com.epsilon.prism.backend;

import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.hip.HipBatch;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.HipPlan;
import com.epsilon.prism.hip.HipSection;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.input.InputSchema.Tensor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * 網の HIP の推論（1 つの slot）。網に依る部分（重みの並び・入力の区画の写し）をここに置き、網を知らない {@link HipPlan} を回す。
 *
 * <p>1 塊は HostBatch の blob の H2D 1 回 → K1〜K7 → s の D2H 1 回。s は要求の順（塊の手の順）の FP32。
 * {@link #submit} はストリームに積むだけで、終わりはストリームの印か {@link #synchronize()} で待つ。スレッド安全ではない。
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
   * slot の計画を作る（作業域と重みの置き場を一度だけ取る）。
   *
   * @param library 読み込み済みの libprism_hip.so
   * @param stream slot の計算ストリーム（GPU）
   * @param layout 網の表
   * @param capacity 塊の容量
   * @return 推論の道
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

  /** 網を知らない計画（カーネルごとの試験と測りで使う）。 */
  public HipPlan plan() {
    return plan;
  }

  public WeightLayout layout() {
    return layout;
  }

  /** 写した重みの版（まだなら −1）。 */
  public long weightsVersion() {
    return plan.weightsVersion();
  }

  /**
   * 重みを公開する（デバイスへ写し、ストリームを同期する）。
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
   * 1 塊の順伝播をストリームに積む。
   *
   * @param batch 封じた塊（blob は直の ByteBuffer。推論ではページ固定）
   * @param out s を書く先（位置 0 から手の数の FP32、直の ByteBuffer）
   * @param version 期待する網の版（写した版と違えば例外）
   */
  public void submit(HostBatch batch, ByteBuffer out, long version) {
    plan.forward(batch.blob().duplicate().position(0), batchOf(batch), out.duplicate().position(0), version);
  }

  /**
   * 1 塊の s を計算して返す（積んで、待って、読む）。
   *
   * @param batch 封じた塊
   * @param version 期待する網の版
   * @return s（塊の手の順）
   */
  public float[] forward(HostBatch batch, long version) {
    submit(batch, scores, version);
    plan.synchronize();
    float[] result = new float[batch.plan().count(Axis.ACTION)];
    scores.duplicate().order(ByteOrder.LITTLE_ENDIAN).position(0).asFloatBuffer().get(result);
    return result;
  }

  /** ストリームの仕事が終わるのを待つ。 */
  public void synchronize() {
    plan.synchronize();
  }

  @Override
  public void close() {
    plan.close();
  }

  /**
   * 封じた塊の数と区画の位置（入力の区画 → HIP の区画は名前で写す）。
   *
   * @param batch 封じた塊
   * @return HIP の塊
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

  /** 入力の区画ごとの HIP の区画（名前・軸・要素の型が同じこと）。 */
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
