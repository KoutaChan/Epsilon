package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.observe.codec.Bfloat16;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.hip.HipArchitecture;
import com.epsilon.prism.hip.HipBatch;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.HipPlan;
import com.epsilon.prism.hip.HipRegion;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.model.EagerStages;
import com.epsilon.prism.model.PrismNetwork;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.function.Function;

/**
 * GPU のテストで、ネットワークの形 1 つに使う材料。重みの束、GPU の計画、バッチ、各段の値を持つ。ネットワークは重みを乱数でずらしたもので、
 * CPU に置く。各段の値は、CPU の eager の実行で BF16 として求める。四人打ちか三人打ちかは、呼び出し元が同じ入力の形式で作るネットワークとバッチで
 * 決まる。GPU の計画は、ネットワークを知らない {@link HipPlan} を {@link HipBackend} が包んだもの。閉じるのは呼び出し元。
 */
final class GpuCase implements AutoCloseable {

  /** eager を CPU で回すバッチの行数。 */
  static final int ROWS = 24;

  /** 記録の行の先頭に書く名前。形の名前で、長い行のバッチでは前に "long " を付ける。 */
  final String name;

  final HipArchitecture architecture;
  final NDManager manager;

  final PrismNetwork network;

  final WeightLayout layout;
  final WeightPack pack;
  final ByteBuffer blob;
  final HipBatch hipBatch;
  final HostBatch batch;

  /** 各段の値（eager で求めた BF16）。 */
  final EagerStages.Stages stages;

  final PtStream stream;
  final HipBackend backend;

  /**
   * 材料を作る。
   *
   * @param name 記録の行の先頭に書く名前
   * @param build CPU のマネージャーにネットワークを作る関数（作ったネットワークの重みを乱数でずらしてから返す）
   * @param batch 確定したバッチ（ネットワークと同じ入力の形式で、行は {@link #ROWS} の 4 倍まで）
   */
  GpuCase(String name, Function<NDManager, PrismNetwork> build, HostBatch batch) {
    this.name = name;
    manager = NDManager.newBaseManager(Device.cpu());
    network = build.apply(manager);
    ParameterStore store = new ParameterStore(manager, false);
    layout = WeightLayout.of(network);
    architecture = layout.architecture();
    pack = WeightPack.build(layout, network, 1);
    this.batch = batch;
    hipBatch = HipBackend.batchOf(batch);
    blob = batch.blob();
    try (NDManager scope = manager.newSubManager()) {
      stages = EagerStages.run(network, store, EagerStages.inputs(scope, batch), Precision.BFLOAT16);
    }
    stream = newStream();
    backend = HipBackend.open(HipLibrary.load(), stream, layout, capacity(ROWS * 4));
    backend.publish(pack);
  }

  /** 指定した行数のバッチが入る容量を返す。行ごとの上限は四人打ちの値なので、三人打ちのバッチも入る。 */
  static BatchPlan.Capacity capacity(int rows) {
    return new BatchPlan.Capacity(rows, rows * 110, rows * 192, rows * 32);
  }

  HipPlan plan() {
    return backend.plan();
  }

  /** 形が候補層の 2 段目を持つなら true。 */
  boolean reread() {
    return architecture.reread();
  }

  /** 全ての行の、局・席・牌のトークンの数の合計 F·R。 */
  int fixedTokens() {
    return architecture.counts().fixed() * hipBatch.rows();
  }

  /** バッチの全てのトークンの数（F·R + イベントの数）。 */
  int tokens() {
    return fixedTokens() + hipBatch.events();
  }

  /** CPU の参照の実装を作り、作業領域をバッチの大きさで確保する。 */
  HipReference reference() {
    HipReference reference = new HipReference(pack);
    reference.allocate(input(reference));
    return reference;
  }

  /** 参照の実装が読む、バッチの入力。 */
  HipReference.Input input(HipReference reference) {
    return reference.input(blob, hipBatch);
  }

  /** 1 バッチの順伝播で s を求める（キューに積み、終わるのを待ってから読む）。 */
  float[] forward() {
    ByteBuffer scores = ByteBuffer.allocateDirect(hipBatch.actions() * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    plan().forward(blob.duplicate().position(0), hipBatch, scores, pack.version());
    plan().synchronize();
    float[] values = new float[hipBatch.actions()];
    scores.asFloatBuffer().get(values);
    return values;
  }

  /** 領域へ値を BF16 で書き込む。最も近い BF16 に丸めるので、float に入れた BF16 の値はそのまま書かれる。 */
  void putBf16(HipRegion region, float[] values) {
    ByteBuffer buffer = ByteBuffer.allocateDirect(values.length * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      buffer.putShort(Bfloat16.fromFloat(value));
    }
    buffer.flip();
    plan().copyIn(region, buffer, buffer.remaining());
  }

  /** 領域へ FP32 の値を書き込む。 */
  void putFloats(HipRegion region, float[] values) {
    ByteBuffer buffer = ByteBuffer.allocateDirect(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    buffer.asFloatBuffer().put(values);
    plan().copyIn(region, buffer, buffer.capacity());
  }

  /** 入力の blob を GPU の入力の領域へ書き込む。 */
  void putInput() {
    plan().copyIn(HipRegion.INPUT, blob.duplicate().position(0), hipBatch.bytes());
  }

  /** 領域の先頭から、指定した個数の BF16 を float にして読む（同期する）。 */
  float[] getBf16(HipRegion region, int count) {
    ByteBuffer buffer = ByteBuffer.allocateDirect(count * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    plan().copyOut(region, buffer, buffer.capacity());
    plan().synchronize();
    float[] values = new float[count];
    for (int i = 0; i < count; i++) {
      values[i] = Float.intBitsToFloat((buffer.getShort(i * Short.BYTES) & 0xFFFF) << 16);
    }
    return values;
  }

  /** 領域の先頭から、指定した個数の FP32 を読む（同期する）。 */
  float[] getFloats(HipRegion region, int count) {
    ByteBuffer buffer = ByteBuffer.allocateDirect(count * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    plan().copyOut(region, buffer, buffer.capacity());
    plan().synchronize();
    float[] values = new float[count];
    buffer.asFloatBuffer().get(values);
    return values;
  }

  /** 領域の先頭から、指定した個数の INT32 を読む（同期する）。 */
  int[] getInts(HipRegion region, int count) {
    ByteBuffer buffer = ByteBuffer.allocateDirect(count * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    plan().copyOut(region, buffer, buffer.capacity());
    plan().synchronize();
    int[] values = new int[count];
    buffer.asIntBuffer().get(values);
    return values;
  }

  @Override
  public void close() {
    backend.close();
    stream.close();
    manager.close();
  }

  static PtStream newStream() {
    return ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(0));
  }
}
