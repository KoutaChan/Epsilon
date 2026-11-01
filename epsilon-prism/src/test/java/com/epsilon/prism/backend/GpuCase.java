package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.training.ParameterStore;
import com.epsilon.prism.hip.HipBatch;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.HipPlan;
import com.epsilon.prism.hip.HipRegion;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.model.EagerStages;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismNetwork;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

/**
 * GPU の試験の 1 つの深さの材料: 乱数で動かした網（CPU に置く）、WeightPack、GPU の計画、塊、CPU の eager の BF16 の段。GPU の計画は
 * 網を知らない {@link HipPlan} を {@link HipBackend} が包んだもの。閉じるのは呼び手。
 */
final class GpuCase implements AutoCloseable {

  /** eager を CPU で回す塊の行（CPU の packed 注意は [行, S, S, 頭 + A] を作るので小さくする）。 */
  static final int ROWS = 24;

  /** 記録の行の頭（"depth L"、長い行の塊は "long depth L"）。 */
  final String name;
  final int depth;
  final NDManager manager;
  final PrismNetwork network;
  final ParameterStore store;
  final WeightLayout layout;
  final WeightPack pack;
  final PtStream stream;
  final HipBackend backend;
  final List<long[]> rows;
  final HostBatch batch;
  final HipBatch hipBatch;
  final EagerStages.Stages eager;

  /** 価値グラフの全ての作り方が出る {@link #ROWS} 行の塊。 */
  GpuCase(int depth, long seed) {
    this("depth " + depth, depth, seed, HipReferenceTest.mixedRows(ROWS));
  }

  /** 行 rows の塊（{@link #ROWS} の 4 倍まで）。 */
  GpuCase(String name, int depth, long seed, List<long[]> rows) {
    this.name = name;
    this.depth = depth;
    manager = NDManager.newBaseManager(Device.cpu());
    network = HipFixtures.network(manager, depth, seed);
    store = new ParameterStore(manager, false);
    layout = WeightLayout.of(network);
    pack = WeightPack.build(layout, network, 1);
    this.rows = rows;
    batch = HipFixtures.encode(rows);
    hipBatch = HipBackend.batchOf(batch);
    try (NDManager scope = manager.newSubManager()) {
      eager = EagerStages.run(network, store, EagerStages.inputs(scope, batch), Precision.BFLOAT16);
    }
    stream = ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(0));
    backend = HipBackend.open(HipLibrary.load(), stream, layout, capacity(ROWS * 4));
    backend.publish(pack);
  }

  /** 行 rows の塊を入れる容量（試験の塊の大きさ）。 */
  static BatchPlan.Capacity capacity(int rows) {
    return new BatchPlan.Capacity(rows, rows * 110, rows * 192, rows * 32);
  }

  HipPlan plan() {
    return backend.plan();
  }

  /** 区画へ BF16 の値（float に入れた BF16）を写す。 */
  void putBf16(HipRegion region, float[] values) {
    ByteBuffer buffer = ByteBuffer.allocateDirect(values.length * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      buffer.putShort((short) (Float.floatToRawIntBits(value) >>> 16));
    }
    buffer.flip();
    plan().copyIn(region, buffer, buffer.remaining());
  }

  /** 区画へ FP32 の値を写す。 */
  void putFloats(HipRegion region, float[] values) {
    ByteBuffer buffer = ByteBuffer.allocateDirect(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    buffer.asFloatBuffer().put(values);
    plan().copyIn(region, buffer, buffer.capacity());
  }

  /** 入力の blob を写す。 */
  void putInput() {
    ByteBuffer blob = batch.blob().duplicate().position(0);
    plan().copyIn(HipRegion.INPUT, blob, hipBatch.bytes());
  }

  /** 区画の先頭 count 個の BF16 を float で読む（同期する）。 */
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

  /** 区画の先頭 count 個の FP32 を読む（同期する）。 */
  float[] getFloats(HipRegion region, int count) {
    ByteBuffer buffer = ByteBuffer.allocateDirect(count * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    plan().copyOut(region, buffer, buffer.capacity());
    plan().synchronize();
    float[] values = new float[count];
    buffer.asFloatBuffer().get(values);
    return values;
  }

  @Override
  public void close() {
    backend.close();
    stream.close();
    manager.close();
  }
}
