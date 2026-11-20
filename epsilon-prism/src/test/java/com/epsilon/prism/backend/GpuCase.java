package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.kernel.CollectionArchitecture;
import com.epsilon.prism.model.EagerStages;
import com.epsilon.prism.model.PrismNetwork;
import java.nio.ByteBuffer;
import java.util.function.Function;

/**
 * GPU のテストで、ネットワークの形 1 つに使う材料。重みの束、製品の入口で開いた GPU 0 のレーン、バッチ、各段の値を持つ。ネットワークは
 * 重みを乱数でずらしたもので、CPU に置く。各段の値は、CPU の eager の実行で BF16 として求める。四人打ちか三人打ちかは、呼び出し元が
 * 同じ入力の形式で作るネットワークとバッチで決まる。閉じるのは呼び出し元。
 */
final class GpuCase implements AutoCloseable {

  /** eager を CPU で回すバッチの行数。 */
  static final int ROWS = 24;

  /** 重みの世代。 */
  static final long GENERATION = 1;

  /** 記録の行の先頭に書く名前。形の名前で、長い行のバッチでは前に "long " を付ける。 */
  final String name;

  final CollectionArchitecture architecture;
  final NDManager manager;

  final PrismNetwork network;

  final WeightLayout layout;
  final WeightPack pack;
  final ByteBuffer blob;
  final ReferenceBatch hipBatch;
  final HostBatch batch;

  /** 各段の値（eager で求めた BF16）。 */
  final EagerStages.Stages stages;

  final NativeLane lane;

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
    pack = WeightPack.build(layout, network, GENERATION);
    this.batch = batch;
    hipBatch = HipReference.batchOf(batch);
    blob = batch.blob();
    try (NDManager scope = manager.newSubManager()) {
      stages =
          EagerStages.run(network, store, EagerStages.inputs(scope, batch), Precision.BFLOAT16);
    }
    lane = NativeLane.open(network, Device.gpu(0), capacity(ROWS * 4), GENERATION);
  }

  /** 指定した行数のバッチが入る容量を返す。行ごとの上限は四人打ちの値なので、三人打ちのバッチも入る。 */
  static BatchPlan.Capacity capacity(int rows) {
    return new BatchPlan.Capacity(rows, rows * 110, rows * 192, rows * 32);
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

  /** このバッチの s を求める。 */
  float[] forward() {
    return lane.forward(batch);
  }

  /** 別のバッチの s を求める。 */
  float[] forward(HostBatch other) {
    return lane.forward(other);
  }

  @Override
  public void close() {
    try {
      lane.close();
    } finally {
      manager.close();
    }
  }
}
