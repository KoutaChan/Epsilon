package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtPinnedBuffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.EnumMap;
import java.util.Map;
import java.util.function.Consumer;

/**
 * 区画をデバイスへ送るホストの置き場（{@link Staging#send}）。型ごとにページ固定（pinned）のバッファを 1 本持ち、足りなければ取り直して使い回す。
 * 送るたびに使い捨てのホストの領域を取らず（GC まで残る直のバッファを作らない）、デバイスへの写しは固定メモリからの DMA になる。写しは同期
 * なので、送り終えたバッファはすぐ次の区画に使える。スレッド安全ではない（学習のレーン・推論の 1 回の仕事など、1 つのスレッドが順に使う）。
 *
 * <p>バッファは作るときの管理元の子に付け、{@link #close} かその管理元を閉じると返る。
 */
public final class Uploads implements AutoCloseable {

  private final PtNDManager host;
  private final Map<Staging.Kind, PtPinnedBuffer> buffers = new EnumMap<>(Staging.Kind.class);

  private Uploads(PtNDManager host) {
    this.host = host;
  }

  /**
   * 置き場を作る。
   *
   * @param owner バッファを付ける管理元（PyTorch。デバイスは問わない）
   * @return 置き場
   */
  public static Uploads on(NDManager owner) {
    return new Uploads((PtNDManager) owner.newSubManager());
  }

  /**
   * 型の区画を elements 個書いて、管理元のデバイスの新しい配列 [elements] へ写す（同期）。バッファが足りなければ取り直す。
   *
   * @param manager 配列を作る管理元
   * @param kind 型
   * @param elements 要素の数
   * @param writer バッファ（位置 0 から、ネイティブのバイト順）に elements 個を書く
   * @return 配列
   */
  NDArray send(NDManager manager, Staging.Kind kind, int elements, Consumer<ByteBuffer> writer) {
    PtPinnedBuffer buffer = buffers.get(kind);
    if (buffer == null || buffer.size() < elements) {
      if (buffer != null) {
        buffer.close();
      }
      buffer = host.allocatePinned(elements, kind.type());
      buffers.put(kind, buffer);
    }
    writer.accept(buffer.getByteBuffer().duplicate().order(ByteOrder.nativeOrder()));
    PtNDArray array = (PtNDArray) manager.create(new Shape(elements), kind.type());
    array.copyFromPinnedBuffer(buffer);
    return array;
  }

  @Override
  public void close() {
    buffers.clear();
    host.close();
  }
}
