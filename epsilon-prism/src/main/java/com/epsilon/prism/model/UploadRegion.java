package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtPinnedBuffer;
import ai.djl.pytorch.engine.PtStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;

/**
 * 前もって並べた区画の組を非同期でデバイスへ写すための、ページ固定の領域（{@link Staging#write}・{@link Staging#enqueue}）。型ごとに
 * ページ固定のバッファを 1 本持ち、足りなければ取り直して使い回す。
 *
 * <p>写し元の寿命の約束: 非同期の写し（DMA）は積んだ後もバッファを読むので、最後の写しの後に記録した event（{@link #copied}）が終わる
 * まで、どのバッファも書き換えず、解放もしない。書く前（{@link Staging#write}）・取り直す前・閉じる前に {@link #await} で待つ。写し元は
 * 全て領域が持つ固定のメモリで、写しのためにその場で固定する一時のホストのメモリは使わない。
 *
 * <p>一度に 1 つの組だけが使い、使う手番（書く・写しを積む・待つ）は呼び手が happens-before でつなぐ（学習の歩の作り手が書き、レーンが
 * 積み、歩の後に返された領域を次の書き手が待つ）。
 */
public final class UploadRegion implements AutoCloseable {

  private final PtNDManager host;
  private final Map<Staging.Kind, PtPinnedBuffer> buffers = new EnumMap<>(Staging.Kind.class);

  /** デバイスごとの写しの終わりの event（そのデバイスのレーンのストリームで作る）。 */
  private final Map<Device, PtEvent> events = new HashMap<>();

  /** 最後に積んだ写しの event（終わりを待った後、または写しを積んでいなければ null）。 */
  private PtEvent pending;

  private UploadRegion(PtNDManager host) {
    this.host = host;
  }

  /**
   * 領域を作る。
   *
   * @param owner バッファを付ける管理元（PyTorch のホスト）
   * @return 領域
   */
  public static UploadRegion on(NDManager owner) {
    return new UploadRegion((PtNDManager) owner.newSubManager());
  }

  /** 最後に積んだ写しが終わるまで待つ（積んでいなければ何もしない）。 */
  void await() {
    if (pending != null) {
      pending.synchronize();
      pending = null;
    }
  }

  /**
   * 型のバッファを elements 個以上にして、位置 0 からのネイティブのバイト順の窓を返す。{@link #await} の後だけ呼ぶ（取り直しは古い
   * バッファを解放する）。
   */
  ByteBuffer buffer(Staging.Kind kind, int elements) {
    if (pending != null) {
      throw new IllegalStateException("the region is written while a copy from it is pending");
    }
    PtPinnedBuffer buffer = buffers.get(kind);
    if (buffer == null || buffer.size() < elements) {
      if (buffer != null) {
        buffer.close();
      }
      buffer = host.allocatePinned(elements, kind.type());
      buffers.put(kind, buffer);
    }
    return buffer.getByteBuffer().order(ByteOrder.nativeOrder());
  }

  /** 型のバッファ（{@link #buffer} で取った後）。 */
  PtPinnedBuffer pinned(Staging.Kind kind) {
    PtPinnedBuffer buffer = buffers.get(kind);
    if (buffer == null) {
      throw new IllegalStateException("no " + kind + " buffer in the region");
    }
    return buffer;
  }

  /**
   * 写しを積んだ直後に、呼び手のスレッドの今のストリームへ event を記録する（CPU では写しが同期なので何もしない）。
   *
   * @param stream 写しを積んだストリーム（呼び手のスレッドの今のストリーム。CPU なら null）
   */
  void copied(PtStream stream) {
    if (stream == null) {
      return;
    }
    PtEvent event = events.computeIfAbsent(stream.getDevice(), device -> stream.newEvent());
    event.record();
    pending = event;
  }

  /**
   * 最後の写しの終わりを待ってから、event とバッファを返す。待てなければ（デバイスの失敗）、写しがまだ読むかもしれないバッファを解放せずに
   * 例外を投げる。
   */
  @Override
  public void close() {
    await();
    for (PtEvent event : events.values()) {
      event.close();
    }
    events.clear();
    buffers.clear();
    host.close();
  }
}
