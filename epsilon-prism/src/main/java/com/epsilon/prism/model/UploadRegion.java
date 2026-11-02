package com.epsilon.prism.model;

import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtPinnedBuffer;
import ai.djl.pytorch.engine.PtTransferTicket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.EnumMap;
import java.util.Map;

/**
 * 前もって並べた区画の組を非同期でデバイスへ写すための、ページ固定の領域（{@link Staging#write}・{@link Staging#enqueue}）。型ごとに
 * ページ固定のバッファを 1 本持ち、足りなければ取り直して使い回す。
 *
 * <p>写し元の寿命の約束: 非同期の写し（DMA）は積んだ後もバッファを読むので、最後の写しの切符（{@link #copied}。写しのストリームで
 * 写しの後に記録した event）が終わるまで、どのバッファも書き換えず、解放もしない。書く前（{@link Staging#write}）・取り直す前・閉じる
 * 前に {@link #await} で待つ。写しを読む計算ストリームの進みには依らない（写し元を読むのは写しのストリームだけ）。写し元は全て領域が
 * 持つ固定のメモリで、写しのためにその場で固定する一時のホストのメモリは使わない。
 *
 * <p>一度に 1 つの組だけが使い、使う手番（書く・写しを積む・待つ）は呼び手が happens-before でつなぐ（学習の歩の作り手が書き、レーンが
 * 積み、歩の後に返された領域を次の書き手が待つ）。
 */
public final class UploadRegion implements AutoCloseable {

  private final PtNDManager host;
  private final Map<Staging.Kind, PtPinnedBuffer> buffers = new EnumMap<>(Staging.Kind.class);

  /** 最後に積んだ写しの切符（終わりを待って閉じた後、または写しを積んでいなければ null）。 */
  private PtTransferTicket pending;

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

  /** 最後に積んだ写しが終わるまで待ち、その切符を閉じる（積んでいなければ何もしない）。 */
  void await() {
    if (pending != null) {
      pending.synchronize();
      pending.close();
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
   * 型ごとの写しを積んで締めた切符を預かる（GPU の写しだけ。CPU の写しは同期で終わるので預けない）。{@link Staging#write} の
   * {@link #await} の後だけ呼ぶので、前の切符は閉じてある。切符は次の {@link #await} が閉じる。
   *
   * @param ticket この領域のバッファから写す全ての写しの切符
   */
  void copied(PtTransferTicket ticket) {
    pending = ticket;
  }

  /**
   * 最後の写しの終わりを待ってから、切符とバッファを返す。待てなければ（デバイスの失敗）、写しがまだ読むかもしれないバッファを解放せずに
   * 例外を投げる。
   */
  @Override
  public void close() {
    await();
    buffers.clear();
    host.close();
  }
}
