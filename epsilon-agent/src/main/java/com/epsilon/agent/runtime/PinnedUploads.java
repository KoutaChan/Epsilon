package com.epsilon.agent.runtime;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtPinnedBuffer;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtTransferBatch;
import ai.djl.pytorch.engine.PtTransferTicket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.function.Consumer;

/**
 * ホストの値をデバイスへ転送するための、ページ固定（pinned）のバッファ。データの型ごとにバッファを 1 本持ち、足りなくなったら大きく
 * 取り直して使い回す。転送のたびに使い捨てのホストのバッファを取らないので、デバイスへの転送はいつも、ここで固定したメモリからの DMA に
 * なる（転送のためにその場で一時のメモリを固定することはない）。GPU の見えない PyTorch では、バッファはふつうのホストのメモリになる。
 *
 * <p>転送の仕方は 2 つある。
 *
 * <ul>
 *   <li>{@link #send}: 型のバッファに書いて、同期で転送する。返ったときには転送を終えているので、バッファはすぐ次に使える。
 *   <li>{@link #write} と {@link #enqueue}: 型ごとのバッファに先に書いておき、後でストリームに非同期の転送を積む（書いた型ごとに 1 回）。
 * </ul>
 *
 * <p>非同期の転送は、積んだ後もバッファを読む。そのため、最後に積んだ転送が終わるまでは、どのバッファも書き換えず、取り直さず、解放も
 * しない。書く前と閉じる前には、転送の終わりを待つ（{@link #await}）。終わりは、転送のストリームに記録したイベント（転送のチケット）で
 * 知る。転送先を読む計算のストリームがどこまで進んだかには依らない。
 *
 * <p>スレッド安全ではない。一度に 1 本のスレッドが使い、使う順（書く・積む・待つ）は呼び出し元が happens-before でつなぐ。バッファは、
 * 作るときに渡したマネージャーの子に付けるので、{@link #close} か、そのマネージャーを閉じると返る。
 */
public final class PinnedUploads implements AutoCloseable {

  private static final DataType[] TYPES = DataType.values();

  private final PtNDManager host;

  /** 型ごとのバッファ（まだ使っていない型は null）。 */
  private final PtPinnedBuffer[] buffers = new PtPinnedBuffer[TYPES.length];

  /** 非同期の転送のために書いて、まだ転送を積んでいない要素の数（型ごと。書いていない型は 0）。 */
  private final int[] written = new int[TYPES.length];

  /** 転送をまだ積んでいない書き込みがあれば true。 */
  private boolean writing;

  /** 最後に積んだ転送のチケット。転送を積んでいないか、終わりを待って閉じた後は null。 */
  private PtTransferTicket pending;

  private PinnedUploads(PtNDManager host) {
    this.host = host;
  }

  /**
   * 転送用のバッファの組を作る。バッファそのものは、初めて使うときに取る。
   *
   * @param owner バッファを付けるマネージャー（PyTorch のもの。デバイスは問わない）
   * @return 作ったバッファの組
   */
  public static PinnedUploads on(NDManager owner) {
    return new PinnedUploads((PtNDManager) owner.newSubManager());
  }

  /**
   * 型のバッファに値を書き、マネージャーのデバイスに作った新しい配列へ同期で転送する。前に積んだ非同期の転送があれば、先にその終わりを
   * 待つ。
   *
   * @param manager 配列を作るマネージャー
   * @param type 型
   * @param shape 配列の形（要素は 1 個以上）
   * @param writer バッファに、形の要素の数だけ値を書く処理（位置 0 から、ネイティブのバイト順）
   * @return 転送した配列
   * @throws IllegalStateException 非同期の転送のために書いて、まだ積んでいないバッファがあるとき
   */
  public NDArray send(NDManager manager, DataType type, Shape shape, Consumer<ByteBuffer> writer) {
    if (writing) {
      throw new IllegalStateException("written buffers are waiting to be enqueued");
    }
    PtPinnedBuffer buffer = buffer(type, Math.toIntExact(shape.size()));
    writer.accept(view(buffer));
    PtNDArray array = (PtNDArray) manager.create(shape, type);
    array.copyFromPinnedBuffer(buffer);
    return array;
  }

  /**
   * 非同期の転送のために、型のバッファを返す。前に積んだ転送があれば、先にその終わりを待つ。転送は {@link #enqueue} が積む。転送する
   * 要素の数は、型ごとに最後に書いたときの数。
   *
   * @param type 型
   * @param elements 書く要素の数（1 以上）
   * @return バッファ（位置 0 から、ネイティブのバイト順。少なくとも要素の数だけ書ける）
   */
  public ByteBuffer write(DataType type, int elements) {
    PtPinnedBuffer buffer = buffer(type, elements);
    written[type.ordinal()] = elements;
    writing = true;
    return view(buffer);
  }

  /**
   * 書いた型ごとに 1 回ずつ転送を積み、転送先の配列 [書いた要素の数] を返す。GPU では転送をストリームに非同期で積み、そのチケットを
   * 預かる（次に書く前と、閉じる前に終わりを待つ）。CPU では同期で転送する。送り直すときは {@link #write} から書き直す。
   *
   * @param manager 転送先の配列を作るマネージャー
   * @param stream 転送を積むストリーム（CPU のマネージャーなら null）
   * @return 積んだ転送
   * @throws IllegalStateException 書いたバッファがないとき
   */
  public Transfer enqueue(NDManager manager, PtStream stream) {
    if (!writing) {
      throw new IllegalStateException("no buffer has been written to enqueue");
    }
    writing = false;
    NDArray[] arrays = new NDArray[TYPES.length];
    try {
      if (stream == null) {
        for (DataType type : TYPES) {
          int elements = written[type.ordinal()];
          if (elements > 0) {
            PtNDArray array = (PtNDArray) manager.create(new Shape(elements), type);
            array.enqueueCopyFrom(buffers[type.ordinal()]);
            arrays[type.ordinal()] = array;
          }
        }
        return new Transfer(arrays, null);
      }
      PtTransferTicket ticket;
      // コミットする前に失敗したら、バッチを閉じる処理が、積んだ転送の終わりを待ってから転送先を返す
      try (PtTransferBatch batch = stream.newTransferBatch((PtNDManager) manager)) {
        for (DataType type : TYPES) {
          int elements = written[type.ordinal()];
          if (elements > 0) {
            arrays[type.ordinal()] = batch.copy(buffers[type.ordinal()], new Shape(elements));
          }
        }
        ticket = batch.commit();
      }
      pending = ticket;
      return new Transfer(arrays, ticket);
    } finally {
      Arrays.fill(written, 0);
    }
  }

  /** 最後に積んだ転送が終わるまで待ち、そのチケットを閉じる。転送を積んでいなければ何もしない。 */
  public void await() {
    if (pending != null) {
      pending.synchronize();
      pending.close();
      pending = null;
    }
  }

  /**
   * 最後の転送の終わりを待ってから、チケットとバッファを返す。デバイスの失敗で待てなければ例外を投げ、転送がまだ読むかもしれない
   * バッファは解放しない。
   */
  @Override
  public void close() {
    await();
    Arrays.fill(buffers, null);
    host.close();
  }

  /** 前の転送の終わりを待ってから、型のバッファを、渡した要素の数以上の大きさにして返す。取り直すときは、古いバッファを解放する。 */
  private PtPinnedBuffer buffer(DataType type, int elements) {
    await();
    PtPinnedBuffer buffer = buffers[type.ordinal()];
    if (buffer == null || buffer.size() < elements) {
      if (buffer != null) {
        buffer.close();
      }
      buffer = host.allocatePinned(elements, type);
      buffers[type.ordinal()] = buffer;
    }
    return buffer;
  }

  private static ByteBuffer view(PtPinnedBuffer buffer) {
    return buffer.getByteBuffer().duplicate().order(ByteOrder.nativeOrder());
  }

  /** 積んだ転送。転送先の配列は、読むストリームで {@link #waitOnStream} を呼んだ後に、そのストリームの上でだけ読む。 */
  public static final class Transfer {

    /** 型ごとの転送先（書いていない型は null）。 */
    private final NDArray[] arrays;

    /** 転送のチケット（CPU の同期の転送では null）。 */
    private final PtTransferTicket ticket;

    private Transfer(NDArray[] arrays, PtTransferTicket ticket) {
      this.arrays = arrays;
      this.ticket = ticket;
    }

    /**
     * 型の転送先の配列を返す。
     *
     * @param type 型
     * @return 配列 [書いた要素の数]
     * @throws IllegalArgumentException その型を書いていなかったとき
     */
    public NDArray array(DataType type) {
      NDArray array = arrays[type.ordinal()];
      if (array == null) {
        throw new IllegalArgumentException("no " + type + " buffer was written for this transfer");
      }
      return array;
    }

    /**
     * 呼び出したスレッドが今使っているストリームに、転送の終わりを待つ処理を積む（ホストは待たない）。あわせて、転送先の配列をそのストリームが
     * 使うものとして記録する。転送先のメモリは転送のストリームの上で取るので、この記録がないと、マネージャーを閉じた時点でメモリが転送の
     * ストリームに返る。すると、読む側のストリームがまだ読んでいる間に、転送のストリームの次の確保がそのメモリを書き換えうる。CPU の
     * 転送は同期で終わっているので、何もしない。
     */
    public void waitOnStream() {
      if (ticket != null) {
        ticket.waitOnStream();
      }
    }
  }
}
