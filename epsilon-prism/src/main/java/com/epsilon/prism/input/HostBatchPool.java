package com.epsilon.prism.input;

import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.function.IntFunction;

/**
 * {@link HostBatch} の池。blob は作るときに一度だけ取り、塊を返すと次の借り手が使い回す。借りるのは符号化の前（受付）で、返すのは scores を
 * 読み終えた後にする。
 *
 * <p>blob の取り方は外から渡す（{@link ByteBuffer#allocateDirect} か、ページ固定の領域を返す関数）。借りると返すは別々のスレッドから呼んで
 * よい。
 */
public final class HostBatchPool {

  private final HostBatch[] members;
  private final ArrayBlockingQueue<HostBatch> free;

  /**
   * ダイレクトの領域で池を作る。
   *
   * @param capacity 1 塊の上限
   * @param batches 塊の数
   */
  public HostBatchPool(BatchPlan.Capacity capacity, int batches) {
    this(capacity, batches, ByteBuffer::allocateDirect);
  }

  /**
   * 池を作る。
   *
   * @param capacity 1 塊の上限
   * @param batches 塊の数
   * @param allocator バイト数を受けて blob の領域を返す関数（作るときにだけ呼ぶ）
   * @throws IllegalArgumentException 塊の数が 1 未満のとき
   */
  public HostBatchPool(
      BatchPlan.Capacity capacity, int batches, IntFunction<ByteBuffer> allocator) {
    if (batches < 1) {
      throw new IllegalArgumentException("pool needs at least one batch: " + batches);
    }
    Objects.requireNonNull(allocator, "allocator");
    int bytes = BatchPlan.maxBytes(capacity);
    members = new HostBatch[batches];
    free = new ArrayBlockingQueue<>(batches);
    for (int index = 0; index < batches; index++) {
      members[index] = new HostBatch(capacity, allocator.apply(bytes));
      free.add(members[index]);
    }
  }

  /**
   * 空いた塊を借りる。空きがなければ返されるまで待つ。借りた塊の計画は空にしてある。
   *
   * @return 塊
   * @throws InterruptedException 待つ間に割り込まれたとき
   */
  public HostBatch acquire() throws InterruptedException {
    HostBatch batch = free.take();
    batch.plan().clear();
    return batch;
  }

  /**
   * 塊を返す。
   *
   * @param batch この池から借りた塊
   * @throws IllegalArgumentException この池の塊でないか、もう返した塊のとき
   */
  public void release(HostBatch batch) {
    if (!member(batch) || free.contains(batch)) {
      throw new IllegalArgumentException("batch is not on loan from this pool");
    }
    free.add(batch);
  }

  /**
   * 空いている塊の数を返す。
   *
   * @return 塊の数
   */
  public int available() {
    return free.size();
  }

  private boolean member(HostBatch batch) {
    for (HostBatch member : members) {
      if (member == batch) {
        return true;
      }
    }
    return false;
  }
}
