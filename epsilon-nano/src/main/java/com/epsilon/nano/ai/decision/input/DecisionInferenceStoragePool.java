package com.epsilon.nano.ai.decision.input;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 推論入力の配列を、所有者が明示的に返すまで貸し出す。
 *
 * <p>保持する未使用領域だけを容量と個数で制限する。使用中の入力数は、呼び出し側の受付制御で制限する。 通常のビルダーや復号済み入力が所有する配列は取り込まない。
 */
public final class DecisionInferenceStoragePool implements AutoCloseable {
  private final long maxRetainedBytes;
  private final int maxRetainedBuffers;
  private final ArrayDeque<DecisionHostInputs.InferenceStorage> available = new ArrayDeque<>();
  private long retainedBytes;
  private int borrowedBuffers;
  private boolean closed;

  public DecisionInferenceStoragePool(long maxRetainedBytes, int maxRetainedBuffers) {
    if (maxRetainedBytes < 0 || maxRetainedBuffers < 0)
      throw new IllegalArgumentException("Storage retention limits must be nonnegative");
    this.maxRetainedBytes = maxRetainedBytes;
    this.maxRetainedBuffers = maxRetainedBuffers;
  }

  /** 一つの要求に含まれる全容量区分の入力を、まとめて所有する借用単位を作る。 */
  public synchronized Lease acquire() {
    requireOpen();
    return new Lease(this);
  }

  private synchronized DecisionHostInputs.InferenceStorage borrow() {
    requireOpen();
    var storage = available.pollFirst();
    if (storage == null) storage = new DecisionHostInputs.InferenceStorage();
    else retainedBytes -= storage.byteCount();
    borrowedBuffers++;
    return storage;
  }

  private synchronized void release(DecisionHostInputs.InferenceStorage storage) {
    borrowedBuffers--;
    long bytes = storage.byteCount();
    if (!closed
        && available.size() < maxRetainedBuffers
        && bytes <= maxRetainedBytes - retainedBytes) {
      available.addLast(storage);
      retainedBytes += bytes;
    }
  }

  private void requireOpen() {
    if (closed) throw new IllegalStateException("Inference storage pool is closed");
  }

  /** 貸出数と未使用配列の保持容量を返す。容量は管理上の配列サイズで、JVMの実測ヒープ量ではない。 */
  public synchronized Snapshot snapshot() {
    return new Snapshot(borrowedBuffers, available.size(), retainedBytes);
  }

  public record Snapshot(int borrowedBuffers, int retainedBuffers, long retainedBytes) {}

  /** 新規貸出を止め、未使用領域を解放する。使用中の領域は、返却された時点で保持せず解放する。 */
  @Override
  public synchronized void close() {
    closed = true;
    available.clear();
    retainedBytes = 0;
  }

  /**
   * 同じ要求の入力配列を所有する。
   *
   * <p>ビルダーと構築済みバッチ、およびその行ビューは、この借用単位を閉じるまで有効。 符号化が失敗した場合も、全ワーカー・保存処理・デバイス転送が入力を使い終えてから閉じる。
   */
  public static final class Lease implements AutoCloseable {
    private final DecisionInferenceStoragePool pool;
    private final List<DecisionHostInputs.InferenceStorage> borrowed = new ArrayList<>();
    private boolean closed;

    private Lease(DecisionInferenceStoragePool pool) {
      this.pool = pool;
    }

    /** 指定行数の入力を直接構築するビルダーを作る。異なるビルダーは配列を共有しない。 */
    public synchronized DecisionBatchBuilder inference(
        int rows, DecisionBucket bucket, DecisionBatchBuilder.EncodingWorkspacePool workspaces) {
      if (closed) throw new IllegalStateException("Inference storage lease is closed");
      Objects.requireNonNull(bucket, "bucket");
      Objects.requireNonNull(workspaces, "workspaces");
      var storage = pool.borrow();
      try {
        var builder =
            new DecisionBatchBuilder(
                DecisionHostBatch.inference(rows, bucket, storage), workspaces);
        borrowed.add(storage);
        return builder;
      } catch (RuntimeException | Error failure) {
        pool.release(storage);
        throw failure;
      }
    }

    /** 全入力の読み書きが終了した後に呼び、借用領域を一度だけ返す。 */
    @Override
    public synchronized void close() {
      if (closed) return;
      closed = true;
      for (var storage : borrowed) pool.release(storage);
      borrowed.clear();
    }
  }
}
