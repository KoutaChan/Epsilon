package com.epsilon.reviewer.model;

import com.epsilon.agent.checkpoint.CheckpointFiles;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.locks.ReentrantLock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * モデルのバージョン（名前と revision の組）ごとに方策を 1 つだけ開き、解析ともしもで共有する。チェックポイントはコピーしない。
 * 評価は、モデルごとの公平なロックで 1 つずつ行う。解析は判断ごとにロックを取るので、もしもが待つのは長くても解析の 1 判断の分になる。
 *
 * <p>開くときは、起動時のフィンガープリントと比べる。models.json に sha256 があれば、初めて開くときに 1 回だけ中身のハッシュも比べる。
 * 違えば、以後そのモデルを使えなくする。同時に開いておくのは {@code maxOpen} 個まで。超えたら、貸し出していないモデルを、最後に
 * 使ってからの時間が長い順に閉じる。閉じられるモデルが無ければ、後の返却や {@link #sweep} のときに減らす。{@code idle} より長く
 * 使われなかったモデルは閉じる。
 */
public final class ModelRegistry implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(ModelRegistry.class);

  private final ModelCatalog catalog;
  private final ModelFamilies families;
  private final int maxOpen;
  private final long idleNanos;

  /** 開いたモデルと、開いている途中のモデル。読み書きは this で同期する。 */
  private final Map<ModelKey, OpenModel> open = new HashMap<>();

  /** sha256 の比較を済ませたモデル。 */
  private final Set<ModelKey> verified = ConcurrentHashMap.newKeySet();

  private boolean closed;

  public ModelRegistry(
      ModelCatalog catalog, ModelFamilies families, int maxOpen, Duration idle) {
    this.catalog = catalog;
    this.families = families;
    this.maxOpen = maxOpen;
    idleNanos = idle.toNanos();
  }

  /**
   * モデルを借りる。開いていなければ開く。同じモデルを同時に借りようとしたときは、最初に来た呼び出しが開き、ほかはそれを待つ。
   *
   * @param key 借りるモデルのバージョン
   * @return 貸し出し。使い終わったら閉じる
   * @throws ReviewerException model_unavailable か shutting_down
   */
  public ModelLease acquire(ModelKey key) {
    OpenModel model;
    boolean opener = false;
    synchronized (this) {
      if (closed) throw new ReviewerException(ErrorCode.SHUTTING_DOWN);
      model = open.get(key);
      if (model == null) {
        model = new OpenModel(catalog.require(key));
        open.put(key, model);
        opener = true;
      }
      model.leases++;
    }
    try {
      if (opener) model.policy.complete(openPolicy(model.entry));
      else awaitOpen(model);
    } catch (RuntimeException | Error failure) {
      synchronized (this) {
        model.leases--;
        if (opener) open.remove(key, model);
      }
      if (opener) model.policy.completeExceptionally(failure);
      throw failure;
    }
    trim();
    return new ModelLease(this, model);
  }

  /** {@code idle} より長く使われていないモデルを閉じ、開きすぎた分を減らす。 */
  public void sweep() {
    long now = System.nanoTime();
    List<OpenModel> idle = new ArrayList<>();
    synchronized (this) {
      open.values()
          .removeIf(
              model -> {
                boolean expired =
                    model.leases == 0 && model.policy.isDone() && now - model.lastUsed > idleNanos;
                if (expired) idle.add(model);
                return expired;
              });
    }
    idle.forEach(ModelRegistry::closePolicy);
    trim();
  }

  @Override
  public void close() {
    List<OpenModel> models;
    synchronized (this) {
      closed = true;
      models = new ArrayList<>(open.values());
      open.clear();
    }
    models.forEach(ModelRegistry::closePolicy);
  }

  /** 貸し出しを返し、開きすぎた分を減らす。 */
  void release(OpenModel model) {
    synchronized (this) {
      model.leases--;
      model.lastUsed = System.nanoTime();
    }
    trim();
  }

  private void awaitOpen(OpenModel model) {
    try {
      model.policy.get();
    } catch (InterruptedException failure) {
      Thread.currentThread().interrupt();
      throw new ReviewerException(ErrorCode.SHUTTING_DOWN);
    } catch (ExecutionException failure) {
      throw new ReviewerException(ErrorCode.MODEL_UNAVAILABLE);
    }
  }

  private Policy openPolicy(ModelEntry entry) {
    if (entry.checkpoint() != null) verifyCheckpoint(entry);
    ModelFamily family =
        families
            .find(entry.series())
            .orElseThrow(() -> new ReviewerException(ErrorCode.MODEL_UNAVAILABLE));
    try {
      long started = System.nanoTime();
      Policy policy = family.open(entry.checkpoint(), entry.variant(), entry.options());
      LOG.info(
          "Opened model {} {} in {} ms.",
          entry.key().modelId(),
          entry.key().revision(),
          (System.nanoTime() - started) / 1_000_000);
      return policy;
    } catch (IOException | RuntimeException | LinkageError failure) {
      LOG.warn(
          "Could not open the model {} {}.",
          entry.key().modelId(),
          entry.key().revision(),
          failure);
      throw new ReviewerException(ErrorCode.MODEL_UNAVAILABLE);
    }
  }

  /** チェックポイントを起動時のフィンガープリントと比べる。sha256 があれば、初めて開くときに 1 回だけ中身のハッシュも比べる。 */
  private void verifyCheckpoint(ModelEntry entry) {
    try {
      if (!CheckpointFingerprint.of(entry.checkpoint()).equals(catalog.fingerprint(entry.key())))
        throw changed(entry);
      if (entry.sha256() != null && !verified.contains(entry.key())) {
        if (!CheckpointFiles.sha256(entry.checkpoint()).equals(entry.sha256()))
          throw changed(entry);
        verified.add(entry.key());
      }
    } catch (IOException | UncheckedIOException failure) {
      LOG.warn(
          "Could not read the checkpoint of {} {}: {}",
          entry.key().modelId(),
          entry.key().revision(),
          failure.toString());
      throw new ReviewerException(ErrorCode.MODEL_UNAVAILABLE);
    }
  }

  private ReviewerException changed(ModelEntry entry) {
    catalog.markChanged(entry.key());
    return new ReviewerException(
        ErrorCode.MODEL_UNAVAILABLE, "The checkpoint changed after it was registered.");
  }

  /** 開いたモデルが {@code maxOpen} 個を超えたら、貸し出していないモデルを、最後に使ってからの時間が長い順に閉じる。 */
  private void trim() {
    List<OpenModel> closing = new ArrayList<>();
    synchronized (this) {
      while (open.size() > maxOpen) {
        OpenModel oldest = null;
        for (OpenModel model : open.values())
          if (model.leases == 0
              && model.policy.isDone()
              && (oldest == null || model.lastUsed < oldest.lastUsed)) oldest = model;
        if (oldest == null) break;
        open.remove(oldest.entry.key());
        closing.add(oldest);
      }
    }
    closing.forEach(ModelRegistry::closePolicy);
  }

  private static void closePolicy(OpenModel model) {
    if (model.policy.state() != CompletableFuture.State.SUCCESS) return;
    try {
      model.policy.resultNow().close();
      LOG.info("Closed model {} {}.", model.entry.key().modelId(), model.entry.key().revision());
    } catch (RuntimeException failure) {
      LOG.warn("Could not close the model {}.", model.entry.key().modelId(), failure);
    }
  }

  /** 開いたモデル 1 つ。{@code leases} と {@code lastUsed} の読み書きは、レジストリで同期する。 */
  static final class OpenModel {
    final ModelEntry entry;
    final ReentrantLock evaluation = new ReentrantLock(true);
    final CompletableFuture<Policy> policy = new CompletableFuture<>();
    int leases;
    long lastUsed = System.nanoTime();

    OpenModel(ModelEntry entry) {
      this.entry = entry;
    }
  }
}
