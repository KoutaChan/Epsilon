package com.epsilon.reviewer.model;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.spi.ReviewPolicy;
import com.epsilon.spi.ReviewPolicyProvider;
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
 * (modelId, revision) ごとに 1 つの方策を開き、解析ともしもが共有する。チェックポイントは写さない。評価はモデルごとの公平な鍵で 1 つずつ
 * 行う（解析は判断ごとに鍵を取るので、もしもは解析の 1 判断より長く待たない）。
 *
 * <p>開く時は起動時の印と比べ、models.json の sha256 があれば初めて開く時に 1 回だけ中身のハッシュも比べる。違えば以後そのモデルを使えなく
 * する。同時に開くのは maxOpen まで（超えれば貸していない一番古いものを閉じ、閉じられなければ掃除が減らす）。idle の間使わなければ閉じる。
 */
public final class ModelRegistry implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(ModelRegistry.class);

  private final ModelCatalog catalog;
  private final PolicyProviders providers;
  private final int maxOpen;
  private final long idleNanos;

  /** 開いた・開いている途中のモデル。this の監視で守る。 */
  private final Map<ModelKey, OpenModel> open = new HashMap<>();

  /** sha256 の比べを済ませたモデル。 */
  private final Set<ModelKey> verified = ConcurrentHashMap.newKeySet();

  private boolean closed;

  public ModelRegistry(
      ModelCatalog catalog, PolicyProviders providers, int maxOpen, Duration idle) {
    this.catalog = catalog;
    this.providers = providers;
    this.maxOpen = maxOpen;
    idleNanos = idle.toNanos();
  }

  /**
   * モデルを借りる。開いていなければ開く（同じ鍵の同時の開きは 1 回を待つ）。
   *
   * @throws ReviewerException model_unavailable・shutting_down
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

  /** 遊休のモデルを閉じ、開きすぎを減らす。 */
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

  /** 貸し出しを返し、開きすぎを減らす。 */
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

  private ReviewPolicy openPolicy(ModelEntry entry) {
    try {
      if (!CheckpointFingerprint.of(entry.checkpoint()).equals(catalog.fingerprint(entry.key())))
        throw changed(entry);
      if (entry.sha256() != null && !verified.contains(entry.key())) {
        if (!CheckpointDigest.of(entry.checkpoint()).equals(entry.sha256())) throw changed(entry);
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
    ReviewPolicyProvider provider =
        providers
            .find(entry.series())
            .orElseThrow(() -> new ReviewerException(ErrorCode.MODEL_UNAVAILABLE));
    try {
      long started = System.nanoTime();
      ReviewPolicy policy = provider.openReview(entry.checkpoint(), entry.options());
      LOG.info(
          "Opened model {} {} in {} ms.",
          entry.key().modelId(),
          entry.key().revision(),
          (System.nanoTime() - started) / 1_000_000);
      return policy;
    } catch (RuntimeException | LinkageError failure) {
      LOG.warn(
          "Could not open the model {} {}.",
          entry.key().modelId(),
          entry.key().revision(),
          failure);
      throw new ReviewerException(ErrorCode.MODEL_UNAVAILABLE);
    }
  }

  private ReviewerException changed(ModelEntry entry) {
    catalog.markChanged(entry.key());
    return new ReviewerException(
        ErrorCode.MODEL_UNAVAILABLE, "The checkpoint changed after it was registered.");
  }

  /** maxOpen を超えた分を、貸していない一番古いものから閉じる。 */
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

  /** 開いた 1 モデル。leases と lastUsed は registry の監視で守る。 */
  static final class OpenModel {
    final ModelEntry entry;
    final ReentrantLock evaluation = new ReentrantLock(true);
    final CompletableFuture<ReviewPolicy> policy = new CompletableFuture<>();
    int leases;
    long lastUsed = System.nanoTime();

    OpenModel(ModelEntry entry) {
      this.entry = entry;
    }
  }
}
