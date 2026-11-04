package com.epsilon.reviewer.scenario;

import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.load.RecordLoader;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.scenario.ScenarioEngine;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.model.ModelCatalog;
import com.epsilon.reviewer.model.ModelKey;
import com.epsilon.reviewer.model.ModelLease;
import com.epsilon.reviewer.model.ModelRegistry;
import com.epsilon.reviewer.store.ResultId;
import com.epsilon.reviewer.store.ResultMeta;
import com.epsilon.reviewer.store.ResultStore;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.NoSuchFileException;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;

/**
 * 保存した解析の 1 判断を、局面を変えて評価し直す（もしも）。走るのは 1 つ、待ちは 4 つまで（超えれば即 busy）、待ちは公平に最大 3 秒。結果ごとの
 * 読み込んだ牌譜は 8 件・10 分の LRU に持ち、削除で外す。モデルは解析と同じ貸し出しを使う。
 */
public final class ScenarioService {
  /** 走る 1 つと待ちの数の和。 */
  static final int ADMITTED = 5;

  static final Duration WAIT = Duration.ofSeconds(3);
  static final int CACHED_RECORDS = 8;
  static final Duration CACHE_LIFETIME = Duration.ofMinutes(10);

  private final ModelCatalog catalog;
  private final ModelRegistry models;
  private final ResultStore store;
  private final Semaphore admission = new Semaphore(ADMITTED);
  private final ReentrantLock running = new ReentrantLock(true);

  /** resultId → 読み込んだ牌譜（使った順）。records の監視で守る。 */
  private final Map<String, Cached> records = new LinkedHashMap<>(16, 0.75f, true);

  public ScenarioService(ModelCatalog catalog, ModelRegistry models, ResultStore store) {
    this.catalog = catalog;
    this.models = models;
    this.store = store;
  }

  /** @throws ReviewerException busy・model_unavailable・invalid_edit・decision_not_found */
  public ScenarioAnswer evaluate(ResultMeta meta, ScenarioQuery query) {
    ModelKey key = ModelKey.of(meta.model());
    catalog.require(key);
    if (!admission.tryAcquire()) throw new ReviewerException(ErrorCode.BUSY);
    try {
      if (!running.tryLock(WAIT.toMillis(), TimeUnit.MILLISECONDS))
        throw new ReviewerException(ErrorCode.BUSY);
      try (ModelLease lease = models.acquire(key)) {
        return ScenarioEngine.evaluate(record(meta), query, lease);
      } finally {
        running.unlock();
      }
    } catch (InterruptedException failure) {
      Thread.currentThread().interrupt();
      throw new ReviewerException(ErrorCode.SHUTTING_DOWN);
    } finally {
      admission.release();
    }
  }

  /** 削除した結果の写しを外す。 */
  public void forget(ResultId id) {
    synchronized (records) {
      records.remove(id.value());
    }
  }

  /** 10 分使わなかった写しを外す。 */
  public void sweep() {
    long limit = System.nanoTime() - CACHE_LIFETIME.toNanos();
    synchronized (records) {
      records.values().removeIf(cached -> cached.usedAt - limit < 0);
    }
  }

  private LoadedRecord record(ResultMeta meta) {
    long now = System.nanoTime();
    synchronized (records) {
      Cached cached = records.get(meta.resultId());
      if (cached != null) {
        cached.usedAt = now;
        return cached.record;
      }
    }
    LoadedRecord record;
    try {
      record = RecordLoader.reload(store.readRecord(meta), meta.record());
    } catch (NoSuchFileException failure) {
      throw new ReviewerException(ErrorCode.RESULT_NOT_FOUND);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    } catch (IllegalArgumentException failure) {
      throw new ReviewerException(
          ErrorCode.INVALID_RECORD,
          "The stored record can no longer be read: " + failure.getMessage());
    }
    synchronized (records) {
      records.put(meta.resultId(), new Cached(record, now));
      if (records.size() > CACHED_RECORDS) records.remove(records.keySet().iterator().next());
    }
    return record;
  }

  private static final class Cached {
    final LoadedRecord record;
    long usedAt;

    Cached(LoadedRecord record, long usedAt) {
      this.record = record;
      this.usedAt = usedAt;
    }
  }
}
