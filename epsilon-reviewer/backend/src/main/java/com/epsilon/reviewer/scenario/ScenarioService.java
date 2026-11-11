package com.epsilon.reviewer.scenario;

import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.load.RecordLoader;
import com.epsilon.reviewer.engine.scenario.DecisionLocator;
import com.epsilon.reviewer.engine.scenario.LocatedDecision;
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
 * 保存した解析から判断を 1 つ選び、局面を変えて評価し直す（もしも）。同時に評価するのは 1 件だけで、待てるのは 4 件まで。それより
 * 多い要求はすぐに busy で断る。待つ順は公平で、3 秒待っても順番が来なければ busy で断る。
 *
 * <p>モデルは、解析したときのバージョンが使えればそれを使い、使えなければ同じモデルの今のバージョン（models.json の最後のバージョン）を
 * 使う。モデルの借り方は解析と同じ。判断は、まず文書のインデックスで、その局でその席が何番目に行った判断かに読み替える。次に牌譜を
 * 再生してその判断を探し、候補が文書と同じかを確かめる。前のバージョンのエンジンが書いた文書も、同じ方法で読む。結果ごとの牌譜と
 * 文書のインデックスは LRU のキャッシュに持ち（最大 8 件）、10 分使われないか結果を削除したら外す。牌譜を読み直すときも、全体を再生して確かめる。
 */
public final class ScenarioService {
  /** 受け付ける件数の上限。評価中の 1 件と、待っている 4 件の合計。 */
  static final int ADMITTED = 5;

  static final Duration WAIT = Duration.ofSeconds(3);
  static final int CACHED_RESULTS = 8;
  static final Duration CACHE_LIFETIME = Duration.ofMinutes(10);

  private final ModelCatalog catalog;
  private final ModelRegistry models;
  private final ResultStore store;
  private final Semaphore admission = new Semaphore(ADMITTED);
  private final ReentrantLock running = new ReentrantLock(true);

  /** 結果の id から、読み込んだ牌譜と文書のインデックスを引く表。使った順に並ぶ。読み書きは、この表で synchronized して守る。 */
  private final Map<String, Cached> results = new LinkedHashMap<>(16, 0.75f, true);

  public ScenarioService(ModelCatalog catalog, ModelRegistry models, ResultStore store) {
    this.catalog = catalog;
    this.models = models;
    this.store = store;
  }

  /**
   * @throws ReviewerException busy・model_unavailable・invalid_request・invalid_record・invalid_edit・
   *     decision_not_found・result_not_found
   */
  public ScenarioAnswer evaluate(ResultMeta meta, ScenarioQuery query) {
    ModelKey key = catalog.scenarioModel(ModelKey.of(meta.model()));
    catalog.require(key);
    if (!admission.tryAcquire()) throw new ReviewerException(ErrorCode.BUSY);
    try {
      if (!running.tryLock(WAIT.toMillis(), TimeUnit.MILLISECONDS))
        throw new ReviewerException(ErrorCode.BUSY);
      try {
        Cached cached = cached(meta);
        LocatedDecision decision = locate(cached, query);
        try (ModelLease lease = models.acquire(key)) {
          return ScenarioEngine.evaluate(
              cached.record, decision, query.edits(), query.call(), lease);
        }
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

  /** 削除した結果のキャッシュを外す。 */
  public void forget(ResultId id) {
    synchronized (results) {
      results.remove(id.value());
    }
  }

  /** 10 分使われなかったキャッシュを外す。 */
  public void sweep() {
    long limit = System.nanoTime() - CACHE_LIFETIME.toNanos();
    synchronized (results) {
      results.values().removeIf(cached -> cached.usedAt - limit < 0);
    }
  }

  /** もしもの座標で指した判断を探す。文書のインデックスで何番目の判断かに読み替えてから、牌譜の中で探す。 */
  private static LocatedDecision locate(Cached cached, ScenarioQuery query) {
    ScenarioEngine.requireSeat(cached.record, query.seat());
    DocumentDecisions.Stored stored =
        cached.decisions.find(query.round(), query.seat(), query.eventIndex());
    if (stored == null) throw new ReviewerException(ErrorCode.DECISION_NOT_FOUND);
    return DecisionLocator.findNth(
        cached.record, query.round(), query.seat(), stored.ordinal(), stored.candidates());
  }

  private Cached cached(ResultMeta meta) {
    long now = System.nanoTime();
    synchronized (results) {
      Cached cached = results.get(meta.resultId());
      if (cached != null) {
        cached.usedAt = now;
        return cached;
      }
    }
    byte[] recordBytes;
    byte[] document;
    try {
      recordBytes = store.readRecord(meta);
      document = store.readDocument(meta);
    } catch (NoSuchFileException failure) {
      throw new ReviewerException(ErrorCode.RESULT_NOT_FOUND);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
    LoadedRecord record;
    try {
      record = RecordLoader.load(recordBytes);
    } catch (IllegalArgumentException failure) {
      throw new ReviewerException(
          ErrorCode.INVALID_RECORD,
          "The stored record can no longer be read: " + failure.getMessage());
    }
    DocumentDecisions decisions;
    try {
      decisions = DocumentDecisions.read(document, record.game().variant().seats());
    } catch (IOException | RuntimeException failure) {
      throw new ReviewerException(
          ErrorCode.INVALID_RECORD,
          "The stored result can no longer be read: " + failure.getMessage());
    }
    var cached = new Cached(record, decisions, now);
    synchronized (results) {
      results.put(meta.resultId(), cached);
      if (results.size() > CACHED_RESULTS) results.remove(results.keySet().iterator().next());
    }
    return cached;
  }

  private static final class Cached {
    final LoadedRecord record;
    final DocumentDecisions decisions;
    long usedAt;

    Cached(LoadedRecord record, DocumentDecisions decisions, long usedAt) {
      this.record = record;
      this.decisions = decisions;
      this.usedAt = usedAt;
    }
  }
}
