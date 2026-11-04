package com.epsilon.reviewer;

import com.epsilon.reviewer.analysis.AnalysisJobs;
import com.epsilon.reviewer.analysis.AnalysisStarted;
import com.epsilon.reviewer.analysis.JobId;
import com.epsilon.reviewer.analysis.JobSnapshot;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.model.CatalogView;
import com.epsilon.reviewer.model.ModelCatalog;
import com.epsilon.reviewer.model.ModelKey;
import com.epsilon.reviewer.model.ModelRegistry;
import com.epsilon.reviewer.model.PolicyProviders;
import com.epsilon.reviewer.record.ImportedRecord;
import com.epsilon.reviewer.record.PreparedRecord;
import com.epsilon.reviewer.record.PreparedRecords;
import com.epsilon.reviewer.record.RecordId;
import com.epsilon.reviewer.record.RecordImporter;
import com.epsilon.reviewer.record.TenhouFetcher;
import com.epsilon.reviewer.scenario.ScenarioService;
import com.epsilon.reviewer.store.OwnerId;
import com.epsilon.reviewer.store.ResultId;
import com.epsilon.reviewer.store.ResultMeta;
import com.epsilon.reviewer.store.ResultStore;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 組み立ての根。サービスを作り、閉じる順を決める。1 つの操作 = 1 つのメソッド（api.Command と同じ並び）。 */
public final class Reviewer implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(Reviewer.class);

  /** 使わないモデルを閉じるまでの時間。 */
  static final Duration MODEL_IDLE = Duration.ofMinutes(10);

  /** 掃除の間隔（最初は 1 間隔の後）。 */
  static final Duration SWEEP_INTERVAL = Duration.ofMinutes(1);

  private final ResultStore store;
  private final ModelCatalog catalog;
  private final ModelRegistry models;
  private final PreparedRecords records = new PreparedRecords();
  private final TenhouFetcher tenhou = new TenhouFetcher();
  private final RecordImporter importer;
  private final AnalysisJobs analyses;
  private final ScenarioService scenarios;
  private final ScheduledExecutorService sweeper =
      Executors.newSingleThreadScheduledExecutor(
          task -> {
            var thread = new Thread(task, "reviewer-sweep");
            thread.setDaemon(true);
            return thread;
          });

  private Reviewer(
      ResultStore store, ModelCatalog catalog, PolicyProviders providers, int maxOpenModels) {
    this.store = store;
    this.catalog = catalog;
    models = new ModelRegistry(catalog, providers, maxOpenModels, MODEL_IDLE);
    importer = new RecordImporter(records, tenhou, store);
    analyses = new AnalysisJobs(records, models, store);
    scenarios = new ScenarioService(catalog, models, store);
    long interval = SWEEP_INTERVAL.toMillis();
    sweeper.scheduleWithFixedDelay(this::sweep, interval, interval, TimeUnit.MILLISECONDS);
  }

  /**
   * カタログを読み、保存先を開く（モデルは開かない）。
   *
   * @throws ReviewerException data_locked
   * @throws IOException models.json か保存先が読めない
   */
  public static Reviewer open(ReviewerConfig config) throws IOException {
    PolicyProviders providers = PolicyProviders.load();
    ModelCatalog catalog =
        config.models() == null
            ? ModelCatalog.empty()
            : ModelCatalog.load(config.models(), providers);
    ResultStore store = ResultStore.open(config.data());
    return new Reviewer(store, catalog, providers, config.maxOpenModels());
  }

  public CatalogView models() {
    return catalog.view();
  }

  public HistoryList history(OwnerId owner) {
    return new HistoryList(store.history(owner).stream().map(HistoryEntry::of).toList());
  }

  public ImportedRecord importBytes(OwnerId owner, byte[] bytes) {
    return importer.importBytes(owner, bytes);
  }

  public ImportedRecord importFile(OwnerId owner, Path path) {
    return importer.importFile(owner, path);
  }

  public ImportedRecord importTenhou(OwnerId owner, String url) {
    return importer.importTenhou(owner, url);
  }

  public ImportedRecord importStored(OwnerId owner, ResultId id) {
    return importer.importStored(owner, id);
  }

  /** 読み込んだ牌譜を取り、カタログで確かめて仕事にする。受け付けられなければ牌譜を戻す。 */
  public AnalysisStarted startAnalysis(OwnerId owner, RecordId record, ModelKey model) {
    PreparedRecord prepared = records.take(owner, record);
    try {
      return analyses.submit(owner, prepared, catalog.require(model));
    } catch (RuntimeException failure) {
      records.restore(prepared);
      throw failure;
    }
  }

  public JobSnapshot job(OwnerId owner, JobId id) {
    return analyses.snapshot(owner, id);
  }

  public JobSnapshot cancelJob(OwnerId owner, JobId id) {
    return analyses.cancel(owner, id);
  }

  public StoredDocument result(OwnerId owner, ResultId id) {
    return document(store.find(owner, id), id, ErrorCode.RESULT_NOT_FOUND);
  }

  /** 索引から外し、もしもの写しも外す。 */
  public void deleteResult(OwnerId owner, ResultId id) {
    try {
      store.delete(owner, id);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
    scenarios.forget(id);
  }

  public void exportResult(OwnerId owner, ResultId id, Path target) {
    try {
      store.export(owner, id, target);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  /** 持ち主の確かめだけ（共有リンクの URL は HTTP が作る）。 */
  public ResultId shareable(OwnerId owner, ResultId id) {
    store.find(owner, id);
    return id;
  }

  public StoredDocument sharedResult(ResultId id) {
    return document(store.findShared(id), id, ErrorCode.SHARE_NOT_FOUND);
  }

  public ScenarioAnswer scenario(OwnerId owner, ResultId id, ScenarioQuery query) {
    return scenarios.evaluate(store.find(owner, id), query);
  }

  public ScenarioAnswer sharedScenario(ResultId id, ScenarioQuery query) {
    return scenarios.evaluate(store.findShared(id), query);
  }

  /** 掃除（走っている回を待つ）→ 解析（取り消して待つ）→ モデル → 保存（鍵を放す）→ 天鳳の順に閉じる。 */
  @Override
  public void close() {
    sweeper.shutdownNow();
    try {
      if (!sweeper.awaitTermination(10, TimeUnit.SECONDS))
        LOG.warn("The sweep did not stop within 10 seconds.");
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
    }
    analyses.close();
    models.close();
    store.close();
    tenhou.close();
  }

  private StoredDocument document(ResultMeta meta, ResultId id, ErrorCode missing) {
    return new StoredDocument(
        id,
        () -> {
          try {
            return store.readDocument(meta);
          } catch (NoSuchFileException failure) {
            throw new ReviewerException(missing);
          } catch (IOException failure) {
            throw new UncheckedIOException(failure);
          }
        });
  }

  private void sweep() {
    List<Runnable> steps =
        List.of(records::sweep, analyses::sweep, models::sweep, scenarios::sweep, store::sweep);
    for (Runnable step : steps)
      try {
        step.run();
      } catch (RuntimeException failure) {
        LOG.error("Could not sweep the reviewer state.", failure);
      }
  }
}
