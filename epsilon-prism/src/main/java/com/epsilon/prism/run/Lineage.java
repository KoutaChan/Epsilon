package com.epsilon.prism.run;

import static com.epsilon.prism.run.Snapshot.deleteTree;
import static com.epsilon.prism.run.Snapshot.iterate;
import static com.epsilon.prism.run.Snapshot.model;
import static com.epsilon.prism.run.Snapshot.net;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.collect.CollectionMetrics;
import com.epsilon.prism.collect.EventLedger;
import com.epsilon.prism.collect.RowSink;
import com.epsilon.prism.collect.Sampling;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.learn.CohortStore;
import com.epsilon.prism.learn.Fit;
import com.epsilon.prism.learn.FitPlan;
import com.epsilon.prism.learn.FitResult;
import com.epsilon.prism.learn.LedgerFit;
import com.epsilon.prism.learn.Selection;
import com.epsilon.prism.learn.Teacher;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.policy.NetworkPolicy;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.RowCounts;
import com.epsilon.runtime.DecisionCollectionDiagnostics;
import com.epsilon.runtime.DecisionExecutionContext;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 網の系譜（train）。版の網で集めた行で次の版の網を育てる。反復 t ごとに
 *
 * <ol>
 *   <li>収集（{@link #collect}）: 版 t の collectWith の網（{@link NetworkPolicy}）の貪欲に床 ε_b を混ぜた b で全席が
 *       打ち、ラベル行（MAIN・ROLLIN）と根の記録（ROOT）を rows-v2/&lt;t&gt; に書いて封印する。封印済みなら使い回す。
 *   <li>運の精算: 生徒の行（コホート t の TRAIN のラベル行と根の記録、{@link Selection#LABELS_AND_ROOTS}）の帳簿で、半荘の偶奇で交差適合
 *       した β_c を当てる（{@link LedgerFit}）。
 *   <li>生徒（{@link #fitStudent}）: 版 t の learnFrom の網 θ_t から温め、生徒の行を 1 行 fitEpochs 周（上限 fitSteps 歩）当てる（{@link Fit}）。
 *       教師は (1 − K′_t)·s_θt + K′_t·Ỹ（Ỹ は帳簿で補正した Y。帳簿のある行がなければ生の Y。錨は凍結した θ_t）、補助の頭の項も K′_t 倍
 *       （{@link Teacher}）、学習率は filterLrMode が K′_t から決める倍率（{@link FilterLrMode}: GAIN なら min(1, K′_t / 0.25)、FULL なら 1）。
 *       K′_t は前の反復で教師に掛けた補正の保留の R² の分だけ K_t を上げたゲイン（{@link PrismConfig#filterGain(int, double)}。最初の反復と補正を
 *       掛けなかった反復の次は K_t）。work/&lt;t&gt;/Single に保存し、再開では使い回す。
 *   <li>方策の記憶（{@link #publish}）: averaging = none なら θ_{t+1} をそのまま版 t+1 にする。それ以外は打つ網を A_{t+1} = (1 − η)·A_t +
 *       η·θ_{t+1}（{@link Averaging}、{@link PrismNetwork#mixIn}）にして、θ_{t+1} を iterate/ に添える。
 *   <li>reports/iterations.jsonl に 1 行、evalEvery 反復ごとに版 t+1 の打つ網（evaluateIterate なら学習の網も）を 3×Heuristic と打たせて
 *       evaluations.jsonl（{@link HeuristicEvaluation}）。どちらも state.json を進める前に書き、再開で同じ反復をやり直したときは書き終えた行を
 *       書き直さない。それから state.json を進め、keepCohorts より古い行と work/&lt;t&gt; を消す。
 * </ol>
 *
 * 版 1 は白紙の網（s ≡ 0 なので argmax p0 で打つ。網の初期値の種は seed + 100、形は {@link PrismConfig#shape}）。再開は state.json の次の
 * コホートから。iterations.jsonl の鍵: iteration・phi・collection（{@link CollectionDriver.Result}）・copies・forks・collectSeconds・totalRows・
 * memory・labels・copyLabels・roots（{@link RowCounts}）・ledger（{@link LedgerFit.Report}）・filter・fitSingle（{@link FitResult}）・accepted・
 * acceptance・average・seconds。filter は使ったゲイン gain = K′_t・錨の重み anchorWeight・学習率の倍率 lrScale に、K_t の scheduledGain と K′ の
 * 元の R² の r2 を足したもの。evaluations.jsonl: iteration・version・heuristic・play・iterate。
 */
public final class Lineage {
  private static final Logger LOG = LoggerFactory.getLogger(Lineage.class);

  private static final Gson GSON =
      new GsonBuilder().setPrettyPrinting().serializeSpecialFloatingPointValues().create();

  /** work/&lt;t&gt; の中の生徒の名前。 */
  private static final String SINGLE = "Single";

  /** 学習の並べ替えの種の塩（derive(seed, LEARN, t, 4)）。 */
  private static final int SINGLE_SALT = 4;

  /** 白紙の網の初期値の種の差（seed + 100）。 */
  private static final int BLANK_SEED = 100;

  /**
   * run の進み（state.json）。
   *
   * @param next 次に収集するコホート（= 次に使う版）
   * @param totalRows これまでに集めた行の総数
   * @param accepted 学習の網を受け入れた反復の数（生徒 1 本なので全ての反復）
   * @param averageWeight 方策の記憶の power の重みの累計（版 next まで。ほかの決め方では 0）
   * @param ledgerR2 前の反復で生徒の教師に掛けた運の精算の補正の保留の R²（掛けなければ 0）
   */
  record State(
      int format, int next, long totalRows, int accepted, double averageWeight, double ledgerR2) {
    static final int FORMAT = 1;
  }

  private final Path root;
  private final PrismConfig config;
  private final RunFiles files;

  private Lineage(Path root, PrismConfig config) {
    this.root = root;
    this.config = config;
    files = new RunFiles(root);
  }

  /**
   * run を始めるか再開する。config.json があればそれを読み（{@link PrismConfig#read}。設定の値は使わないが、知らない鍵は {@link
   * PrismConfig#requireKnownSettings} で拒む）、なければ設定の [epsilon.prism] から作って書く（どちらも {@link PrismConfig#validateTraining}
   * で確かめる）。
   *
   * @param stopAfter この呼び出しで回す反復の数の上限
   * @return 0
   */
  public static int run(Path root, int stopAfter, SettingsLoader settings) throws Exception {
    Path configPath = root.resolve("config.json");
    PrismConfig config;
    if (Files.exists(configPath)) {
      PrismConfig.requireKnownSettings(settings);
      config = PrismConfig.read(Files.readString(configPath));
      config.validateTraining();
      LOG.info("resuming {} {}", root, config);
    } else {
      config = PrismConfig.fromSettings(settings);
      config.validateTraining();
      RunFiles.writeAtomically(configPath, GSON.toJson(config));
    }
    return new Lineage(root, config).loop(stopAfter);
  }

  /** 反復を回す。資源（{@link TrainingResources}）は反復をまたいで 1 つずつ持つ。報告は 1 反復で 1 行。 */
  private int loop(int stopAfter) throws Exception {
    NetworkDevices learnerDevices = config.learnerDevices();
    List<Device> collectDevices = PrismConfig.devices(config.collectDevices());
    var profile = EpsilonUtilityProfile.valueOf(config.utility());
    State state = loadOrInitialize();
    RunFiles.removeFinishedWork(root.resolve("work"), state.next());
    try (var resources = TrainingResources.open(config, profile)) {
      int next = state.next();
      long totalRows = state.totalRows();
      int accepted = state.accepted();
      double averageWeight = state.averageWeight();
      double ledgerR2 = state.ledgerR2();
      Averaging averaging = config.policyMemory();
      for (int done = 0; next <= config.iterations() && done < stopAfter; done++) {
        int t = next;
        long started = System.nanoTime();
        var report = new LinkedHashMap<String, Object>();
        report.put("iteration", t);
        report.put("phi", config.phi());
        // 1. 収集（版 t の b）
        totalRows += collect(t, resources, profile, collectDevices, report);
        report.put("collectSeconds", (System.nanoTime() - started) / 1e9);
        report.put("totalRows", totalRows);
        // 2〜3. 運の精算と生徒
        try (var store = new CohortStore()) {
          report.put("memory", loadMemory(store, t));
          var counts = RowCounts.of(store.select(t, t, index -> true), store::index);
          counts.report(report);
          LOG.info(
              "t={} rows labels={} copyLabels={} roots={} (holdout rows {})",
              t,
              counts.labels(),
              counts.copyLabels(),
              counts.roots(),
              counts.holdout());
          ledgerR2 = learn(t, store, learnerDevices, ledgerR2, report);
        }
        accepted++;
        // 4. 方策の記憶（版 t+1）
        averageWeight = publish(t, averaging, averageWeight, report);
        // 5. 記録と評価を state.json より前に書く（再開で同じ反復をやり直したときは、書き終えた記録と評価を書き直さない）
        int version = t + 1;
        report.put("seconds", (System.nanoTime() - started) / 1e9);
        if (files.lastRecorded(RunFiles.ITERATIONS, "iteration") < t)
          files.append(RunFiles.ITERATIONS, report);
        LOG.info(
            String.format(
                "t=%d published the student as version %d %.1fs",
                t, version, (System.nanoTime() - started) / 1e9));
        if (config.evalEvery() > 0
            && t % config.evalEvery() == 0
            && files.lastRecorded(RunFiles.EVALUATIONS, "version") < version) {
          var evaluation = new LinkedHashMap<String, Object>();
          evaluation.put("iteration", t);
          evaluation.put("version", version);
          Path published = snapshot(version);
          evaluation.putAll(
              evaluate(version, model(published), "", resources, collectDevices, profile));
          if (config.evaluateIterate() && averaging.enabled())
            evaluation.put(
                Averaging.ITERATE,
                evaluate(
                    version, iterate(published), " (iterate)", resources, collectDevices, profile));
          files.append(RunFiles.EVALUATIONS, evaluation);
        }
        next = version;
        RunFiles.writeAtomically(
            root.resolve("state.json"),
            GSON.toJson(
                new State(State.FORMAT, next, totalRows, accepted, averageWeight, ledgerR2)));
        // 古い行と途中の網を消す（state.json の後。間で止まった残りは、次の起動と次の反復の後始末が消す）
        for (int c = 1; c <= t - config.keepCohorts(); c++)
          if (Files.exists(cohort(c))) deleteTree(cohort(c));
        if (Files.exists(work(t))) deleteTree(work(t));
      }
      return 0;
    }
  }

  /**
   * state.json を読む。なければ版 1 を書いて始める: 白紙の網（形は {@link PrismConfig#shape}、初期値の種は seed + 100）。
   */
  private State loadOrInitialize() throws IOException {
    Path path = root.resolve("state.json");
    if (Files.exists(path)) {
      State state = GSON.fromJson(Files.readString(path), State.class);
      if (state == null || state.format() != State.FORMAT)
        throw new IOException(
            "state.json is not a PRISM lineage state of format " + State.FORMAT + ": " + path);
      return state;
    }
    LOG.warn(
        "starting a PRISM lineage from a blank network: s = 0 everywhere, so version 1 plays"
            + " argmax p0");
    try (Model blank = Fit.blank(config.shape(), Device.cpu(), config.seed() + BLANK_SEED)) {
      Snapshot.write(snapshot(1), (PrismNetwork) blank.getBlock(), PolicyParams.of(1));
    }
    // 版 1 は学習の網と打つ網が同じ（iterate/ を持たない）。power の重みの累計は版 1 の分から始める
    var state =
        new State(State.FORMAT, 1, 0, 0, config.policyMemory().step(1, 0).totalWeight(), 0);
    RunFiles.writeAtomically(path, GSON.toJson(state));
    return state;
  }

  /**
   * コホート t を集めて封印する。封印済みなら使い回す（{@link CohortWriter#MANIFEST} がある）。途中のコホートは消して集め直す（その行で当てた
   * work/&lt;t&gt; も消す）。方策は版 t の collectWith の網（{@link Snapshot#open}、p0 の係数は {@link
   * PrismConfig#priorSettings}）で、Cohort(seed, cohortSeed(seed, t), cohortGames(t, gamesPerCohort(t)), 効用, sampling(), behaviour()) を
   * 収集する。受付とデバイスのパイプラインの計測（待ち・合成・空き）は収集の区間だけで測り、方策を閉じる前に記録に書く（{@link
   * DecisionCollectionDiagnostics}）。
   *
   * @return そのコホートの行数（封印した目録の値）
   */
  private long collect(
      int t,
      TrainingResources resources,
      EpsilonUtilityProfile profile,
      List<Device> devices,
      Map<String, Object> report)
      throws IOException {
    Path cohort = cohort(t);
    if (Files.exists(cohort.resolve(CohortWriter.MANIFEST))) {
      report.put("collection", "reused the sealed cohort");
      LOG.info("t={} reusing the sealed cohort", t);
      return CohortWriter.readManifest(cohort).rows();
    }
    if (Files.exists(cohort)) deleteTree(cohort);
    if (Files.exists(work(t))) deleteTree(work(t));
    Sampling sampling = config.sampling();
    CollectionDriver.Result collected;
    try (var policy =
            Snapshot.open(
                snapshot(t),
                net(snapshot(t), config.collectWith()),
                devices,
                resources.execution(),
                config.inferBatch(),
                config.priorSettings(),
                config.inference());
        var writer = new CohortWriter(cohort, t, EventLedger.FEATURES, config.rowWriters())) {
      // 計測の区間をこの収集に合わせる（網を開いた後でまだ推論はない。同じ JVM の前の評価の予約の最大などを持ち越さない）
      DecisionExecutionContext inference = resources.execution().devices();
      inference.resetInferenceMetrics();
      collected =
          resources
              .driver()
              .collect(
                  new CollectionDriver.Cohort(
                      config.seed(),
                      PrismConfig.cohortSeed(config.seed(), t),
                      PrismConfig.cohortGames(t, config.gamesPerCohort(t)),
                      profile,
                      sampling,
                      config.behaviour(),
                      null,
                      null),
                  policy,
                  resources.phi(),
                  new RowSink(writer));
      // 推論を回収してから、方策（デバイスの利用権）を閉じる前に、受付とデバイスごとの待ち・合成・空きを記録に書く
      inference.awaitInferenceIdle();
      DecisionCollectionDiagnostics.log(inference, collected.admission());
      writer.seal();
    }
    // 収集の終わりの数は書き手のキューに残る行を含まないので、封印した目録の値に差し替える
    CohortWriter.Manifest written = CohortWriter.readManifest(cohort);
    CollectionMetrics.Snapshot counts = collected.counts();
    if (written.rows() != counts.rowsWritten())
      throw new IllegalStateException(
          "Cohort " + t + " sealed " + written.rows() + " rows but " + counts.rowsWritten()
              + " were sent");
    collected = collected.withRows(new CollectionDriver.RowStats(written.rows(), written.bytes()));
    report.put("collection", collected);
    report.put("copies", counts.copies());
    report.put("forks", counts.forks());
    LOG.info("t={} {}", t, collected.summary());
    return written.rows();
  }

  /**
   * データの記憶: コホート t だけを store に積む（消えていれば飛ばす）。
   *
   * @return 報告の memory: cohorts（積んだコホート）、rows（積んだ行の数）
   */
  private Map<String, Object> loadMemory(CohortStore store, int t) throws IOException {
    List<Integer> cohorts = new ArrayList<>();
    if (Files.exists(cohort(t).resolve(CohortWriter.MANIFEST))) {
      store.addCohort(cohort(t));
      cohorts.add(t);
    } else {
      LOG.warn("t={} memory skips cohort {} (not on disk)", t, t);
    }
    var memory = new LinkedHashMap<String, Object>();
    memory.put("cohorts", cohorts);
    memory.put("rows", store.size());
    LOG.info("t={} memory cohorts={} rows={}", t, cohorts, store.size());
    return memory;
  }

  /**
   * 反復 t の学習（クラスの説明の 2〜3）: 生徒の行の帳簿で β を当て、生徒 1 本を当てる。帳簿のある行を当てはめたときだけ教師を補正する。
   *
   * @param ledgerR2 前の反復で教師に掛けた補正の保留の R²（この反復の K′ の元）
   * @return この反復で教師に掛けた補正の保留の R²（次の反復の K′ の元。掛けなければ 0）
   */
  private double learn(
      int t, CohortStore store, NetworkDevices devices, double ledgerR2, Map<String, Object> report)
      throws IOException {
    int[] rows = store.select(t, t, Selection.LABELS_AND_ROOTS);
    LedgerFit.Result ledger = fitLedger(t, store, rows, report);
    LedgerFit.Correction correction = ledger.report().rows() > 0 ? ledger.correction() : null;
    if (correction == null)
      LOG.warn("t={} no rows carry a ledger: the student learns the raw Y", t);
    fitStudent(t, store, rows, correction, ledgerR2, devices, report);
    report.put("accepted", true);
    report.put("acceptance", "single");
    double r2 = correction == null ? 0 : ledger.report().all().r2();
    return Double.isNaN(r2) ? 0 : r2;
  }

  /**
   * 運の精算の係数を、生徒の行の帳簿で半荘の偶奇で交差適合して当てる。行は貯蔵から batchSize 行ずつ decodeThreads 本で読む。
   */
  private LedgerFit.Result fitLedger(int t, CohortStore store, int[] rows, Map<String, Object> report) {
    LedgerFit.Result ledger;
    try (ExecutorService readers = Executors.newFixedThreadPool(config.decodeThreads())) {
      ledger = LedgerFit.fit(store.iterate(rows, config.batchSize(), readers, config.decodeThreads()));
    }
    LedgerFit.Report fit = ledger.report();
    report.put("ledger", fit);
    LOG.info(
        String.format(
            "t=%d ledger rows=%d skipped=%d r2=%.4f forks=%.3f groups=%s %.1fs",
            t,
            fit.rows(),
            fit.skippedRows(),
            fit.all().r2(),
            fit.all().equivalentForks(),
            fit.all().r2ByGroup(),
            fit.seconds()));
    return ledger;
  }

  /**
   * 生徒を当てて work/&lt;t&gt;/Single に保存する。保存済み（網と work/&lt;t&gt;/Single.json がそろっている）なら使い回す。行がないか fitSteps が 0
   * なら、θ_t の重みをそのまま保存する。
   *
   * @param correction 運の精算の補正（null なら生の Y）
   * @param ledgerR2 K′ の元の R²
   */
  private void fitStudent(
      int t,
      CohortStore store,
      int[] rows,
      LedgerFit.Correction correction,
      double ledgerR2,
      NetworkDevices devices,
      Map<String, Object> report)
      throws IOException {
    Path directory = work(t).resolve(SINGLE), record = work(t).resolve(SINGLE + ".json");
    String key = "fit" + SINGLE;
    if (Files.isDirectory(directory) && Files.exists(record)) {
      FitResult reused = GSON.fromJson(Files.readString(record), FitResult.class);
      report.put(key, reused);
      LOG.info("t={} reusing student {} ({})", t, SINGLE, directory);
      return;
    }
    // 錨は学習の網 θ_t（始めの網と同じ）で、重み 1 − K′_t。補助の頭の項も K′_t に、学習率は filterLrMode の倍率にする
    double gain = config.filterGain(t, ledgerR2);
    double lrScale = config.filterLrScale(t, ledgerR2);
    var filter = new LinkedHashMap<String, Object>();
    filter.put("gain", gain);
    filter.put("anchorWeight", 1 - gain);
    filter.put("lrScale", lrScale);
    filter.put("scheduledGain", config.filterGain(t));
    filter.put("r2", ledgerR2);
    report.put("filter", filter);
    int steps =
        PrismConfig.steps(config.fitSteps(), config.fitEpochs(), rows.length, config.batchSize());
    Path start = learnStart(t);
    FitPlan plan =
        new FitPlan(
            config.fitSettings(steps, lrScale),
            PrismNetwork.readShape(start),
            config.auxiliaryLoss(),
            FitPlan.TOKEN_BUDGET,
            CrnSeeds.derive(config.seed(), CrnSeeds.Domain.LEARN, t, SINGLE_SALT));
    FitResult result;
    try (Model student = Fit.load(start, devices.primary())) {
      result =
          steps == 0
              ? FitResult.empty(Fit.NAME, rows.length)
              : Fit.run(
                  store,
                  rows,
                  student,
                  plan,
                  new Teacher(correction, start, 1 - gain, (float) gain, config.inference()),
                  devices,
                  config.decodeThreads());
      PrismNetwork network = (PrismNetwork) student.getBlock();
      Snapshot.save(network, directory);
    }
    RunFiles.writeAtomically(record, GSON.toJson(result));
    report.put(key, result);
    LOG.info(result.summary(t));
  }

  /**
   * 版 t+1 を書く。方策の記憶がなければ学習の網 θ_{t+1}（work/&lt;t&gt;/Single）をそのまま版 t+1 にする（ファイルの写し）。あれば打つ網を
   * A_{t+1} = (1 − η)·A_t + η·θ_{t+1}（{@link Averaging#step}、{@link PrismNetwork#mixIn}、CPU・FP32）にして、θ_{t+1} と一緒に書く。
   *
   * @param averageWeight 版 t までの power の重みの累計
   * @return 版 t+1 までの power の重みの累計
   */
  private double publish(int t, Averaging averaging, double averageWeight, Map<String, Object> report)
      throws IOException {
    int version = t + 1;
    Path learned = work(t).resolve(SINGLE);
    PolicyParams params = PolicyParams.of(version);
    if (!averaging.enabled()) {
      Snapshot.writeFiles(learned, snapshot(version), params);
      return 0;
    }
    Averaging.Step step = averaging.step(version, averageWeight);
    var record =
        new Snapshot.AverageRecord(
            averaging.scheme(), step.eta(), step.totalWeight(), averaging.from());
    try (NDManager cpu = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      PrismNetwork average = PrismNetwork.load(model(snapshot(t)), cpu);
      average.mixIn(PrismNetwork.load(learned, cpu), step.eta());
      Snapshot.writeAveraged(learned, average, record, snapshot(version), params);
    }
    report.put("average", record);
    LOG.info(
        "t={} averaged version {}: scheme={} eta={} totalWeight={}",
        t,
        version,
        averaging.scheme(),
        String.format("%.4f", step.eta()),
        step.totalWeight());
    return step.totalWeight();
  }

  /** 版 version の網 net（打つ網か学習の網）を 3×Heuristic と打たせる（貪欲）。結果と対局の内訳。 */
  private Map<String, Object> evaluate(
      int version,
      Path net,
      String label,
      TrainingResources resources,
      List<Device> devices,
      EpsilonUtilityProfile profile)
      throws IOException {
    var evaluation =
        HeuristicEvaluation.run(
            config, snapshot(version), net, resources.driver(), resources.execution(), devices, profile);
    LOG.info(evaluation.summary(version, label));
    LOG.info(evaluation.playSummary(version, label));
    return evaluation.record();
  }

  /** 生徒の始めの網と錨: 版 t の learnFrom の網。 */
  private Path learnStart(int t) {
    return net(snapshot(t), config.learnFrom());
  }

  private Path snapshot(int t) {
    return root.resolve("snapshots").resolve(String.format("%05d", t));
  }

  private Path cohort(int t) {
    return CohortWriter.directory(root, t);
  }

  /** 反復 t の途中の網（Single）と記録のディレクトリ。反復を終えたら消す。 */
  private Path work(int t) {
    return root.resolve("work").resolve(String.format("%05d", t));
  }
}
