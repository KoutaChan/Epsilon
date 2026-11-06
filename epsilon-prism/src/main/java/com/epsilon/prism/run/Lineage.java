package com.epsilon.prism.run;

import static com.epsilon.prism.run.Snapshot.deleteTree;
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
import com.epsilon.prism.eval.Arena;
import com.epsilon.prism.eval.PlayStats;
import com.epsilon.prism.grp.ActivePhi;
import com.epsilon.prism.grp.GameLog;
import com.epsilon.prism.grp.GameLogObserver;
import com.epsilon.prism.grp.GrpBackground;
import com.epsilon.prism.grp.GrpFiles;
import com.epsilon.prism.grp.PhiEntry;
import com.epsilon.prism.learn.AnchorScores;
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
import com.epsilon.prism.run.StopSignal.SafePoint;
import com.epsilon.runtime.DecisionCollectionDiagnostics;
import com.epsilon.runtime.DecisionExecutionContext;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 網の系譜（train）。バージョンの網で集めた行で次のバージョンの網を育てる。反復 t ごとに
 *
 * <ol>
 *   <li>Φ: 反復 t の局末の値（{@link TrainingResources#phi}。SELF_GRP なら phi.jsonl の効く項目で、grpSwitch = AUTO なら合格済みの候補を
 *       反復 t から入れてから決める）。Φ の種類が点数だけの Φ から GRP に変わる反復では、運の精算の R² を持ち越さない（K′_t = K_t）。
 *   <li>収集（{@link #collect}）: バージョン t の collectWith の網（{@link NetworkPolicy}）の貪欲に床 ε_b を混ぜた b で全席が
 *       打ち、ラベル行（MAIN・ROLLIN）と根の記録（ROOT）を rows-v2/&lt;t&gt; に書いて封印する。封印済みなら使い回す。phiMonitor が OFF で
 *       なければ、封印の前に主経路の対局を grp/games に記録する（{@link GameLogObserver}。記録はラベル行を変えず、失敗しても系譜を止めない）。
 *       封印の後（そこで止まらなければ）GRP の裏の作業を頼む（{@link GrpBackground}。採点と、SELF_GRP なら GRP の学習。裏のスレッドで、
 *       待たない。止まった反復のコホートは次の起動の初めに頼む）。
 *   <li>運の精算: 生徒の行（コホート t の TRAIN のラベル行と根の記録、{@link Selection#LABELS_AND_ROOTS}）の帳簿で、半荘の偶奇で交差適合
 *       した β_c を当てる（{@link LedgerFit}）。和は収集の書き手が行を書くときに作り（{@link LedgerFit.Stream}）、使い回したコホートでは
 *       読み直して作る（どちらも同じ行を同じ順に足すので、β はビットで同じ）。
 *   <li>生徒（{@link #fitStudent}）: バージョン t の learnFrom の網 θ_t から温め、生徒の行を 1 行 fitEpochs 周（上限 fitSteps 歩）当てる（{@link Fit}）。
 *       教師は (1 − K′_t)·s_θt + K′_t·Ỹ（Ỹ は帳簿で補正した Y。帳簿のある行がなければ生の Y。錨は凍結した θ_t）、補助の頭の項も K′_t 倍
 *       （{@link Teacher}）、学習率は filterLrMode が K′_t から決める倍率（{@link FilterLrMode}: GAIN なら min(1, K′_t / 0.25)、FULL なら 1）。
 *       K′_t は前の反復で教師に掛けた補正の保留の R² の分だけ K_t を上げたゲイン（{@link PrismConfig#filterGain(int, double)}。最初の反復と補正を
 *       掛けなかった反復の次は K_t）。運の精算の当てはめ（CPU だけ）と並べて、全ての生徒の行の錨の s を GPU で出し（{@link AnchorScores}）、
 *       生徒を読んで学習器を温める（{@link Fit#prepare}）。work/&lt;t&gt;/Single に保存し、再開では使い回す。
 *   <li>方策の記憶（{@link #publish}）: averaging = none なら θ_{t+1} をそのままバージョン t+1 にする。それ以外は打つ網を A_{t+1} = (1 − η)·A_t +
 *       η·θ_{t+1}（{@link Averaging}、{@link PrismNetwork#mixIn}）にして、θ_{t+1} を iterate/ に添える。
 *   <li>reports/iterations.jsonl に 1 行を state.json を進める前に書き（再開で同じ反復をやり直したときは書き終えた行を書き直さない。番号で
 *       引く。{@link RunFiles#recorded}）、state.json を進め、keepCohorts より古い行と work/&lt;t&gt; を消す。
 *   <li>評価（{@link Evaluation}）: 測っていない評価（{@link EvaluationOutputs#pending}。バージョン t+1 を含む）を古い順に測る（{@link
 *       #measure}）。
 * </ol>
 *
 * 反復は iterations まで（stopAfter 回まで）回し、止まる合図（{@link StopSignal}）が立てば次の安全点で止まる: コホートの封印の後（{@link
 * SafePoint#SEALED}）、反復の後（{@link SafePoint#ITERATED}）、評価 1 つの後（{@link SafePoint#MEASURED}）。どこで止まっても（落ちても）
 * 再開は同じ結果に戻る: 封印したコホートと保存した生徒は使い回し、バージョンの公開は差し替え、記録は番号で引いて書き直さず、評価は .done のないものを
 * 初めから測る。バージョンが公開されたのは state.json の next が進んだときだけ。
 *
 * バージョン 1 は白紙の網（s ≡ 0 なので argmax p0 で打つ。網の初期値の種は seed + 100、形は {@link PrismConfig#shape}）。再開は state.json の次の
 * コホートから。iterations.jsonl の鍵: iteration・phi（使った Φ の項目 {@link PhiEntry}）・ledgerR2Reset（R² を持ち越さなかった反復だけ、
 * 捨てた R²）・collection（{@link CollectionDriver.Result}）・copies・forks・collectSeconds・totalRows・
 * memory・labels・copyLabels・roots（{@link RowCounts}）・ledger（{@link LedgerFit.Report}）・filter・fitSingle（{@link FitResult}）・accepted・
 * acceptance・average・seconds。filter は使ったゲイン gain = K′_t・錨の重み anchorWeight・学習率の倍率 lrScale に、K_t の scheduledGain と K′ の
 * 元の R² の r2 を足したもの。
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
   * @param next 次に収集するコホート（= 次に使うバージョン）
   * @param totalRows これまでに集めた行の総数
   * @param accepted 学習の網を受け入れた反復の数（生徒 1 本なので全ての反復）
   * @param averageWeight 方策の記憶の power の重みの累計（バージョン next まで。ほかの決め方では 0）
   * @param ledgerR2 前の反復で生徒の教師に掛けた運の精算の補正の保留の R²（掛けなければ 0）
   */
  record State(
      int format, int next, long totalRows, int accepted, double averageWeight, double ledgerR2) {
    static final int FORMAT = 1;
  }

  private final Path root;
  private final PrismConfig config;
  private final StopSignal stop;
  private final RunFiles files;
  private final GrpFiles grpFiles;
  private final EpsilonUtilityProfile profile;
  private final NetworkDevices learnerDevices;
  private final List<Device> collectDevices;

  private Lineage(Path root, PrismConfig config, StopSignal stop) {
    this.root = root;
    this.config = config;
    this.stop = stop;
    files = new RunFiles(root);
    grpFiles = new GrpFiles(root);
    profile = EpsilonUtilityProfile.valueOf(config.utility());
    learnerDevices = config.learnerDevices();
    collectDevices = PrismConfig.devices(config.collectDevices());
  }

  /**
   * run を始めるか再開する。config.json があればそれを読み（{@link PrismConfig#read}。設定の値は使わないが、知らない鍵は {@link
   * PrismConfig#requireKnownSettings} で拒む）、なければ設定の [epsilon.prism] から作って書く（どちらも {@link PrismConfig#validateTraining}
   * で確かめる）。
   *
   * @param stopAfter この呼び出しで回す反復の数の上限
   * @param stop 止まる合図（立てば次の安全点で止まる）
   * @return 0（反復を終えたか、上限に達したか、合図で止まった）
   */
  public static int run(Path root, int stopAfter, SettingsLoader settings, StopSignal stop)
      throws Exception {
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
    return new Lineage(root, config, stop).loop(stopAfter);
  }

  /**
   * 反復を回す。資源（{@link TrainingResources}）と GRP の裏の作業（{@link GrpBackground}）は反復をまたいで 1 つずつ持つ。起動の初めに未採点の
   * コホートの採点を頼み、測っていない評価を測る。報告は 1 反復で 1 行。
   */
  private int loop(int stopAfter) throws Exception {
    State state = loadOrInitialize();
    RunFiles.removeFinishedWork(root.resolve("work"), state.next());
    try (var resources = TrainingResources.open(config, profile, root);
        var background = GrpBackground.start(grpFiles, config)) {
      background.request();
      if (!measurePending(state.next(), resources)) return 0;
      for (int done = 0;
          state.next() <= config.iterations() && done < stopAfter && !stop.requested();
          done++) {
        int t = state.next();
        state = iterate(state, resources, background);
        if (state == null || stop.stopsAt(SafePoint.ITERATED, t)) break;
        if (!measurePending(state.next(), resources)) break;
      }
    }
    return 0;
  }

  /**
   * 反復 t（クラスの説明の Φ〜評価）。コホートの封印の後に止まる合図が立っていれば、裏の作業を頼まず運の精算の前に止まる（{@link
   * SafePoint#SEALED}。再開では封印したコホートを使い回す）。
   *
   * @param state 反復の前の state.json（next = t）
   * @return 反復の後の state.json（封印の後で止まったときは null）
   */
  private State iterate(State state, TrainingResources resources, GrpBackground background)
      throws Exception {
    int t = state.next();
    long started = System.nanoTime();
    var report = new LinkedHashMap<String, Object>();
    report.put("iteration", t);
    double ledgerR2 = state.ledgerR2();
    // 0. 反復 t の Φ（Φ の種類が点数だけの Φ から GRP に変わる反復は、運の精算の R² を持ち越さない）
    ActivePhi phi = resources.phi(t);
    report.put("phi", phi.entry());
    if (phi.resetsLedger()) {
      report.put("ledgerR2Reset", ledgerR2);
      ledgerR2 = 0;
    }
    // 1. 収集（バージョン t の b）と封印
    Collected collected = collect(t, resources, phi, report);
    if (stop.stopsAt(SafePoint.SEALED, t)) return null;
    background.request();
    long totalRows = state.totalRows() + collected.rows();
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
      ledgerR2 = learn(t, store, collected.ledger(), ledgerR2, resources.learning(), report);
    }
    // 4. 方策の記憶（バージョン t+1）
    Averaging averaging = config.policyMemory();
    double averageWeight = publish(t, averaging, state.averageWeight(), report);
    // 5. 記録を state.json より前に書く（再開で同じ反復をやり直したときは、書き終えた記録を書き直さない）
    int version = t + 1;
    report.put("seconds", (System.nanoTime() - started) / 1e9);
    if (!files.recorded(RunFiles.ITERATIONS, "iteration", t)) files.append(RunFiles.ITERATIONS, report);
    LOG.info(
        String.format(
            "t=%d published the student as version %d %.1fs",
            t, version, (System.nanoTime() - started) / 1e9));
    var next =
        new State(
            State.FORMAT, version, totalRows, state.accepted() + 1, averageWeight, ledgerR2);
    RunFiles.writeAtomically(root.resolve("state.json"), GSON.toJson(next));
    // 古い行と途中の網を消す（state.json の後。間で止まった残りは、次の起動と次の反復の後始末が消す）
    for (int c = 1; c <= t - config.keepCohorts(); c++)
      if (Files.exists(cohort(c))) deleteTree(cohort(c));
    if (Files.exists(work(t))) deleteTree(work(t));
    return next;
  }

  /**
   * 測っていない評価を古い順に測る（{@link EvaluationOutputs#pending}）。評価の前と後に止まる合図を見る（{@link SafePoint#MEASURED}）。
   *
   * @param latest 公開した最新のバージョン
   * @return 合図で止まるなら false
   */
  private boolean measurePending(int latest, TrainingResources resources) throws IOException {
    for (EvaluationOutputs outputs : EvaluationOutputs.pending(root, config.evaluations(), latest)) {
      if (stop.requested()) return false;
      measure(evaluation(outputs.name()), outputs, resources);
      if (stop.stopsAt(SafePoint.MEASURED, outputs.version())) return false;
    }
    return true;
  }

  /** 名前の評価。 */
  private Evaluation evaluation(String name) {
    return config.evaluations().stream().filter(evaluation -> evaluation.name().equals(name)).findFirst().orElseThrow();
  }

  /**
   * バージョンの打つ網 1 席対 opponent 3 席を games 局打ち（{@link NanoTable}。学習の資源で打ち、Nano は {@value NanoTable#NANO_DEVICE}）、
   * 出力に書く（{@link EvaluationOutputs}）。log は evaluate の出力と同じ文言の 3 行（started、evaluate play、complete）。Java の例外で
   * 測れなければ、log に失敗を書いて .failed にし、系譜は続ける。
   *
   * @throws IOException 印か失敗の log が書けないとき
   */
  private void measure(Evaluation evaluation, EvaluationOutputs outputs, TrainingResources resources) throws IOException {
    int version = outputs.version();
    outputs.begin();
    var log = new StringBuilder();
    logLine(
        log,
        String.format(
            "evaluation workflow started: command=evaluate version=%d opponent=%s games=%d",
            version, evaluation.opponent(), evaluation.games()));
    LOG.info("{} v{} started", evaluation.name(), version);
    long started = System.nanoTime();
    try {
      var stats = new PlayStats();
      Arena.Detailed detailed =
          NanoTable.play(
              config,
              snapshot(version),
              evaluation.opponent(),
              evaluation.games(),
              resources.driver(),
              resources.execution(),
              NanoTable.NANO_DEVICE,
              stats);
      String complete =
          String.format(
              "evaluation workflow complete: command=evaluate successful=true elapsedMs=%d %s",
              (System.nanoTime() - started) / 1_000_000L,
              NanoTable.summary(detailed.result()));
      logLine(log, NanoTable.playLine(stats));
      logLine(log, complete);
      outputs.commit(NanoTable.perGame(detailed.games()), log.toString());
      LOG.info("{} v{} {}", evaluation.name(), version, complete);
    } catch (Exception failure) {
      LOG.warn("{} v{} failed", evaluation.name(), version, failure);
      logLine(
          log,
          String.format(
              "evaluation workflow failed: command=evaluate elapsedMs=%d %s",
              (System.nanoTime() - started) / 1_000_000L,
              failure));
      outputs.fail(log.toString());
    }
  }

  /** 評価の log に時刻つきの 1 行を足す。 */
  private static void logLine(StringBuilder log, String message) {
    log.append(Instant.now()).append(' ').append(message).append('\n');
  }

  /**
   * 集めたコホート。
   *
   * @param rows 行数（封印した目録の値）
   * @param ledger 書き手が作った運の精算の和（封印済みのコホートを使い回したときは null）
   */
  private record Collected(long rows, LedgerFit.Stream ledger) {}

  /**
   * state.json を読む。なければバージョン 1 を書いて始める: 白紙の網（形は {@link PrismConfig#shape}、初期値の種は seed + 100）。
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
    // バージョン 1 は学習の網と打つ網が同じ（iterate/ を持たない）。power の重みの累計はバージョン 1 の分から始める
    var state =
        new State(State.FORMAT, 1, 0, 0, config.policyMemory().step(1, 0).totalWeight(), 0);
    RunFiles.writeAtomically(path, GSON.toJson(state));
    return state;
  }

  /**
   * コホート t を集めて封印する。封印済みなら使い回す（{@link CohortWriter#MANIFEST} がある）。途中のコホートは消して集め直す（その行で当てた
   * work/&lt;t&gt; も消す）。方策はバージョン t の collectWith の網（{@link Snapshot#open}、p0 の係数は {@link
   * PrismConfig#priorSettings}）で、Cohort(seed, cohortSeed(seed, t), cohortGames(t, gamesPerCohort(t)), 効用, sampling(), behaviour()) を
   * 収集する。書き手は書いた行で運の精算の和を作る（{@link LedgerFit.Stream}）。受付とデバイスのパイプラインの計測（待ち・合成・空き）は
   * 収集の区間だけで測り、方策を閉じる前に記録に書く（{@link DecisionCollectionDiagnostics}）。
   *
   * @return そのコホートの行数と運の精算の和
   */
  private Collected collect(
      int t, TrainingResources resources, ActivePhi phi, Map<String, Object> report)
      throws IOException {
    Path cohort = cohort(t);
    if (Files.exists(cohort.resolve(CohortWriter.MANIFEST))) {
      report.put("collection", "reused the sealed cohort");
      LOG.info("t={} reusing the sealed cohort", t);
      return new Collected(CohortWriter.readManifest(cohort).rows(), null);
    }
    if (Files.exists(cohort)) deleteTree(cohort);
    if (Files.exists(work(t))) deleteTree(work(t));
    Sampling sampling = config.sampling();
    CollectionDriver.Result collected;
    var ledger = new LedgerFit.Stream(config.rowWriters());
    GameLogObserver games = gameLog(t, phi);
    try (var policy =
            Snapshot.open(
                snapshot(t),
                net(snapshot(t), config.collectWith()),
                collectDevices,
                resources.execution(),
                config.inferBatch(),
                config.priorSettings(),
                config.inference());
        var writer =
            new CohortWriter(cohort, t, EventLedger.FEATURES, config.rowWriters(), ledger)) {
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
                      games),
                  policy,
                  phi.value(),
                  new RowSink(writer));
      // 推論を回収してから、方策（デバイスの利用権）を閉じる前に、受付とデバイスごとの待ち・合成・空きを記録に書く
      inference.awaitInferenceIdle();
      DecisionCollectionDiagnostics.log(inference, collected.admission());
      // 対局の記録は封印の前に書く（封印済みのコホートを使い回すときは記録し直さない）
      writeGameLog(t, games, collected.counts());
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
    return new Collected(written.rows(), ledger);
  }

  /**
   * コホート t の主経路の対局の記録の口（phiMonitor が OFF なら null）。開けなければ 1 回書き残して記録しない（記録の失敗で系譜を止めない）。
   */
  private GameLogObserver gameLog(int t, ActivePhi phi) {
    if (!config.phiMonitor().records()) return null;
    try {
      return GameLogObserver.open(t, phi.entry(), config.behaviour());
    } catch (RuntimeException failure) {
      LOG.warn("t={} the game log is off for this cohort: {}", t, failure.toString());
      return null;
    }
  }

  /**
   * コホート t の対局の記録を grp/games に書く（記録しなければ何もしない）。失敗は 1 回書き残して続ける（記録の失敗で系譜を止めない）。
   *
   * @param counts 収集の数（局末の数を記録と突き合わせる）
   */
  private void writeGameLog(int t, GameLogObserver games, CollectionMetrics.Snapshot counts) {
    if (games == null) return;
    try {
      GameLog.write(grpFiles.games(t), games.seal(counts));
    } catch (IOException | RuntimeException failure) {
      LOG.warn("t={} the game log was not written: {}", t, failure.toString());
    }
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
   * 生徒を当てるなら、帳簿の当てはめの間に錨の s と学習器の温めを別のスレッドで進める（帳簿は CPU だけを使い、学習のデバイスは空いている）。
   *
   * @param stream 収集の書き手が作った運の精算の和（null ならコホートを読み直す）
   * @param ledgerR2 前の反復で教師に掛けた補正の保留の R²（この反復の K′ の元）
   * @param learning 学習器のストリームを借りる実行の文脈（{@link TrainingResources#learning}）
   * @return この反復で教師に掛けた補正の保留の R²（次の反復の K′ の元。掛けなければ 0）
   */
  private double learn(
      int t,
      CohortStore store,
      LedgerFit.Stream stream,
      double ledgerR2,
      DecisionExecutionContext learning,
      Map<String, Object> report)
      throws IOException {
    int[] rows = store.select(t, t, Selection.LABELS_AND_ROOTS);
    boolean fits = !studentSaved(t) && fitSteps(rows.length) > 0;
    Path start = learnStart(t);
    LedgerFit.Result ledger;
    AnchorScores anchor = null;
    Fit.Prepared prepared;
    try (ExecutorService ahead = Executors.newFixedThreadPool(2)) {
      Future<AnchorScores> scores =
          fits
              ? ahead.submit(
                  () ->
                      AnchorScores.compute(
                          store, rows, start, config.inference(), learnerDevices, config.decodeThreads()))
              : null;
      Future<Fit.Prepared> warmed =
          fits
              ? ahead.submit(
                  () ->
                      Fit.prepare(
                          store,
                          rows,
                          start,
                          studentPlan(t, rows.length, ledgerR2),
                          (float) config.filterGain(t, ledgerR2),
                          learnerDevices,
                          config.decodeThreads(),
                          learning))
              : null;
      try {
        ledger = fitLedger(t, store, rows, stream, report);
        if (scores != null) anchor = await(scores);
      } catch (IOException | RuntimeException | Error failure) {
        if (warmed != null) closeAfter(warmed, failure);
        throw failure;
      }
      prepared = warmed == null ? null : await(warmed);
    }
    LedgerFit.Correction correction = ledger.report().rows() > 0 ? ledger.correction() : null;
    try {
      if (correction == null)
        LOG.warn("t={} no rows carry a ledger: the student learns the raw Y", t);
      fitStudent(t, rows, correction, anchor, prepared, ledgerR2, learnerDevices, report);
    } finally {
      if (prepared != null) prepared.close();
    }
    report.put("accepted", true);
    report.put("acceptance", "single");
    double r2 = correction == null ? 0 : ledger.report().all().r2();
    return Double.isNaN(r2) ? 0 : r2;
  }

  /**
   * 運の精算の係数を、生徒の行の帳簿で半荘の偶奇で交差適合して当てる。和は書き手が作ったもの（stream）を使い、なければ行を貯蔵から batchSize
   * 行ずつ decodeThreads 本で読み直す。
   *
   * @throws IllegalStateException stream の行の数が選んだ行の数と違うとき
   */
  private LedgerFit.Result fitLedger(
      int t, CohortStore store, int[] rows, LedgerFit.Stream stream, Map<String, Object> report) {
    LedgerFit.Result ledger;
    if (stream != null) {
      ledger = stream.fit();
      long seen = ledger.report().rows() + ledger.report().skippedRows();
      if (seen != rows.length)
        throw new IllegalStateException(
            "Ledger of cohort " + t + " summed " + seen + " rows but " + rows.length + " are selected");
    } else {
      try (ExecutorService readers = Executors.newFixedThreadPool(config.decodeThreads())) {
        ledger =
            LedgerFit.fit(store.iterate(rows, config.batchSize(), readers, config.decodeThreads()));
      }
    }
    LedgerFit.Report fit = ledger.report();
    report.put("ledger", fit);
    LOG.info(
        String.format(
            "t=%d ledger %s rows=%d skipped=%d r2=%.4f forks=%.3f groups=%s %.1fs",
            t,
            stream != null ? "streamed" : "re-read",
            fit.rows(),
            fit.skippedRows(),
            fit.all().r2(),
            fit.all().equivalentForks(),
            fit.all().r2ByGroup(),
            fit.seconds()));
    return ledger;
  }

  /**
   * 生徒を当てて work/&lt;t&gt;/Single に保存する。保存済み（{@link #studentSaved}）なら使い回す。行がないか fitSteps が 0 なら、θ_t の重みを
   * そのまま保存する。
   *
   * @param correction 運の精算の補正（null なら生の Y）
   * @param anchor 錨 θ_t の全ての生徒の行の s（使い回すときと歩数が 0 のときは null）
   * @param prepared θ_t から読んで温めた生徒（使い回すときと歩数が 0 のときは null。閉じるのは呼び手）
   * @param ledgerR2 K′ の元の R²
   * @throws IllegalStateException 錨の s が θ_t と違う重みの網から出たとき
   */
  private void fitStudent(
      int t,
      int[] rows,
      LedgerFit.Correction correction,
      AnchorScores anchor,
      Fit.Prepared prepared,
      double ledgerR2,
      NetworkDevices devices,
      Map<String, Object> report)
      throws IOException {
    Path directory = work(t).resolve(SINGLE), record = work(t).resolve(SINGLE + ".json");
    String key = "fit" + SINGLE;
    if (studentSaved(t)) {
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
    Path start = learnStart(t);
    FitResult result;
    if (prepared == null) {
      try (Model student = Fit.load(start, devices.primary())) {
        result = FitResult.empty(rows.length);
        Snapshot.save((PrismNetwork) student.getBlock(), directory);
      }
    } else {
      anchor.requireNetwork(start);
      result = prepared.fit(new Teacher(correction, anchor, 1 - gain, (float) gain));
      Snapshot.save(prepared.network(), directory);
    }
    RunFiles.writeAtomically(record, GSON.toJson(result));
    report.put(key, result);
    LOG.info(result.summary(t));
  }

  /**
   * バージョン t+1 を書く。方策の記憶がなければ学習の網 θ_{t+1}（work/&lt;t&gt;/Single）をそのままバージョン t+1 にする（ファイルの写し）。あれば打つ網を
   * A_{t+1} = (1 − η)·A_t + η·θ_{t+1}（{@link Averaging#step}、{@link PrismNetwork#mixIn}、CPU・FP32）にして、θ_{t+1} と一緒に書く。
   *
   * @param averageWeight バージョン t までの power の重みの累計
   * @return バージョン t+1 までの power の重みの累計
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

  /** 生徒を保存済み（網と work/&lt;t&gt;/Single.json がそろっている）なら true。 */
  private boolean studentSaved(int t) {
    return Files.isDirectory(work(t).resolve(SINGLE))
        && Files.exists(work(t).resolve(SINGLE + ".json"));
  }

  /**
   * 反復 t の生徒の当てはめの計画: 歩数は {@link #fitSteps}、学習率は filterLrMode の倍率、形は θ_t の形、並べ替えの種は
   * derive(seed, LEARN, t, {@value #SINGLE_SALT})。
   */
  private FitPlan studentPlan(int t, int rows, double ledgerR2) throws IOException {
    return new FitPlan(
        config.fitSettings(fitSteps(rows), config.filterLrScale(t, ledgerR2)),
        PrismNetwork.readShape(learnStart(t)),
        config.auxiliaryLoss(),
        FitPlan.TOKEN_BUDGET,
        CrnSeeds.derive(config.seed(), CrnSeeds.Domain.LEARN, t, SINGLE_SALT));
  }

  /** 失敗の後に、先に始めた温めの完了を待って閉じる（温めと閉じるときの例外は失敗に添える）。 */
  private static void closeAfter(Future<Fit.Prepared> warmed, Throwable failure) {
    try {
      await(warmed).close();
    } catch (IOException | RuntimeException | Error problem) {
      failure.addSuppressed(problem);
    }
  }

  /** 生徒の行の数から当てはめの歩数（1 行 fitEpochs 周、上限 fitSteps 歩）。 */
  private int fitSteps(int rows) {
    return PrismConfig.steps(config.fitSteps(), config.fitEpochs(), rows, config.batchSize());
  }

  /** 先に始めた仕事の完了を待つ。仕事の例外は包みを外して投げる。 */
  private static <T> T await(Future<T> future) throws IOException {
    try {
      return future.get();
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IOException("interrupted while waiting for the learner preparation", interrupted);
    } catch (ExecutionException failure) {
      Throwable cause = failure.getCause();
      if (cause instanceof IOException unreadable) throw unreadable;
      if (cause instanceof RuntimeException runtime) throw runtime;
      if (cause instanceof Error error) throw error;
      throw new IOException("the learner preparation failed", cause);
    }
  }

  /** 生徒の始めの網と錨: バージョン t の learnFrom の網。 */
  private Path learnStart(int t) {
    return net(snapshot(t), config.learnFrom());
  }

  private Path snapshot(int t) {
    return RunFiles.snapshot(root, t);
  }

  private Path cohort(int t) {
    return CohortWriter.directory(root, t);
  }

  /** 反復 t の途中の網（Single）と記録のディレクトリ。反復を終えたら消す。 */
  private Path work(int t) {
    return root.resolve("work").resolve(String.format("%05d", t));
  }
}
