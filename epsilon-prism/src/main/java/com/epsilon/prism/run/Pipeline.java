package com.epsilon.prism.run;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.collect.CollectionMetrics;
import com.epsilon.prism.collect.FixedGrpPhi;
import com.epsilon.prism.collect.HandValue;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.eval.Arena;
import com.epsilon.prism.eval.PlayStats;
import com.epsilon.prism.learn.BatchAssembler;
import com.epsilon.prism.learn.Certification;
import com.epsilon.prism.learn.DistillFit;
import com.epsilon.prism.learn.FitResult;
import com.epsilon.prism.learn.FitSettings;
import com.epsilon.prism.learn.HoldoutSkill;
import com.epsilon.prism.learn.StudentFit;
import com.epsilon.prism.model.PrismModel;
import com.epsilon.prism.policy.PlayerPolicy;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.rows.PrismRow;
import com.epsilon.prism.rows.RowFormat;
import com.epsilon.prism.rows.RowSink;
import com.epsilon.prism.rows.RowStore;
import com.epsilon.prism.rows.RowWriter;
import com.epsilon.spi.PolicyExecutionContext;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * PRISM-min の輪（train。論文 3.6 を K = 1・恒等の核・MH なしにしたもの）。反復 t ごとに
 *
 * <ol>
 *   <li>収集（{@link #collect}）: 版 t の collectWith の網（打つ網か学習の網）の貪欲に床 ε_b を混ぜた b で全席が打ち、ラベル行（MAIN・ROLLIN）と
 *       根の記録（ROOT）を rows/&lt;t&gt; に書いて封印する。封印済みなら使い回す。
 *   <li>データの記憶（{@link #loadMemory}）: 直近 memoryCohorts コホートを積む。年齢 a のコホートはラベル行の割合 memoryDecay^a だけを使う。
 *   <li>生徒（{@link #fitStudent}）: 生徒 A と B を版 t の learnFrom の網から温め込み、データの記憶の組 A（B）の TRAIN のラベル行で測った手が 2 つ
 *       以上の行に差分整合を当てる。行の並べ替えの種は A が derive(seed, LEARN, t, 1)、B が derive(seed, LEARN, t, 2)。work/&lt;t&gt;/A・B に保存
 *       し、再開では使い回す。
 *   <li>蒸留（{@link #distill}）: 版 t の learnFrom の網から始め、データの記憶の両方の組の TRAIN のラベル行に、A と B の平均の分布への CE を当てる
 *       （並べ替えの種 derive(seed, LEARN, t, 3)）。work/&lt;t&gt;/distilled に保存し、再開では使い回す。
 *   <li>learner = single なら、生徒・蒸留・認証の代わりに生徒 1 本（work/&lt;t&gt;/Single）を版 t の learnFrom の網から温め込み、両方の組の
 *       TRAIN のラベル行で測った手が 2 つ以上の行に差分整合を当てて、そのまま学習の網 θ_{t+1} にする（並べ替えの種 derive(seed, LEARN, t, 4)）。
 *   <li>認証（{@link com.epsilon.prism.learn.Certification}）: コホート t の根の記録（両方の組）で Î_A・Î_B（判断あたりの改善）を求め、certify なら
 *       確約の類の下限 &gt; −certifyTolerance かつ C の外の乗り換えの割合 ≤ certifyMaxOutsideShare のとき蒸留した網を受け入れる（certify = false
 *       なら常に受け入れる）。学習の網 θ_{t+1} は、受け入れたら蒸留した網、受け入れなければ θ_t。
 *   <li>方策の記憶（{@link #publish}）: averaging = none なら θ_{t+1} をそのまま版 t+1 にする（ファイルの写し）。それ以外は打つ網を
 *       A_{t+1} = (1 − η)·A_t + η·θ_{t+1}（{@link Averaging}）にして、θ_{t+1} を iterate/ に添えて版 t+1 を書く。
 *   <li>保留の指標（{@link com.epsilon.prism.learn.HoldoutSkill}）: コホート t の HOLDOUT のラベル行で測った手が 2 つ以上の行で、蒸留した網と版 t を測る。
 *   <li>reports/iterations.jsonl に 1 行、evalEvery 反復ごとに版 t+1 の打つ網（evaluateIterate なら学習の網も）を 3×Heuristic と打たせて
 *       reports/evaluations.jsonl（{@link #evaluate}）。どちらも
 *       state.json を進める前に書き、再開で同じ反復をやり直したときは書き終えた行を書き直さない（{@link #lastRecorded}）。それから state.json を
 *       next = t+1 にし、keepCohorts より古い行と work/&lt;t&gt; を消す。
 * </ol>
 *
 * 版 1 は startSnapshot（epsilon-prism の版）の重みか白紙の網を、この run の構成で書いたもの。再開は state.json の次のコホートから（JVM を周期的に作り
 * 直しても続きから進む）。封印済みのコホートと work/&lt;t&gt; に保存済みの生徒・蒸留した網は使い回し、認証・保留の指標・版 t+1 は作り直す。Φ は
 * phi = grp なら固定の GRP（grpCheckpoint 必須）、score-rank は試験用の代用（{@link PrismConfig#validateTraining}）。
 *
 * <p>iterations.jsonl の 1 行の鍵: iteration、phi、collection（{@link com.epsilon.prism.collect.CollectionDriver.Result}。rows は封印した manifest の
 * 値。封印済みを使い回したら文字列）、
 * copies・forks（新しく集めたときだけ。{@link CollectionMetrics.Snapshot}）、collectSeconds、totalRows、labels・copyLabels・roots（コホート t の
 * 索引から数えた、類の名前 → 行の数）、fitA・fitB・distill（{@link com.epsilon.prism.learn.FitResult}）、certification（{@link
 * com.epsilon.prism.learn.Certification.Report}）、accepted（真偽）、acceptance（"certified"・"rejected"・"uncertified"）、holdout（candidate
 * と current の {@link com.epsilon.prism.learn.HoldoutSkill.Report}）、memory（積んだコホート・割合・行の数）、average（方策の記憶があるとき
 * {@link Snapshot.AverageRecord}）、decode（この反復の展開の {読み出し秒, 復号秒, 行数}）、seconds。
 * evaluations.jsonl: iteration、version（t+1）、heuristic（{@link com.epsilon.prism.eval.Arena.Result}）、play（{@link
 * com.epsilon.prism.eval.PlayStats#report}）は打つ網のもの。evaluateIterate なら iterate（学習の網の heuristic と play）。
 */
public final class Pipeline {
  private static final Logger LOG = LoggerFactory.getLogger(Pipeline.class);
  private static final Gson GSON =
      new GsonBuilder().setPrettyPrinting().serializeSpecialFloatingPointValues().create();

  /** jsonl は 1 行 1 記録。 */
  private static final Gson LINES =
      new GsonBuilder().serializeSpecialFloatingPointValues().create();

  private static final String ITERATIONS = "iterations.jsonl", EVALUATIONS = "evaluations.jsonl";

  /** work/&lt;t&gt; の中の蒸留した網と、その記録の名前。 */
  private static final String DISTILLED = "distilled", DISTILL_RECORD = "distill.json";

  /** work/&lt;t&gt; の中の全データの生徒（learner = single）の名前。 */
  private static final String SINGLE = "Single";

  /** Φ のメモの上限。 */
  private static final int PHI_MEMO = 1 << 16;

  private final Path root;
  private final PrismConfig config;

  /** 反復の途中の網（生徒・蒸留した網）の置き場。train は root/work、refit は出力の下。 */
  private final Path workRoot;

  /** 学習に使う行の範囲と間引き（train は {@link RowWindow#ALL}）。 */
  private final RowWindow window;

  /** 学習の行の間引きの硬貨の座標（"thin"）。 */
  private static final long THIN = 0x7468696eL;

  /**
   * 学習（生徒と蒸留）に使う行の範囲と間引き。refit で、新しいコホートだけ・古いコホートだけ・行の一部だけで学ばせるのに使う。
   *
   * @param from 学習の行の最初のコホート（0 なら記憶の窓の最初）
   * @param to 学習の行の最後のコホート（0 なら反復 t）
   * @param fraction 行を残す割合（(0, 1]）
   * @param limit 行の数の上限の目安（0 なら上限なし。残す割合を limit / 行の数 以下にする）
   */
  public record RowWindow(int from, int to, double fraction, long limit) {
    /** 記憶の窓の全ての行（train）。 */
    public static final RowWindow ALL = new RowWindow(0, 0, 1, 0);

    public RowWindow {
      if (from < 0 || to < 0 || (from > 0 && to > 0 && from > to))
        throw new IllegalArgumentException("Invalid cohort range " + from + ".." + to);
      if (!(fraction > 0 && fraction <= 1))
        throw new IllegalArgumentException("fraction must be in (0, 1]: " + fraction);
      if (limit < 0) throw new IllegalArgumentException("limit must not be negative: " + limit);
    }

    /** rows 行のうち残す割合。 */
    double keep(int rows) {
      return limit > 0 && rows > 0 ? Math.min(fraction, (double) limit / rows) : fraction;
    }
  }

  private Pipeline(Path root, PrismConfig config, Path workRoot, RowWindow window) {
    this.root = root;
    this.config = config;
    this.workRoot = workRoot;
    this.window = window;
  }

  /**
   * run を始めるか再開する。config.json がなければ設定から書く。
   *
   * @param stopAfter この呼び出しで回す反復の数の上限
   * @return 0
   */
  public static int run(Path root, int stopAfter, SettingsLoader settings) throws Exception {
    Path configPath = root.resolve("config.json");
    PrismConfig config;
    if (Files.exists(configPath)) {
      config = GSON.fromJson(Files.readString(configPath), PrismConfig.class);
      config.validateTraining();
      LOG.info("resuming {} {}", root, config);
    } else {
      config = PrismConfig.fromSettings(settings);
      config.validateTraining();
      writeAtomically(configPath, GSON.toJson(config));
    }
    return new Pipeline(root, config, root.resolve("work"), RowWindow.ALL).loop(stopAfter);
  }

  /**
   * run の反復 t の学習（データの記憶・生徒・蒸留・認証・保留の指標）を、収集せずにやり直す（refit）。行と版 t は run のものを読むだけで、途中の網は
   * out/work、学習の網は版 t+1 の形で out/snapshot（打つ網 m0 = 学習の網。方策の記憶は掛けない）、記録は out/report.json に書く。replicate が
   * 0 なら並べ替えの種は train と同じ、それ以外は replicate ごとに別の種。config は run の config.json に学習の設定（learner など）を重ねたもの。
   * window で学習の行の範囲と間引きを変えられる（保留の指標はいつもコホート t の HOLDOUT の行）。
   *
   * @return 学習の網の版のディレクトリ（out/snapshot）
   */
  public static Path refit(
      Path root, PrismConfig config, int t, int replicate, RowWindow window, Path out)
      throws IOException {
    config.validateTraining();
    var pipeline = new Pipeline(root, config, out.resolve("work"), window);
    if (!Files.isDirectory(pipeline.snapshot(t)))
      throw new IllegalArgumentException("Missing version " + t + ": " + pipeline.snapshot(t));
    if (Files.exists(pipeline.work(t))) Snapshot.deleteTree(pipeline.work(t));
    var report = new LinkedHashMap<String, Object>();
    report.put("iteration", t);
    report.put("learner", config.learner());
    report.put("replicate", replicate);
    report.put("window", window);
    long started = System.nanoTime();
    Path learned;
    int first = Math.max(1, t - config.memoryCohorts() + 1);
    try (var store = new RowStore();
        var assembler = new BatchAssembler(config.decodeThreads())) {
      report.put("memory", pipeline.loadMemory(store, first, t));
      learned =
          pipeline.learn(
              t, first, store, assembler, config.learnerDevices(), replicate, report);
    }
    Path next = learned != null ? learned : Snapshot.iterate(pipeline.snapshot(t));
    Path snapshot = out.resolve("snapshot");
    if (Files.exists(snapshot)) Snapshot.deleteTree(snapshot);
    Snapshot.writeFiles(next, snapshot, PolicyParams.of(t + 1));
    report.put("seconds", (System.nanoTime() - started) / 1e9);
    writeAtomically(out.resolve("report.json"), GSON.toJson(report));
    LOG.info("refit t={} learner={} replicate={} -> {}", t, config.learner(), replicate, snapshot);
    return snapshot;
  }

  /**
   * 反復を回す。資源（PolicyExecutionContext・CollectionDriver・GRP・Φ・BatchAssembler）は反復をまたいで 1 つずつ持つ。各反復は上の
   * 1〜6 の順で、報告は 1 反復で 1 行。
   */
  private int loop(int stopAfter) throws Exception {
    NetworkDevices learnerDevices = config.learnerDevices();
    List<Device> collectDevices = PrismConfig.devices(config.collectDevices());
    var profile = EpsilonUtilityProfile.valueOf(config.utility());
    RunState state = loadOrInitialize(learnerDevices);
    Files.createDirectories(root.resolve("rows"));
    removeFinishedWork(state.next());
    try (var execution =
            new PolicyExecutionContext(config.encoders(), config.slots(), config.readyBatches());
        var driver =
            new CollectionDriver(
                new CollectionDriver.Settings(
                    config.maxActiveGames(),
                    config.advanceWorkers(),
                    new InferenceBatchingSettings(config.batchWaitMicros())));
        var grp =
            config.phi().equals(PrismConfig.PHI_GRP)
                ? EpsilonGrpTrainingSession.openInferenceOnly(
                    Path.of(config.grpCheckpoint()), PrismConfig.device(config.grpDevice()))
                : null;
        var phi =
            grp == null
                ? HandValue.scoreRank(profile)
                : new FixedGrpPhi(
                    grp.inference(), profile, driver.executor(), config.grpBatch(), PHI_MEMO);
        var assembler = new BatchAssembler(config.decodeThreads())) {
      if (grp == null)
        LOG.warn(
            "phi = {}: hand ends are valued by the final score ranking, not the fixed GRP (tests"
                + " only)",
            config.phi());
      int next = state.next();
      long totalRows = state.totalRows();
      int acceptedCount = state.accepted();
      double averageWeight = state.averageWeight();
      Averaging averaging = config.policyMemory();
      for (int done = 0; next <= config.iterations() && done < stopAfter; done++) {
        int t = next;
        long started = System.nanoTime();
        var report = new LinkedHashMap<String, Object>();
        report.put("iteration", t);
        report.put("phi", config.phi());
        // 1. 収集（版 t の b）
        totalRows += collect(t, driver, phi, profile, execution, collectDevices, report);
        report.put("collectSeconds", (System.nanoTime() - started) / 1e9);
        report.put("totalRows", totalRows);
        boolean accepted;
        double[] decodeBefore = assembler.stats();
        int first = Math.max(1, t - config.memoryCohorts() + 1);
        try (var store = new RowStore()) {
          report.put("memory", loadMemory(store, first, t));
          countRows(store, t, report);
          // 2〜5. 生徒（と蒸留・認証）と保留の指標
          Path learned = learn(t, first, store, assembler, learnerDevices, 0, report);
          accepted = learned != null;
          // 学習の網 θ_{t+1}: 受け入れたら学習した網、受け入れなければ θ_t。方策の記憶があれば打つ網 A_{t+1} = (1 − η)·A_t + η·θ_{t+1}
          Path nextIterate = accepted ? learned : Snapshot.iterate(snapshot(t));
          averageWeight = publish(t, nextIterate, averaging, averageWeight, report);
        }
        double[] decodeAfter = assembler.stats();
        double[] decode = new double[decodeAfter.length];
        for (int i = 0; i < decode.length; i++) decode[i] = decodeAfter[i] - decodeBefore[i];
        report.put("decode", decode);
        // 6. 記録と評価を state.json より前に書く（間で止まっても反復 t の記録と版 t+1 の評価が抜けない）。再開で同じ反復をやり直したときは、
        // 書き終えた記録と評価を書き直さない
        int version = t + 1;
        if (accepted) acceptedCount++;
        report.put("seconds", (System.nanoTime() - started) / 1e9);
        if (lastRecorded(ITERATIONS, "iteration") < t) append(ITERATIONS, report);
        LOG.info(
            String.format(
                "t=%d %s version %d (accepted %d of %d iterations) %.1fs",
                t,
                accepted ? "published the distilled net as" : "kept the previous net as",
                version,
                acceptedCount,
                t,
                (System.nanoTime() - started) / 1e9));
        // 評価（版 t+1 を 3×Heuristic と、貪欲）
        if (config.evalEvery() > 0
            && t % config.evalEvery() == 0
            && lastRecorded(EVALUATIONS, "version") < version) {
          var evaluation = new LinkedHashMap<String, Object>();
          evaluation.put("iteration", t);
          evaluation.put("version", version);
          Path published = snapshot(version);
          evaluation.putAll(
              evaluate(
                  version,
                  Snapshot.model(published),
                  "",
                  driver,
                  execution,
                  collectDevices,
                  profile));
          if (config.evaluateIterate() && averaging.enabled())
            evaluation.put(
                Averaging.ITERATE,
                evaluate(
                    version,
                    Snapshot.iterate(published),
                    " (iterate)",
                    driver,
                    execution,
                    collectDevices,
                    profile));
          append(EVALUATIONS, evaluation);
        }
        next = version;
        writeAtomically(
            root.resolve("state.json"),
            GSON.toJson(
                new RunState(RunState.FORMAT, next, totalRows, acceptedCount, averageWeight)));
        // 古い行と途中の網を消す（state.json の後。間で止まった残りは、次の起動の removeFinishedWork と次の反復のこの後始末が消す）
        for (int c = 1; c <= t - config.keepCohorts(); c++)
          if (Files.exists(cohort(c))) Snapshot.deleteTree(cohort(c));
        if (Files.exists(work(t))) Snapshot.deleteTree(work(t));
      }
      return 0;
    }
  }

  /**
   * 反復 t の学習（クラスの説明の 2〜5）。learner = crossfit なら生徒 A・B → 蒸留 → 認証、single なら全データの生徒 1 本。どちらも学習した網と
   * 版 t を、コホート t の HOLDOUT の行で測る（保留の指標）。replicate は並べ替えの種を変える（0 は train と同じ種）。
   *
   * @return 受け入れた学習の網のディレクトリ（work/&lt;t&gt; の中）。認証で受け入れなければ null
   */
  private Path learn(
      int t,
      int first,
      RowStore store,
      BatchAssembler assembler,
      NetworkDevices devices,
      int replicate,
      Map<String, Object> report)
      throws IOException {
    Device device = devices.primary();
    boolean accepted;
    String candidateName;
    if (config.singleStudent()) {
      fitStudent(
          t,
          SINGLE,
          trainRows(),
          learnSeed(t, 4, replicate),
          first,
          store,
          assembler,
          devices,
          report);
      accepted = true;
      candidateName = SINGLE;
      report.put("accepted", true);
      report.put("acceptance", "single");
    } else {
      // 生徒 A・B（版 t の重みから、自分の組の行だけで）
      fitStudent(
          t,
          PrismRow.Group.A.name(),
          studentRows(PrismRow.Group.A),
          learnSeed(t, 1, replicate),
          first,
          store,
          assembler,
          devices,
          report);
      fitStudent(
          t,
          PrismRow.Group.B.name(),
          studentRows(PrismRow.Group.B),
          learnSeed(t, 2, replicate),
          first,
          store,
          assembler,
          devices,
          report);
      try (var studentA = loadWork(t, PrismRow.Group.A.name(), device);
          var studentB = loadWork(t, PrismRow.Group.B.name(), device)) {
        // 蒸留（教師は凍結した A と B）
        distill(
            t,
            first,
            store,
            assembler,
            studentA,
            studentB,
            devices,
            learnSeed(t, 3, replicate),
            report);
        // 認証（組 B の根で A を、組 A の根で B を測る）
        Certification.Report certification =
            Certification.certify(
                store,
                assembler,
                studentA,
                studentB,
                store.select(t, rootRows()),
                config.batchSize());
        boolean passes =
            certification.passes(config.certifyTolerance(), config.certifyMaxOutsideShare());
        accepted = !config.certify() || passes;
        String acceptance = !config.certify() ? "uncertified" : passes ? "certified" : "rejected";
        report.put("certification", certification);
        report.put("accepted", accepted);
        report.put("acceptance", acceptance);
        logCertification(t, certification, acceptance);
      }
      candidateName = DISTILLED;
    }
    // 保留の指標（学習した網と版 t）
    try (var candidate = loadWork(t, candidateName, device);
        var current = Snapshot.loadFrozen(snapshot(t), device)) {
      int[] holdout = store.select(t, holdoutRows());
      FitSettings lambdas = config.fitSettings(config.fitSteps());
      HoldoutSkill.Report candidateSkill =
          HoldoutSkill.measure(store, assembler, candidate, holdout, config.batchSize(), lambdas);
      HoldoutSkill.Report currentSkill =
          HoldoutSkill.measure(store, assembler, current, holdout, config.batchSize(), lambdas);
      var skills = new LinkedHashMap<String, Object>();
      skills.put("candidate", candidateSkill);
      skills.put("current", currentSkill);
      report.put("holdout", skills);
      LOG.info(
          "t={} holdout rows={} candidate[{}] current[{}]",
          t,
          holdout.length,
          skillText(candidateSkill),
          skillText(currentSkill));
    }
    return accepted ? work(t).resolve(candidateName) : null;
  }

  /**
   * 学習に使う行。train は窓 first..t の selection の行の全て。refit は {@link #window} でコホートの範囲を変え、行ごとの硬貨
   * unit(derive(seed, MEMORY, THIN, 行)) &lt; 残す割合 で間引く（同じ索引なら何度選んでも同じ行。割合が小さいほど部分集合になる）。
   */
  private int[] trainingRows(
      RowStore store, int first, int t, Predicate<RowFormat.Index> selection) {
    int[] rows =
        store.select(
            window.from() > 0 ? window.from() : first,
            window.to() > 0 ? window.to() : t,
            selection);
    double keep = window.keep(rows.length);
    if (keep >= 1) return rows;
    return Arrays.stream(rows)
        .filter(
            row ->
                CrnSeeds.unit(CrnSeeds.derive(config.seed(), CrnSeeds.Domain.MEMORY, THIN, row))
                    < keep)
        .toArray();
  }

  /** 学習の並べ替えの種（replicate 0 は derive(seed, LEARN, t, salt)、それ以外は replicate も混ぜる）。 */
  private long learnSeed(int t, int salt, int replicate) {
    return replicate == 0
        ? CrnSeeds.derive(config.seed(), CrnSeeds.Domain.LEARN, t, salt)
        : CrnSeeds.derive(config.seed(), CrnSeeds.Domain.LEARN, t, salt, replicate);
  }

  /** TRAIN のラベル行で測った手が 2 つ以上の行（全データの生徒の行。両方の組）。 */
  static Predicate<RowFormat.Index> trainRows() {
    return index ->
        index.purpose() == PrismRow.Purpose.TRAIN
            && index.source().label()
            && index.measured() >= 2;
  }

  /** TRAIN のラベル行で測った手が 2 つ以上、組 group の行（生徒の行）。 */
  static Predicate<RowFormat.Index> studentRows(PrismRow.Group group) {
    return index ->
        index.purpose() == PrismRow.Purpose.TRAIN
            && index.group() == group
            && index.source().label()
            && index.measured() >= 2;
  }

  /** TRAIN のラベル行（蒸留の行。両方の組、|C| に依らない）。 */
  static Predicate<RowFormat.Index> distillRows() {
    return index -> index.purpose() == PrismRow.Purpose.TRAIN && index.source().label();
  }

  /** 根の記録（認証。両方の組・両方の用途）。 */
  static Predicate<RowFormat.Index> rootRows() {
    return index -> index.source() == PrismRow.Source.ROOT;
  }

  /** HOLDOUT のラベル行で測った手が 2 つ以上の行（保留の指標）。 */
  static Predicate<RowFormat.Index> holdoutRows() {
    return index ->
        index.purpose() == PrismRow.Purpose.HOLDOUT
            && index.source().label()
            && index.measured() >= 2;
  }

  /**
   * state.json を読む。なければ版 1 を書いて始める: startSnapshot があればその網の重み（epsilon-prism の版。幅は版のまま）を、なければ白紙の網
   * （幅 hidden）を、この run の構成（{@link PrismConfig#modelConfig}: バッチ・精度・転送・種）で書く。版の model.json の構成は
   * 引き継がない。
   */
  private RunState loadOrInitialize(NetworkDevices devices) throws IOException {
    Path path = root.resolve("state.json");
    if (Files.exists(path)) {
      RunState state = GSON.fromJson(Files.readString(path), RunState.class);
      if (state == null || state.format() != RunState.FORMAT)
        throw new IOException("state.json is not format " + RunState.FORMAT + ": " + path);
      return state;
    }
    Path start =
        config.startSnapshot().isEmpty() ? null : Snapshot.model(Path.of(config.startSnapshot()));
    if (start == null)
      LOG.warn(
          "starting from a blank network: s = 0 everywhere, so version 1 plays argmax p0");
    try (var model =
        start == null
            ? new PrismModel(
                config.modelConfig(devices.primary(), config.hidden()), "m0", devices.first(1))
            : PrismModel.load(
                start,
                config.modelConfig(devices.primary(), PrismModel.readConfig(start).hiddenSize()),
                devices.first(1),
                true)) {
      Snapshot.write(snapshot(1), model, PolicyParams.of(1));
    }
    // 版 1 は学習の網と打つ網が同じ（iterate/ を持たない）。power の重みの累計は版 1 の分から始める
    var state =
        new RunState(RunState.FORMAT, 1, 0, 0, config.policyMemory().step(1, 0).totalWeight());
    writeAtomically(path, GSON.toJson(state));
    return state;
  }

  /**
   * データの記憶: コホート first..t を store に積む。年齢 a = t − c のコホートはラベル行の割合 {@link PrismConfig#memoryKeep}(a) だけを載せる
   * （{@link RowStore#addCohort(Path, double, long)}。最新のコホート t は全行なので、認証の根と保留の行は変わらない）。消えたコホートは飛ばす。
   *
   * @return 報告の memory: cohorts（積んだコホート）、keep（その割合）、rows（積んだ行の数）
   */
  private Map<String, Object> loadMemory(RowStore store, int first, int t) throws IOException {
    List<Integer> cohorts = new ArrayList<>();
    List<Double> keeps = new ArrayList<>();
    for (int c = first; c <= t; c++) {
      if (!Files.exists(cohort(c).resolve(RowWriter.MANIFEST))) {
        LOG.warn("t={} memory skips cohort {} (not on disk)", t, c);
        continue;
      }
      double keep = config.memoryKeep(t - c);
      store.addCohort(cohort(c), keep, config.seed());
      cohorts.add(c);
      keeps.add(keep);
    }
    var memory = new LinkedHashMap<String, Object>();
    memory.put("cohorts", cohorts);
    memory.put("keep", keeps);
    memory.put("rows", store.size());
    LOG.info("t={} memory cohorts={} keep={} rows={}", t, cohorts, keeps, store.size());
    return memory;
  }

  /**
   * 版 t+1 を書く。方策の記憶がなければ学習の網 nextIterate をそのまま版 t+1 にする（ファイルの写し）。あれば打つ網を A_{t+1} = (1 − η)·A_t +
   * η·θ_{t+1}（{@link Averaging#step}、{@link PrismModel#mixIn}、CPU・FP32）にして、θ_{t+1} と一緒に書く（{@link Snapshot#writeAveraged}）。
   *
   * @param averageWeight 版 t までの power の重みの累計
   * @return 版 t+1 までの power の重みの累計
   */
  private double publish(
      int t, Path nextIterate, Averaging averaging, double averageWeight, Map<String, Object> report)
      throws IOException {
    int version = t + 1;
    PolicyParams params = PolicyParams.of(version);
    if (!averaging.enabled()) {
      Snapshot.writeFiles(nextIterate, snapshot(version), params);
      return 0;
    }
    Averaging.Step step = averaging.step(version, averageWeight);
    var record =
        new Snapshot.AverageRecord(
            averaging.scheme(), step.eta(), step.totalWeight(), averaging.from());
    NetworkDevices cpu = NetworkDevices.of(Device.cpu());
    try (var average = PrismModel.load(Snapshot.model(snapshot(t)), cpu, false);
        var iterate = PrismModel.load(nextIterate, cpu, false)) {
      average.mixIn(iterate, step.eta());
      Snapshot.writeAveraged(nextIterate, average, record, snapshot(version), params);
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

  /**
   * 終わった反復の work/&lt;c&gt;（c &lt; next）を消す。反復の終わりで消す前に止まったときの残りで、再開では使わない。work/&lt;next&gt; は続きに
   * 使うので残す。
   */
  private void removeFinishedWork(int next) throws IOException {
    Path work = workRoot;
    if (!Files.isDirectory(work)) return;
    List<Path> entries;
    try (Stream<Path> listed = Files.list(work)) {
      entries = listed.toList();
    }
    for (Path entry : entries) {
      int iteration;
      try {
        iteration = Integer.parseInt(entry.getFileName().toString());
      } catch (NumberFormatException other) {
        continue;
      }
      if (iteration < next) {
        Snapshot.deleteTree(entry);
        LOG.info("removed the leftover work of finished iteration {}", iteration);
      }
    }
  }

  /**
   * コホート t を集めて封印する。封印済みなら使い回す（{@link RowWriter#MANIFEST} がある）。途中のコホートは消して集め直す（その行で当てた
   * work/&lt;t&gt; も消す）。方策は版 t を {@link Snapshot#open}（p0 の係数は {@link PrismConfig#priorSettings}）で開き、{@link
   * CollectionDriver#collect} に Cohort(t, seed, cohortSeed(seed, t), cohortGames(t, gamesPerCohort), 効用, sampling(), behaviour(), null,
   * null) を渡す。
   *
   * @return そのコホートの行数（封印した manifest の値）
   */
  private long collect(
      int t,
      CollectionDriver driver,
      HandValue phi,
      EpsilonUtilityProfile profile,
      PolicyExecutionContext execution,
      List<Device> devices,
      Map<String, Object> report)
      throws IOException {
    Path cohort = cohort(t);
    if (Files.exists(cohort.resolve(RowWriter.MANIFEST))) {
      report.put("collection", "reused the sealed cohort");
      LOG.info("t={} reusing the sealed cohort", t);
      return manifest(cohort).rows();
    }
    if (Files.exists(cohort)) Snapshot.deleteTree(cohort);
    // 集め直す行で当てた網は使い回せない
    if (Files.exists(work(t))) Snapshot.deleteTree(work(t));
    CollectionDriver.Result collected;
    try (var policy =
            Snapshot.open(
                snapshot(t),
                Snapshot.net(snapshot(t), config.collectWith()),
                devices,
                execution,
                config.inferBatch(),
                config.multiTransitionBatch(),
                config.priorSettings());
        var writer = new RowWriter(cohort, t, config.rowWriters())) {
      collected =
          driver.collect(
              new CollectionDriver.Cohort(
                  t,
                  config.seed(),
                  PrismConfig.cohortSeed(config.seed(), t),
                  PrismConfig.cohortGames(t, config.gamesPerCohort()),
                  profile,
                  config.sampling(),
                  config.behaviour(),
                  null,
                  null),
              policy,
              phi,
              writer);
      writer.seal();
    }
    // 収集の終わりの受け口の数は書き手のキューに残る行を含まないので、封印した manifest の値に差し替える
    RowWriter.Manifest written = manifest(cohort);
    CollectionMetrics.Snapshot counts = collected.counts();
    if (written.rows() != counts.rowsWritten())
      throw new IllegalStateException(
          "Cohort "
              + t
              + " sealed "
              + written.rows()
              + " rows but "
              + counts.rowsWritten()
              + " were sent");
    collected = collected.withRows(new RowSink.RowStats(written.rows(), written.bytes()));
    report.put("collection", collected);
    report.put("copies", counts.copies());
    report.put("forks", counts.forks());
    LOG.info("t={} {}", t, collected.summary());
    return written.rows();
  }

  /** コホート t の行を索引から数える: labels（MAIN）・copyLabels（ROLLIN）・roots（ROOT）を、類の名前 → 行の数で。 */
  private static void countRows(RowStore store, int t, Map<String, Object> report) {
    long[][] counts = new long[PrismRow.Source.values().length][DecisionClass.COUNT];
    long holdout = 0;
    for (int row : store.select(t, index -> true)) {
      RowFormat.Index index = store.index(row);
      counts[index.source().ordinal()][index.decisionClass().ordinal()]++;
      if (index.purpose() == PrismRow.Purpose.HOLDOUT) holdout++;
    }
    Map<String, Long> labels = byClass(counts[PrismRow.Source.MAIN.ordinal()]);
    Map<String, Long> copyLabels = byClass(counts[PrismRow.Source.ROLLIN.ordinal()]);
    Map<String, Long> roots = byClass(counts[PrismRow.Source.ROOT.ordinal()]);
    report.put("labels", labels);
    report.put("copyLabels", copyLabels);
    report.put("roots", roots);
    LOG.info(
        "t={} rows labels={} copyLabels={} roots={} (holdout rows {})",
        t,
        labels,
        copyLabels,
        roots,
        holdout);
  }

  /** 類ごとの数を、0 でない類だけ類の順に並べた表にする。 */
  private static Map<String, Long> byClass(long[] counts) {
    Map<String, Long> result = new LinkedHashMap<>();
    for (DecisionClass c : DecisionClass.values())
      if (counts[c.ordinal()] > 0) result.put(c.name(), counts[c.ordinal()]);
    return result;
  }

  /**
   * 生徒（名前は "A"・"B"・"Single"）を selection の行で当てて work/&lt;t&gt;/名前 に保存する。保存済み（網と work/&lt;t&gt;/名前.json がそろって
   * いる）なら使い回す。行がないか fitSteps が 0 なら、版 t の重みをそのまま保存する。
   *
   * @param first 窓の最初のコホート
   * @param seed 行の並べ替えの種
   */
  private void fitStudent(
      int t,
      String name,
      Predicate<RowFormat.Index> selection,
      long seed,
      int first,
      RowStore store,
      BatchAssembler assembler,
      NetworkDevices devices,
      Map<String, Object> report)
      throws IOException {
    Path directory = work(t).resolve(name), record = work(t).resolve(name + ".json");
    String key = "fit" + name;
    if (Files.isDirectory(directory) && Files.exists(record)) {
      FitResult reused = GSON.fromJson(Files.readString(record), FitResult.class);
      report.put(key, reused);
      LOG.info("t={} reusing student {} ({})", t, name, directory);
      return;
    }
    int[] rows = trainingRows(store, first, t, selection);
    int steps =
        PrismConfig.steps(config.fitSteps(), config.fitEpochs(), rows.length, config.batchSize());
    FitResult result;
    Path start = Snapshot.net(snapshot(t), config.learnFrom());
    try (var student = PrismModel.load(start, devices, true)) {
      result =
          steps == 0
              ? FitResult.empty(name, rows.length)
              : StudentFit.fit(
                  name,
                  store,
                  assembler,
                  student,
                  rows,
                  config.fitSettings(steps),
                  seed);
      saveModel(student, directory);
    }
    writeAtomically(record, GSON.toJson(result));
    report.put(key, result);
    logFit(t, result);
  }

  /**
   * 蒸留して work/&lt;t&gt;/distilled に保存する。保存済み（網と work/&lt;t&gt;/distill.json がそろっている）なら使い回す。生徒は版 t の重み
   * （学習用）、教師は凍結した生徒 A・B。行がないか distillSteps が 0 なら、版 t の重みをそのまま保存する。
   *
   * @param first 窓の最初のコホート
   */
  private void distill(
      int t,
      int first,
      RowStore store,
      BatchAssembler assembler,
      PrismModel teacherA,
      PrismModel teacherB,
      NetworkDevices devices,
      long seed,
      Map<String, Object> report)
      throws IOException {
    Path directory = work(t).resolve(DISTILLED), record = work(t).resolve(DISTILL_RECORD);
    if (Files.isDirectory(directory) && Files.exists(record)) {
      FitResult reused = GSON.fromJson(Files.readString(record), FitResult.class);
      report.put("distill", reused);
      LOG.info("t={} reusing the distilled net ({})", t, directory);
      return;
    }
    int[] rows = trainingRows(store, first, t, distillRows());
    int steps =
        PrismConfig.steps(
            config.distillSteps(), config.distillEpochs(), rows.length, config.batchSize());
    FitResult result;
    Path start = Snapshot.net(snapshot(t), config.learnFrom());
    try (var student = PrismModel.load(start, devices, true)) {
      result =
          steps == 0
              ? FitResult.empty("distill", rows.length)
              : DistillFit.distill(
                  store,
                  assembler,
                  student,
                  teacherA,
                  teacherB,
                  rows,
                  config.fitSettings(steps),
                  seed);
      saveModel(student, directory);
    }
    writeAtomically(record, GSON.toJson(result));
    report.put("distill", result);
    logFit(t, result);
  }

  /** work/&lt;t&gt;/name の網を推論用に読む（凍結。{@link Snapshot#loadFrozen} と同じ読み方）。 */
  private PrismModel loadWork(int t, String name, Device device) throws IOException {
    return PrismModel.load(work(t).resolve(name), NetworkDevices.of(device), false);
  }

  private static void logFit(int t, FitResult result) {
    double[] stages = result.stageSeconds();
    LOG.info(
        String.format(
            "t=%d fit %s rows=%d steps=%d clipped=%d loss=%.4e first=%.4e last=%.4e %.1fs"
                + " stages=[pack %.0fs transfer %.0fs forward/backward %.0fs optimizer %.0fs]",
            t,
            result.name(),
            result.rows(),
            result.steps(),
            result.clippedSteps(),
            result.loss(),
            result.firstLoss(),
            result.lastLoss(),
            result.seconds(),
            stage(stages, 0),
            stage(stages, 1),
            stage(stages, 2),
            stage(stages, 3)));
  }

  private static double stage(double[] stages, int index) {
    return stages != null && index < stages.length ? stages[index] : Double.NaN;
  }

  private static void logCertification(int t, Certification.Report report, String acceptance) {
    LOG.info(
        String.format(
            "t=%d certification commitment mean=%.6f se=%.6f lower=%.6f outside=%.4f -> %s"
                + " | A on %s: %s | B on %s: %s (%.1fs)",
            t,
            report.commitmentMean(),
            report.commitmentError(),
            report.lowerBound(),
            report.outsideShare(),
            acceptance,
            report.a() == null ? "-" : report.a().evaluatedOn(),
            sideText(report.a()),
            report.b() == null ? "-" : report.b().evaluatedOn(),
            sideText(report.b()),
            report.seconds()));
  }

  private static String sideText(Certification.Side side) {
    if (side == null) return "-";
    return String.format(
        "commitment %s all %s", estimateText(side.commitment()), estimateText(side.all()));
  }

  private static String estimateText(Certification.Estimate estimate) {
    if (estimate == null) return "-";
    return String.format(
        "I=%.6f±%.6f per decision (roots %d decisions %.0f games %d switches %d hits %d outside %d)",
        estimate.value(),
        estimate.error(),
        estimate.roots(),
        estimate.decisions(),
        estimate.games(),
        estimate.switches(),
        estimate.hits(),
        estimate.outside());
  }

  private static String skillText(HoldoutSkill.Report report) {
    if (report == null || report.all() == null) return "-";
    HoldoutSkill.Decided decided = report.all().decided();
    String decidedText =
        decided == null ? "-" : String.format("%.3f of %d", decided.rate(), decided.rows());
    return "raw "
        + pairText(report.all().pairs())
        + " debiased "
        + pairText(report.all().debiased())
        + " decided="
        + decidedText;
  }

  private static String pairText(HoldoutSkill.Pairs pairs) {
    if (pairs == null) return "-";
    return String.format(
        "S=%.3e±%.1e explained=%.3f slope=%.3f corr=%.3f sign=%.3f",
        pairs.skill(),
        pairs.skillError(),
        pairs.explained(),
        pairs.slope(),
        pairs.correlation(),
        pairs.signAgreement());
  }

  /**
   * 版 version の網 net（打つ網か学習の網）を 3×Heuristic と打たせる（貪欲）。結果と対局の内訳。
   *
   * @param label 記録の文字列に添える名前（打つ網は空）
   */
  private Map<String, Object> evaluate(
      int version,
      Path net,
      String label,
      CollectionDriver driver,
      PolicyExecutionContext execution,
      List<Device> devices,
      EpsilonUtilityProfile profile)
      throws IOException {
    var stats = new PlayStats();
    Arena.Result result;
    try (var candidate =
            Snapshot.open(
                snapshot(version),
                net,
                devices,
                execution,
                config.inferBatch(),
                config.multiTransitionBatch(),
                config.priorSettings());
        var opponent = PlayerPolicy.heuristic(0)) {
      result =
          Arena.duel(
              driver,
              candidate,
              opponent,
              "heuristic",
              config.evalGames(),
              config.seed() ^ Arena.EVALUATION_SALT,
              Arena.HEURISTIC_BASE,
              profile,
              stats);
    }
    Map<String, Object> play = stats.report();
    LOG.info(
        String.format(
            "version %d%s vs 3xHeuristic: games=%d rank=%.3f±%.3f utility=%.4f±%.4f"
                + " shares=%.3f/%.3f/%.3f/%.3f %.1fs",
            version,
            label,
            result.games(),
            result.meanRank(),
            result.rankError(),
            result.meanUtility(),
            result.utilityError(),
            result.rankShares()[0],
            result.rankShares()[1],
            result.rankShares()[2],
            result.rankShares()[3],
            result.seconds()));
    LOG.info("version {}{} play {}", version, label, LINES.toJson(play));
    var map = new LinkedHashMap<String, Object>();
    map.put("heuristic", result);
    map.put("play", play);
    return map;
  }

  /** 網を directory.tmp に書いてから directory に差し替える（途中で止まっても読める網が残る）。 */
  static void saveModel(PrismModel model, Path directory) throws IOException {
    Path temporary = directory.resolveSibling(directory.getFileName() + ".tmp");
    if (Files.exists(temporary)) Snapshot.deleteTree(temporary);
    model.save(temporary);
    if (Files.exists(directory)) Snapshot.deleteTree(directory);
    Files.move(temporary, directory, StandardCopyOption.ATOMIC_MOVE);
  }

  static RowWriter.Manifest manifest(Path cohort) throws IOException {
    return RowWriter.readManifest(cohort);
  }

  /** 版 t のディレクトリ。 */
  Path snapshot(int t) {
    return root.resolve("snapshots").resolve(String.format("%05d", t));
  }

  /** コホート t の行のディレクトリ。 */
  Path cohort(int t) {
    return root.resolve("rows").resolve(String.format("%05d", t));
  }

  /** 反復 t の途中の網（A・B・distilled・Single）と記録のディレクトリ。反復を終えたら消す。 */
  Path work(int t) {
    return workRoot.resolve(String.format("%05d", t));
  }

  /** reports/name の最後の記録の鍵 key の整数（ファイルか記録か鍵がなければ 0）。再開で同じ反復の記録と評価を書き直さないために使う。 */
  int lastRecorded(String name, String key) throws IOException {
    Path path = root.resolve("reports").resolve(name);
    if (!Files.exists(path)) return 0;
    List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
    for (int i = lines.size() - 1; i >= 0; i--) {
      String line = lines.get(i).trim();
      if (line.isEmpty()) continue;
      JsonElement value;
      try {
        value = JsonParser.parseString(line).getAsJsonObject().get(key);
      } catch (RuntimeException truncated) {
        // 書きかけで止まった行は読み飛ばす
        continue;
      }
      return value == null || value.isJsonNull() ? 0 : value.getAsInt();
    }
    return 0;
  }

  void append(String name, Map<String, Object> record) throws IOException {
    Path reports = root.resolve("reports");
    Files.createDirectories(reports);
    Files.writeString(
        reports.resolve(name),
        LINES.toJson(record) + "\n",
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE,
        StandardOpenOption.APPEND);
  }

  static void writeAtomically(Path path, String content) throws IOException {
    Files.createDirectories(path.getParent());
    Path temporary = path.resolveSibling(path.getFileName() + ".tmp");
    Files.writeString(temporary, content, StandardCharsets.UTF_8);
    Files.move(
        temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
  }
}
