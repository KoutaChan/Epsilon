package com.epsilon.prism.app;

import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.Span;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.Difference;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.learn.AnchorScores;
import com.epsilon.prism.learn.CohortStore;
import com.epsilon.prism.learn.Fit;
import com.epsilon.prism.learn.FitPlan;
import com.epsilon.prism.learn.FitResult;
import com.epsilon.prism.learn.Selection;
import com.epsilon.prism.learn.Teacher;
import com.epsilon.prism.luck.LedgerOutcome;
import com.epsilon.prism.luck.LuckSettlement;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism.noise.TempoSettings;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.settings.FittingSettings;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * run に保存したコホート（rows/&lt;t&gt;）でネットワークを学習し直し、バージョンの形で out/snapshot に書く。バージョン番号は最後の
 * コホートの番号 + 1 にする。学習の値は run の最新の設定の行から読む。{@code --set} で変えられるのは、学習から効くキー（FIT）だけ。run は
 * 読むだけで書かない。
 *
 * <p>教師は 2 つある。{@link Teacher#RAW RAW} は、範囲の全てのコホートから TRAIN のラベル行と根の記録（{@link Selection#LABELS_AND_ROOTS}）を
 * 選び、生の Y でアンカーなしに {@code fit.epochs} エポック学ぶ。{@link Mode#LINEAGE LINEAGE} は、系譜の更新をコホートごとに順にたどる。
 * まず生徒の行（{@code fit.rows}）で運の精算を当てはめる（noise.ledger）。当てはめは学習の段と同じ {@link LuckSettlement#learn} で行い、
 * 報告の全部をコホートごとに out/{@value #LEDGER}/&lt;t&gt;.json に書く。次に今のネットワークをアンカーにし（noise.tempo）、K′ と学習率の
 * 倍率を前のコホートの R² から決める。ステップ数は {@code fit.steps} で決める。
 *
 * <p>始めのネットワークは、{@code --start} があれば、そのバージョンの重みにする（バージョンの形は学ぶ形と同じであること）。無ければゼロから
 * 学ぶ。形は {@code --shape} のアーム（{@code --depth} と一緒に渡す）にし、無ければ run の network の形にする。
 *
 * @param run 行を読む run のディレクトリ
 * @param cohorts 学ぶコホートの範囲（a..b）
 * @param out 書き出すディレクトリ
 * @param teacher 教師の作り方
 * @param shape 形のアーム（無ければ run の network の形）
 * @param depth 形のアームのバックボーンの層の数（V2 のアームでは必須、v3 のアームは 3 だけ）
 * @param start 始めのネットワークのバージョンのディレクトリ（無ければゼロから学ぶ）
 * @param maxRows 学ぶ行の数の上限（試しに学ぶときに使い、選んだ行の先頭から取る。無ければ全ての行）
 * @param appliedR2 前のコホートで教師に掛けた補正の R²。LINEAGE の最初のコホートの K′ を決めるのに使う（無ければ持ち越さない）
 * @param replicate アームの複製の番号（ネットワークの初期値と、行の並べ替えの乱数の種を変える）
 */
@Named(value = "fit-cohorts", doc = "Refit a network on saved cohorts")
public record FitCohorts(
    @Key(doc = "Run directory to read the rows from", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Range of cohorts to learn from (a..b)", presence = Presence.REQUIRED) String cohorts,
    @Key(doc = "Output directory", presence = Presence.REQUIRED) Path out,
    @Key(doc = "How to build the teacher (RAW: learn the raw Y for fit.epochs epochs; LINEAGE: replay the lineage updates cohort by cohort)", value = "RAW") Mode teacher,
    @Key(doc = "Shape arm (default: the run's network shape)", presence = Presence.ABSENT) ShapeArm shape,
    @Key(doc = "Backbone layer count of the shape arm (required for V2 arms; v3 arms take only 3)", presence = Presence.ABSENT)
        @Range(min = PrismShape.MIN_DEPTH, max = PrismShape.MAX_DEPTH)
        Integer depth,
    @Key(doc = "Version directory of the starting network (default: learn from scratch)", presence = Presence.ABSENT) Path start,
    @Key(doc = "Cap on the rows to learn (for trial fits; takes the first selected rows)", presence = Presence.ABSENT) @Range(min = 1) Integer maxRows,
    @Key(doc = "R² of the teacher correction on the previous cohort; sets K′ of LINEAGE's first cohort (default: no carry-over)", presence = Presence.ABSENT)
        @Range(min = 0, max = 1)
        Double appliedR2,
    @Key(doc = "Replica number of the arm (changes the network init and the row shuffle seed)", value = "0") @Range(min = 0) int replicate)
    implements PrismCommand {

  /** 教師の作り方。 */
  public enum Mode {
    /** 範囲の全ての行を、生の Y でアンカーなしに {@code fit.epochs} エポック学ぶ。 */
    RAW,
    /** 系譜の更新（運の精算、TEMPO のアンカー、K′、学習率の倍率）をコホートごとに順にたどる。 */
    LINEAGE
  }

  /** バージョンを書くディレクトリの名前。 */
  public static final String SNAPSHOT = "snapshot";

  /** LINEAGE のアンカー（コホートを学ぶ前のネットワーク）を置くディレクトリの名前。終わったら消す。 */
  static final String ANCHOR = "anchor";

  /** LINEAGE が当てはめた帳簿の報告の全部を、コホートごとに書くディレクトリの名前。形は学習の段の ledger.json と同じ。 */
  public static final String LEDGER = "ledger";

  private static final Logger LOG = LoggerFactory.getLogger(FitCohorts.class);

  /** アームの乱数の種を導くときの座標（"v2ar"）。 */
  private static final long ARM = 0x76326172L;

  public FitCohorts {
    Span.of("cohorts", cohorts);
    if (shape == null && depth != null) {
      throw new InvalidSetting("shape", "--depth shapes an arm: give --shape too");
    }
    if (shape == ShapeArm.V2 && depth == null) {
      throw new InvalidSetting("depth", "the V2 arm needs --depth");
    }
    if (shape != null && shape != ShapeArm.V2 && depth != null && depth != ShapeArm.V3_DEPTH) {
      throw new InvalidSetting("depth", "the " + shape + " arm has depth " + ShapeArm.V3_DEPTH + ", not " + depth);
    }
    if (teacher == Mode.RAW && appliedR2 != null) {
      throw new InvalidSetting("appliedR2", "--applied-r2 applies to the LINEAGE teacher only");
    }
  }

  /** 学ぶコホートの範囲。 */
  public Span span() {
    return Span.of("cohorts", cohorts);
  }

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    Span span = span();
    PrismSettings settings = settings(run, context.layers()).value();
    settings.runtime().apply();
    settings.diagnostics().apply();
    PrismShape network = network(settings);
    NetworkDevices devices = settings.learner().networkDevices();
    try (CohortStore store = new CohortStore()) {
      for (int cohort = span.first(); cohort <= span.last(); cohort++) {
        store.addCohort(CohortWriter.directory(run, cohort));
      }
      try (Model model =
          Fit.blank(InputFormat.of(store.variant()), network, devices.primary(), seed(settings, span.last()))) {
        PrismNetwork fitted = (PrismNetwork) model.getBlock();
        if (start != null) {
          try (NDManager manager = NDManager.newBaseManager(devices.primary())) {
            PrismNetwork first = PrismNetwork.load(Snapshot.model(start), manager);
            if (!first.shape().equals(network)) {
              throw new IllegalArgumentException(
                  "--start " + start + " has the shape " + first.shape() + ", the fit learns " + network);
            }
            LOG.info("fit-cohorts starts from {} ({})", start, network);
            fitted.mixIn(first, 1);
          }
        }
        Fitted result =
            teacher == Mode.RAW
                ? raw(settings, store, span, model, network, devices)
                : lineage(settings, store, span, model, devices);
        Path snapshot = out.resolve(SNAPSHOT);
        Snapshot.write(
            snapshot,
            fitted,
            PolicyParams.of(span.last() + 1),
            new Snapshot.FitRecord(run.toString(), span.first(), span.last(), replicate, result.plan(), result.result()));
        LOG.info(
            "fit-cohorts {} teacher={} steps={} loss {} -> {} clipped={} {}s -> {}",
            span,
            teacher,
            result.result().steps(),
            result.result().firstLoss(),
            result.result().lastLoss(),
            result.result().clippedSteps(),
            String.format(Locale.ROOT, "%.1f", result.result().seconds()),
            snapshot);
        return Outcome.ok("fit-cohorts " + span + " -> " + snapshot);
      }
    }
  }

  /** 学習の計画と結果（LINEAGE では最後のコホートのもの）。 */
  private record Fitted(FitPlan plan, FitResult result) {}

  /** 範囲の全ての行を、生の Y でアンカーなしに {@code fit.epochs} エポック学ぶ。ステップ数は fit.maxSteps に依らない。 */
  private Fitted raw(
      PrismSettings settings, CohortStore store, Span span, Model model, PrismShape network, NetworkDevices devices) {
    FittingSettings fitting = settings.fit();
    int[] rows = limit(store.select(span.first(), span.last(), Selection.LABELS_AND_ROOTS));
    if (rows.length == 0) {
      throw new IllegalStateException("No training rows in cohorts " + span + " of " + run);
    }
    int steps = (int) Math.max(1, (long) Math.ceil(fitting.epochs() * rows.length / fitting.batch()));
    FitPlan plan =
        new FitPlan(fitting.fit(steps, 1), network, fitting.auxiliary(), FitPlan.TOKEN_BUDGET, seed(settings, span.last()));
    LOG.info(
        "fit-cohorts {} raw rows={} steps={} shape={} auxiliary={} devices={}",
        span,
        rows.length,
        steps,
        network,
        fitting.auxiliary(),
        devices.devices());
    FitResult result = Fit.run(store, rows, model, plan, Teacher.RAW, devices, settings.learner().decodeThreads());
    return new Fitted(plan, result);
  }

  /** 系譜の更新をコホートごとに順にたどる。アンカーは各コホートを学ぶ前のネットワークで、out/{@value #ANCHOR} に置き、終わったら消す。 */
  private Fitted lineage(PrismSettings settings, CohortStore store, Span span, Model model, NetworkDevices devices)
      throws IOException {
    PrismNetwork network = (PrismNetwork) model.getBlock();
    FittingSettings fitting = settings.fit();
    LuckSettlement luck = LuckSettlement.of(settings.noise(), settings.run().variant());
    TempoSettings tempo = settings.noise().tempo();
    int threads = settings.learner().decodeThreads();
    Path anchorDirectory = out.resolve(ANCHOR);
    double r2 = appliedR2 == null ? 0 : appliedR2;
    Fitted last = null;
    try {
      for (int t = span.first(); t <= span.last(); t++) {
        int[] rows = limit(store.select(t, t, fitting.rows()));
        LedgerOutcome ledger = LedgerOutcome.NONE;
        if (rows.length > 0) {
          try (ExecutorService readers = Executors.newFixedThreadPool(threads)) {
            ledger = luck.learn(t, null, store.iterate(rows, fitting.batch(), readers, threads), rows.length);
          }
          if (ledger.report() != null) {
            DurableFiles.write(
                out.resolve(LEDGER).resolve(String.format(Locale.ROOT, "%05d.json", t)), ledger.report().full());
          }
        }
        double gain = tempo.gain(t, r2);
        double lrScale = tempo.learningRateScale(t, r2);
        AnchorScores anchor = null;
        if (tempo.anchored() && rows.length > 0) {
          DurableFiles.delete(anchorDirectory);
          network.save(anchorDirectory);
          anchor =
              AnchorScores.compute(
                  store, rows, anchorDirectory, t, InferencePath.of(settings.inference().kernels()), devices, threads);
          anchor.requireVersion(t);
        }
        FitPlan plan =
            new FitPlan(
                fitting.fit(fitting.steps(rows.length), lrScale),
                network.shape(),
                fitting.auxiliary(),
                FitPlan.TOKEN_BUDGET,
                CrnSeeds.derive(settings.run().seed(), CrnSeeds.Domain.LEARN, t, ARM, replicate));
        FitResult result =
            plan.learner().steps() == 0
                ? FitResult.empty(rows.length)
                : Fit.run(
                    store,
                    rows,
                    model,
                    plan,
                    new Teacher(ledger.correction(), anchor, anchor == null ? 0 : 1 - gain, (float) gain),
                    devices,
                    threads);
        LOG.info(
            String.format(
                Locale.ROOT,
                "fit-cohorts t=%d lineage rows=%d steps=%d gain=%.4f lrScale=%.4f r2=%.4f loss %s -> %s %.1fs",
                t,
                rows.length,
                result.steps(),
                gain,
                lrScale,
                r2,
                result.firstLoss(),
                result.lastLoss(),
                result.seconds()));
        r2 = ledger.appliedR2();
        last = new Fitted(plan, result);
      }
    } finally {
      if (Files.exists(anchorDirectory)) {
        Snapshot.deleteTree(anchorDirectory);
      }
    }
    return last;
  }

  /** 始めのネットワークの形。アームがあればアームの形、無ければ run の network の形。 */
  private PrismShape network(PrismSettings settings) {
    if (shape == null) {
      return settings.network().shape();
    }
    int layers = depth != null ? depth : ShapeArm.V3_DEPTH;
    return shape.shape(layers);
  }

  /** 行を上限の数で切ったもの（上限が無ければそのまま）。 */
  private int[] limit(int[] rows) {
    return maxRows != null && rows.length > maxRows ? Arrays.copyOf(rows, maxRows) : rows;
  }

  /** ネットワークの初期値と、RAW の行の並べ替えに使う乱数の種。 */
  private long seed(PrismSettings settings, int lastCohort) {
    return CrnSeeds.derive(settings.run().seed(), CrnSeeds.Domain.LEARN, lastCohort, ARM, replicate);
  }

  /**
   * run の最新の設定の行に、呼び出しの層（{@code --set} など）を重ねた設定を返す。
   *
   * @throws SettingsException 層が、学習から効くキーのほかを変えているとき
   */
  static Settings<PrismSettings> settings(Path run, List<Layer> layers) throws IOException {
    SettingsRow last = new SettingsRows(new RunDirectory(run)).latest();
    Settings<PrismSettings> recorded = last.bind(PrismSettings.class);
    if (layers.isEmpty()) {
      return recorded;
    }
    List<Layer> stacked = new ArrayList<>();
    stacked.add(last.layer());
    stacked.addAll(layers);
    Settings<PrismSettings> settings = Settings.bind(PrismSettings.class, stacked);
    List<String> refused = new ArrayList<>();
    for (Difference difference : settings.since(recorded)) {
      if (difference.change() != Change.FIT) {
        refused.add(
            "Setting " + difference.key() + " cannot change in fit-cohorts: it is a " + difference.change() + " key");
      }
    }
    if (!refused.isEmpty()) {
      throw new SettingsException(refused);
    }
    return settings;
  }
}
