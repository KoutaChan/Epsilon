package com.epsilon.prism.luck;

import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.noise.LedgerCalibration;
import com.epsilon.prism.noise.RootCoefficients;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 時間順に別のコホートで係数、縮小強度、倍率の曲線、曲線の形を決める。 */
public final class RootTraining {
  private static final Logger LOG = LoggerFactory.getLogger(RootTraining.class);
  private static final int STATE_VERSION = 4;

  /** 完了した反復の状態。次に学ぶ係数と、教師に使える凍結済みの係数を持つ。 */
  public record State(
      int version,
      int cohort,
      int first,
      @Key(doc = "この周期で帳簿の行がなく学習を休んだ世代数", value = "0") int skipped,
      Variant variant,
      LedgerMeaning meaning,
      RootCoefficients settings,
      @Key(doc = "監査した反復の群別の符号付き補正", value = "{}") Map<String, LedgerReport.Paired> signed,
      @Key(doc = "訓練世代で得た対称行列の上三角", presence = Presence.ABSENT) String moments,
      @Key(doc = "次世代で縮小強度を選んだ係数", presence = Presence.ABSENT) RootModel.Weights weights,
      List<Candidate> curves,
      @Key(doc = "仕様の選択を終えた係数と倍率", presence = Presence.ABSENT) RootModel active) {
    public State {
      int phase = cohort - first - skipped;
      if ((version < 1 || version > STATE_VERSION)
          || first < 1
          || skipped < 0
          || phase < -1
          || phase
              >= (long) settings.trainingCohorts()
                  + settings.curveCohorts()
                  + settings.selectionCohorts()) {
        throw new IllegalArgumentException("Invalid root ledger generation");
      }
      boolean accumulating = phase >= 0 && phase < settings.trainingCohorts();
      boolean validated = phase >= settings.trainingCohorts();
      if (accumulating != (moments != null)
          || validated != (weights != null)
          || (weights != null
              && (weights.first() != first
                  || weights.variant() != variant
                  || !weights.settings().equals(settings)))) {
        throw new IllegalArgumentException(
            "Root ledger checkpoint does not match its training phase");
      }
      int foldCount = Math.clamp(phase - settings.trainingCohorts(), 0, settings.curveCohorts());
      int specs = foldCount == 0 ? 0 : settings.contexts().size() * settings.directions().size();
      if (curves.size() != specs
          || curves.stream().anyMatch(candidate -> candidate.rates().folds().size() != foldCount)) {
        throw new IllegalArgumentException("Invalid root ledger curve history");
      }
      if (active != null
          && (active.selectedThrough() >= first
              || active.weights().variant() != variant
              || !active.weights().settings().equals(settings))) {
        throw new IllegalArgumentException(
            "Root ledger predictor overlaps its next training cohort");
      }
      curves = List.copyOf(curves);
      signed = Map.copyOf(signed);
    }

    /** 指定した反復を閉じたときの状態を読む。 */
    public static State read(Path directory, int cohort) throws IOException {
      var file = file(directory, cohort);
      if (!Files.exists(file)) return null;
      var state = RecordFile.read(State.class, file).value();
      if (state.cohort() != cohort)
        throw new IOException("Ledger checkpoint cohort mismatch in " + file);
      return state;
    }

    public static Path file(Path directory, int cohort) {
      return directory
          .resolve("root-coefficients")
          .resolve(String.format(Locale.ROOT, "%05d.json", cohort));
    }

    public void write(Path directory) throws IOException {
      var path = file(directory, cohort);
      DurableFiles.createDirectories(path.getParent());
      DurableFiles.write(path, this);
    }

    public boolean matches(int next, Variant v, LedgerMeaning ledger, RootCoefficients cfg) {
      return mismatch(next, v, ledger, cfg) == null;
    }

    private String mismatch(int next, Variant v, LedgerMeaning ledger, RootCoefficients cfg) {
      int expected = next - 1;
      if (cohort != expected) return "expected cohort " + expected + ", found " + cohort;
      if (variant != v) return "variant changed from " + variant + " to " + v;
      if (!meaning.equals(ledger.scaleFree())) return "ledger meaning changed";
      if (!settings.equals(cfg.fitSettings())) return "root coefficient settings changed";
      return null;
    }
  }

  static State restartAfter(
      int cohort, Variant variant, LedgerMeaning meaning, RootCoefficients settings) {
    return new State(
        STATE_VERSION,
        cohort,
        cohort + 1,
        0,
        variant,
        meaning.scaleFree(),
        settings.fitSettings(),
        Map.of(),
        null,
        null,
        List.of(),
        null);
  }

  /** 倍率の候補と、選択に使った世代での生に対する損失差の和。 */
  public record Candidate(
      RootRates rates,
      double selectionLoss,
      @Key(doc = "同じ保留行の腕別の損失と対応差", presence = Presence.ABSENT) RootHoldout.Scores scores) {}

  private final int cohort, first, offset;
  private final Variant variant;
  private final LedgerMeaning meaning;
  private final RootCoefficients config;
  private final int[] columns;
  private final State before;
  final String restart;
  private final RootFeatures.Reader reader;
  private final RootRidge gram;
  private final RootHoldout holdout;
  private final RootFeatures.Layout layout;
  private final RootModel.Predictor estimator;
  private final Map<LedgerCell.Standing, double[]> moments = new HashMap<>();
  private int[] measured = new int[0];
  private double[] predictions = new double[0];
  private final double[] base;
  private long missing;
  private long rows;

  RootTraining(
      int cohort,
      Variant v,
      LedgerMeaning meaning,
      RootCoefficients cfg,
      State carried,
      LedgerCalibration.Apply reference,
      LedgerCorrection baseline) {
    this.cohort = cohort;
    this.variant = v;
    this.meaning = meaning.scaleFree();
    this.config = cfg.fitSettings();
    restart =
        carried == null
            ? "no preceding root checkpoint"
            : carried.mismatch(cohort, v, meaning, cfg);
    before = restart == null ? carried : null;
    if (restart != null) LOG.warn("t={} starting root ledger training: {}", cohort, restart);
    first = before == null ? cohort : before.first;
    offset = cohort - first - (before == null ? 0 : before.skipped());
    holdout =
        offset > cfg.trainingCohorts() + cfg.curveCohorts()
            ? new RootHoldout(cohort, v.seats(), before.curves(), reference, baseline)
            : null;
    columns =
        meaning.columns().stream()
            .filter(
                c ->
                    meaning.apply().contains(c.group())
                        && (cfg.information() != RootCoefficients.Information.ROOT_HANDS
                            || c.centeredGivenHands()))
            .mapToInt(Enum::ordinal)
            .toArray();
    reader = new RootFeatures.Reader(v);
    layout = RootFeatures.Layout.of(v, cfg);
    gram =
        offset <= cfg.trainingCohorts()
            ? new RootRidge(
                layout.width,
                columns.length,
                before != null && offset < cfg.trainingCohorts() ? before.moments : null)
            : null;
    estimator =
        before != null && before.weights != null
            ? new RootModel.Predictor(before.weights, null)
            : null;
    base = new double[columns.length];
  }

  void parallelize(java.util.concurrent.Executor executor, int capacity) {
    if (gram != null) gram.parallelize(executor, capacity);
  }

  RootModel active() {
    return before == null ? null : before.active;
  }

  long missing() {
    return missing;
  }

  void add(FitRow row, RootAudit audit) {
    float[] y = row.values();
    if (measured.length < y.length) {
      measured = new int[y.length];
      predictions = new double[y.length];
    }
    int n = LedgerFit.Accumulator.measured(row, y, measured);
    if (n < 0) return;
    rows++;
    boolean hands = config.information() == RootCoefficients.Information.ROOT_HANDS;
    RootFeatures root = row.root(reader, measured, n, hands);
    if (audit != null) audit.add(row, root, measured, n, y);
    if (hands && !root.hands()) {
      missing++;
      if (holdout == null) return;
    }
    if (gram != null) {
      gram.add(row, root, layout, measured, n, columns, y);
      return;
    }
    LedgerCell.Standing s = root.standing();
    double h = estimator.prepare(root, s, base), my = 0, mb = 0;
    for (int j = 0; j < n; j++) {
      my += y[measured[j]];
      predictions[j] = estimator.amount(root, j, row, measured[j], base, h, null);
      mb += predictions[j];
    }
    my /= n;
    mb /= n;
    double a = 0, b = 0;
    for (int j = 0; j < n; j++) {
      double d = predictions[j] - mb;
      a += d * d;
      b += d * (y[measured[j]] - my);
    }
    if (holdout != null) {
      holdout.add(row, s, y, measured, n, my, a / n, b / n);
      return;
    }
    double[] sum = moments.computeIfAbsent(s, unused -> new double[2]);
    sum[0] += a / n;
    sum[1] += b / n;
  }

  Map<String, LedgerReport.Paired> previousSigned() {
    return before == null ? Map.of() : before.signed();
  }

  State finish(Map<String, LedgerReport.Paired> signed) {
    RootModel active = active();
    RootModel.Weights weights = before == null ? null : before.weights;
    List<Candidate> curves = before == null ? List.of() : before.curves;
    if (rows == 0) {
      boolean started = before != null && before.cohort() >= first;
      return new State(
          STATE_VERSION,
          cohort,
          started ? first : cohort + 1,
          started ? before.skipped() + 1 : 0,
          variant,
          meaning,
          config,
          signed,
          before == null ? null : before.moments(),
          weights,
          curves,
          active);
    }
    int nextFirst = first;
    String packed = null;
    if (offset < config.trainingCohorts()) packed = gram.save();
    else if (offset == config.trainingCohorts()) {
      var training = new RootRidge(layout.width, columns.length, before.moments);
      var solution = training.select(gram, layout);
      weights =
          new RootModel.Weights(
              RootModel.Weights.VERSION,
              variant,
              config,
              Arrays.stream(columns).mapToObj(i -> LedgerColumn.values()[i]).toList(),
              first,
              cohort,
              solution.ridge(),
              Arrays.stream(solution.theta()).boxed().toList());
    } else if (offset <= config.trainingCohorts() + config.curveCohorts()) {
      List<Candidate> next = new ArrayList<>();
      for (var ctx : config.contexts().stream().sorted().toList())
        for (var dir : config.directions().stream().sorted().toList()) {
          List<RootRates.Fold> folds = new ArrayList<>();
          if (!curves.isEmpty()) folds.addAll(curves.get(next.size()).rates.folds());
          folds.add(RootRates.fit(cohort, moments, variant.seats(), ctx, dir));
          next.add(new Candidate(new RootRates(ctx, dir, folds), 0, null));
        }
      curves = next;
    } else {
      var next = new ArrayList<Candidate>();
      for (int i = 0; i < curves.size(); i++) {
        Candidate c = curves.get(i);
        var score = holdout.scores(i);
        double loss =
            c.selectionLoss()
                + score.rootMinusRaw().paired().mean() * score.rootMinusRaw().paired().rows();
        next.add(
            new Candidate(c.rates(), loss, c.scores() == null ? score : c.scores().plus(score)));
      }
      curves = next;
      if (offset == config.trainingCohorts() + config.curveCohorts() + config.selectionCohorts()) {
        active = choose(weights, curves, cohort);
        if (active.choice() != RootModel.Choice.FITTED)
          LOG.warn(
              "t={} root candidate has zero correction for the next cycle: {}",
              cohort,
              active.choice());
        nextFirst = cohort + 1;
        weights = null;
        curves = List.of();
      }
    }
    return new State(
        STATE_VERSION,
        cohort,
        nextFirst,
        nextFirst != first || before == null ? 0 : before.skipped(),
        variant,
        meaning,
        config,
        signed,
        packed,
        weights,
        curves,
        active);
  }

  static RootModel choose(RootModel.Weights weights, List<Candidate> curves, int cohort) {
    Candidate winner =
        curves.stream().min(Comparator.comparingDouble(Candidate::selectionLoss)).orElseThrow();
    RootRates rates = winner.rates();
    RootModel.Choice choice =
        weights.ridge() == null
            ? RootModel.Choice.INFINITE_RIDGE
            : winner.selectionLoss() < 0 ? RootModel.Choice.FITTED : RootModel.Choice.ZERO_RATE;
    if (choice != RootModel.Choice.FITTED) {
      rates =
          new RootRates(
              rates.context(),
              rates.direction(),
              rates.folds().stream().map(f -> new RootRates.Fold(f.cohort(), Map.of())).toList());
    }
    var scores = winner.scores();
    if (scores != null && choice != RootModel.Choice.FITTED) scores = scores.zeroRate();
    RootHoldout.Selection guard =
        scores == null || scores.cohorts() != weights.settings().selectionCohorts()
            ? null
            : new RootHoldout.Selection(scores.best(weights.ridge() != null), scores);
    return new RootModel(weights, rates, cohort, choice, guard);
  }
}
