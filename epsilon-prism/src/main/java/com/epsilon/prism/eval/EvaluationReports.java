package com.epsilon.prism.eval;

import com.epsilon.agent.eval.Alarm;
import com.epsilon.agent.eval.AlarmBudget;
import com.epsilon.agent.eval.Alarms;
import com.epsilon.agent.eval.Baseline;
import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.Estimate;
import com.epsilon.agent.eval.EvaluationSettings;
import com.epsilon.agent.eval.Ratings;
import com.epsilon.agent.eval.Score;
import com.epsilon.agent.eval.Signal;
import com.epsilon.agent.eval.Split;
import com.epsilon.agent.eval.StopSettings;
import com.epsilon.agent.eval.StoreEvent;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.eval.WallSet;
import com.epsilon.agent.eval.WallValues;
import com.epsilon.agent.eval.Watch;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.prism.core.TwinDiscard;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 評価の報告（run の reports にある baseline.tsv・ratings.tsv・alarms.tsv）。ジョブの結果を反映するたびに、結果のあるジョブから書き直す。
 * 表に入れるのは、結果を反映したジョブの設定の行にある卓のジョブだけ。
 *
 * <ul>
 *   <li>baseline.tsv（系譜の主な数値）: 基準の卓（相手が version:N の卓）の、バージョンごとの δ と標準誤差、候補が 1 席だけの分け方の δ、計画を
 *       決めた jar、同値な打牌の切り方、卓ごとの区切りの番号（前のバージョンと jar か切り方が替わるたびに 1 増える）。相手のバージョンが
 *       いちばん古い基準の卓の尺度に直した値も並べる。ほかの基準の卓は、それより前の卓と両方で測ったバージョンの差を平均し、その平均でつなぐ
 *       （{@link Baseline#link}）。
 *   <li>ratings.tsv: evaluation.anchor があれば、基準の卓とラダー（相手が version:-d で、止め方が判定でない卓）の結果を辺にした尺度
 *       （{@link Ratings}）。θ は、尺度の基準にする卓の相手を 0 にする。
 *   <li>alarms.tsv: ヘッダーには、結果のあるジョブの数から見込んだ、偶然に出るアラートの数（{@link AlarmBudget}）と、実際に出た数を書く。
 *       続けてアラートの行を書く（中身は reports/alarms.jsonl と同じ）。
 * </ul>
 */
final class EvaluationReports {

  /** 系譜の主な数値の表（reports の中のファイルの名前）。 */
  static final String BASELINE = "baseline.tsv";

  /** 推移しない組を見つけるための尺度（reports の中のファイルの名前）。 */
  static final String RATINGS = "ratings.tsv";

  /** アラートの予算と一覧（reports の中のファイルの名前）。 */
  static final String ALARMS = "alarms.tsv";

  private static final Logger LOG = LoggerFactory.getLogger(EvaluationReports.class);

  private final RunDirectory run;
  private final boolean lineage;

  /** 尺度の辺の、牌山ごとの値。キーはジョブと判定に使う量。済んだジョブの games.tsv は変わらないので、JVM が動いている間は覚えておく。 */
  private final Map<String, double[]> edgeValues = new HashMap<>();

  /**
   * @param run ジョブと報告のディレクトリ
   * @param lineage 系譜の run なら true。このとき、予定のラダーを基準の卓の結果で判定する回も、ラダーの回数に数える
   */
  EvaluationReports(RunDirectory run, boolean lineage) {
    this.run = run;
    this.lineage = lineage;
  }

  /**
   * 報告を書き直す。基準の卓の表は基準の卓のジョブの後に、尺度は基準の卓かラダーのジョブの後に（evaluation.anchor があるときだけ）、
   * アラートの表はどのジョブの後にも書き直す。
   *
   * @param evaluation 結果を反映したジョブの設定の行にある、評価のセクション
   * @param kind 結果を反映したジョブの卓の、相手の種類
   */
  void write(EvaluationSettings evaluation, TableSpec.Opponent kind) throws IOException {
    List<RunDirectory.JobName> done = done(evaluation);
    if (kind == TableSpec.Opponent.VERSION) {
      baseline(evaluation, done);
    }
    if (evaluation.anchor() != null && (kind == TableSpec.Opponent.VERSION || kind == TableSpec.Opponent.DISTANCE)) {
      ratings(evaluation, done);
    }
    alarms(evaluation, done);
  }

  /** 結果のあるジョブのうち、設定の行に卓があるものを返す。並びはバージョンの順で、同じバージョンでは卓の名前の順。 */
  private List<RunDirectory.JobName> done(EvaluationSettings evaluation) throws IOException {
    if (!Files.isDirectory(run.jobs())) {
      return List.of();
    }
    List<Path> homes;
    try (Stream<Path> listed = Files.list(run.jobs())) {
      homes = listed.toList();
    }
    List<RunDirectory.JobName> done = new ArrayList<>();
    for (Path home : homes) {
      Optional<RunDirectory.JobName> name = RunDirectory.parseJobName(home.getFileName().toString());
      if (name.isPresent()
          && evaluation.tables().containsKey(name.get().table())
          && Files.exists(home.resolve(Jobs.RESULT))) {
        done.add(name.get());
      }
    }
    done.sort(Comparator.comparingInt(RunDirectory.JobName::version).thenComparing(RunDirectory.JobName::table));
    return done;
  }

  /** 基準の卓のジョブ 1 つ分の行。 @param order 基準の卓の番号（尺度に直す順） */
  private record BaselineRow(int version, int order, String text) {}

  private void baseline(EvaluationSettings evaluation, List<RunDirectory.JobName> done) throws IOException {
    List<String> tables =
        evaluation.tables().entrySet().stream()
            .filter(entry -> TableSpec.kind(entry.getValue().opponent()) == TableSpec.Opponent.VERSION)
            .sorted(
                Comparator.comparingInt((Map.Entry<String, TableSpec> entry) -> base(entry.getValue()))
                    .thenComparing(Map.Entry::getKey))
            .map(Map.Entry::getKey)
            .toList();
    List<BaselineRow> rows = new ArrayList<>();
    List<Baseline.Point> scale = new ArrayList<>();
    Set<Long> scaled = new HashSet<>();
    for (int order = 0; order < tables.size(); order++) {
      String table = tables.get(order);
      List<Measured> measured = new ArrayList<>();
      List<Baseline.Point> points = new ArrayList<>();
      for (RunDirectory.JobName name : done) {
        if (name.table().equals(table)) {
          Measured job = measured(name);
          measured.add(job);
          points.add(new Baseline.Point(name.version(), job.result().rank().mean(), job.result().rank().error()));
        }
      }
      Baseline.Link link =
          order > 0 && points.stream().anyMatch(point -> scaled.contains(point.version()))
              ? Baseline.link(scale, points)
              : null;
      String jar = null;
      TwinDiscard twinDiscard = null;
      int segment = -1;
      List<Baseline.Point> moved = new ArrayList<>();
      for (int index = 0; index < measured.size(); index++) {
        Measured job = measured.get(index);
        if (!job.plan().jar().equals(jar) || job.plan().twinDiscard() != twinDiscard) {
          jar = job.plan().jar();
          twinDiscard = job.plan().twinDiscard();
          segment++;
        }
        Baseline.Point point = order == 0 ? points.get(index) : link != null ? link.apply(points.get(index)) : null;
        if (point != null) {
          moved.add(point);
        }
        rows.add(new BaselineRow(job.version(), order, baselineLine(table, job, segment, point)));
      }
      for (Baseline.Point point : moved) {
        scale.add(point);
        scaled.add(point.version());
      }
    }
    rows.sort(Comparator.comparingInt(BaselineRow::version).thenComparingInt(BaselineRow::order));
    StringBuilder text =
        new StringBuilder(
            "version\ttable\topponent\trank\terror\toneVsRest\tjar\ttwinDiscard\tsegment\tscaled\tscaledError\n");
    for (BaselineRow row : rows) {
      text.append(row.text()).append('\n');
    }
    DurableFiles.write(run.report(BASELINE), text.toString());
  }

  /**
   * baseline.tsv の 1 行を作る。
   *
   * @param point 1 つ目の基準の卓の尺度に直した値（つなげなければ null）
   */
  private static String baselineLine(String table, Measured job, int segment, Baseline.Point point) {
    DuelSetup duel = job.plan().duel();
    String oneVsRest = Split.oneAgainstRest(duel.variant()).toString();
    String split = "-";
    for (DuelResult.SplitValue value : job.result().splits()) {
      if (value.split().equals(oneVsRest)) {
        split = number(value.rank().mean());
      }
    }
    Estimate rank = job.result().rank();
    return String.join(
        "\t",
        Integer.toString(job.version()),
        table,
        duel.opponent().label(),
        number(rank.mean()),
        number(rank.error()),
        split,
        job.plan().jar(),
        job.plan().twinDiscard().name(),
        Integer.toString(segment),
        point == null ? "-" : number(point.value()),
        point == null ? "-" : number(point.error()));
  }

  /**
   * 尺度の辺 1 本。
   *
   * @param table 卓の名前
   * @param jar 計画を決めた jar の ID
   */
  private record Rated(Ratings.Edge edge, String table, String jar) {}

  /**
   * 尺度を当てはめ直す。辺にするのは、基準の卓とラダーの結果のうち、辺をたどって尺度の基準にする卓の相手につながるものだけ。全ての牌山で
   * 値が同じ辺は、分散が 0 で重みが決まらないので入れない。入れなかった辺の数はヘッダーに書く。
   */
  private void ratings(EvaluationSettings evaluation, List<RunDirectory.JobName> done) throws IOException {
    TableSpec anchorTable = evaluation.tables().get(evaluation.anchor());
    String anchor = EvaluationJobs.label(base(anchorTable));
    Score score = anchorTable.score();
    List<Rated> all = new ArrayList<>();
    int constant = 0;
    List<Rated> rated;
    Ratings fitted;
    try {
      for (RunDirectory.JobName name : done) {
        if (!onScale(evaluation.tables().get(name.table()))) {
          continue;
        }
        String job = RunDirectory.jobName(name.version(), name.table());
        EvaluationPlan plan = plan(job);
        DuelSetup duel = plan.duel();
        double[] values = edge(job, duel, score);
        if (Arrays.stream(values).allMatch(value -> value == values[0])) {
          constant++;
          continue;
        }
        Ratings.Edge edge =
            new Ratings.Edge(plan.candidate().label(), duel.opponent().label(), duel.wallSeed(), values);
        all.add(new Rated(edge, name.table(), plan.jar()));
      }
      rated = connected(all, anchor);
      if (rated.isEmpty()) {
        return;
      }
      fitted = Ratings.fit(rated.stream().map(Rated::edge).toList(), anchor);
    } catch (IllegalArgumentException unfit) {
      LOG.warn("{} is not refit: {}", run.report(RATINGS), unfit.getMessage());
      return;
    }
    SortedSet<String> tables = new TreeSet<>();
    SortedSet<String> jars = new TreeSet<>();
    for (Rated edge : rated) {
      tables.add(edge.table());
      jars.add(edge.jar());
    }
    StringBuilder text = new StringBuilder();
    text.append(
        String.format(
            Locale.ROOT,
            "# Ratings against %s, the opponent of the anchor table %s, from %d edges of %s planned by jars %s"
                + " (left out: %d edges not connected to it yet, %d edges with the same value on every wall);"
                + " score %s, a %s theta is stronger\n",
            anchor,
            evaluation.anchor(),
            rated.size(),
            tables,
            jars,
            all.size() - rated.size(),
            constant,
            score,
            score == Score.RANK ? "smaller" : "larger"));
    text.append(String.format(Locale.ROOT, "# chi-square per degree of freedom %.4f\n", fitted.chiSquarePerDegree()));
    List<String> outliers = new ArrayList<>();
    for (Ratings.Residual residual : fitted.outliers()) {
      outliers.add(String.format(Locale.ROOT, "%s-%s z=%.4f", residual.a(), residual.b(), residual.z()));
    }
    text.append(
        String.format(
            Locale.ROOT,
            "# outlier line |z| > %.4f: %s\n",
            fitted.outlierLine(),
            outliers.isEmpty() ? "none" : String.join(", ", outliers)));
    text.append(
        String.format(
            Locale.ROOT,
            "# |z| > 3: %d edges, %.4f expected by chance\n",
            fitted.beyondThree(),
            fitted.beyondThreeExpected()));
    text.append("label\ttheta\terror\tedges\n");
    List<Ratings.Rating> ratings = new ArrayList<>(fitted.ratings());
    ratings.sort(Comparator.comparing(Ratings.Rating::label));
    for (Ratings.Rating rating : ratings) {
      text.append(
          String.join(
              "\t", rating.label(), number(rating.value()), number(rating.error()), Integer.toString(rating.edges())));
      text.append('\n');
    }
    DurableFiles.write(run.report(RATINGS), text.toString());
  }

  /** 辺をたどって {@code anchor} の点につながる辺を、渡した順のまま返す。 */
  private static List<Rated> connected(List<Rated> rated, String anchor) {
    Set<String> reached = new HashSet<>();
    reached.add(anchor);
    boolean grew = true;
    while (grew) {
      grew = false;
      for (Rated edge : rated) {
        boolean a = reached.contains(edge.edge().a());
        boolean b = reached.contains(edge.edge().b());
        if (a != b) {
          reached.add(a ? edge.edge().b() : edge.edge().a());
          grew = true;
        }
      }
    }
    return rated.stream().filter(edge -> reached.contains(edge.edge().a())).toList();
  }

  /** 尺度の辺にする卓（基準の卓と、止め方が判定でないラダー）なら true。 */
  private static boolean onScale(TableSpec table) {
    TableSpec.Opponent kind = TableSpec.kind(table.opponent());
    return kind == TableSpec.Opponent.VERSION
        || (kind == TableSpec.Opponent.DISTANCE && !(table.stop() instanceof StopSettings.Test));
  }

  /** 尺度の辺の、牌山ごとの値（順位なら δ、効用なら υ）。 */
  private double[] edge(String job, DuelSetup duel, Score score) throws IOException {
    String key = job + "/" + score;
    double[] values = edgeValues.get(key);
    if (values == null) {
      WallValues walls = JobGames.read(run.job(job), duel);
      values = score == Score.RANK ? walls.rank() : walls.utility();
      edgeValues.put(key, values);
    }
    return values;
  }

  private void alarms(EvaluationSettings evaluation, List<RunDirectory.JobName> done) throws IOException {
    List<ChampionStore.Entry> history = ChampionStore.open(run.champion()).history();
    OptionalInt crowned =
        !history.isEmpty() && history.getFirst().event() instanceof StoreEvent.Initial initial
            ? OptionalInt.of(initial.champion().version())
            : OptionalInt.empty();
    long promotions = history.stream().filter(entry -> entry.event() instanceof StoreEvent.Promoted).count();
    int duels = 0;
    int ladders = 0;
    int baselines = 0;
    int anchors = 0;
    double duelAlpha = 0;
    double ladderAlpha = 0;
    for (RunDirectory.JobName name : done) {
      TableSpec table = evaluation.tables().get(name.table());
      switch (TableSpec.kind(table.opponent())) {
        case CHAMPION -> {
          if (table.stop() instanceof StopSettings.Test test && !crowning(name, crowned)) {
            duels++;
            duelAlpha = Math.max(duelAlpha, test.alpha());
          }
        }
        case DISTANCE -> {
          if (!(table.stop() instanceof StopSettings.Test)) {
            ladders++;
            ladderAlpha = Math.max(ladderAlpha, alpha(table.stop()));
          }
        }
        case VERSION -> {
          baselines++;
          if (lineage && !evaluation.coveredLadders(name.table(), name.version()).isEmpty()) {
            ladders++;
            ladderAlpha = Math.max(ladderAlpha, alpha(table.stop()));
          }
        }
        case HEURISTIC -> {
          if (table.wallSet() == WallSet.FIXED) {
            anchors++;
          }
        }
        case CHECKPOINT -> {}
      }
    }
    Watch watch = Watch.of(evaluation);
    AlarmBudget budget =
        AlarmBudget.of(
            watch,
            new AlarmBudget.Exposure(
                duels,
                duelAlpha > 0 ? duelAlpha : EvaluationJobs.FIXED_ALPHA,
                ladders,
                ladderAlpha > 0 ? ladderAlpha : EvaluationJobs.FIXED_ALPHA,
                baselines,
                anchors));
    List<Alarm> raised = new Alarms(watch, run).read();
    Map<Signal, Integer> counts = new EnumMap<>(Signal.class);
    for (Alarm line : raised) {
      counts.merge(line.signal(), 1, Integer::sum);
    }
    StringBuilder text = new StringBuilder();
    text.append(
        String.format(
            Locale.ROOT,
            "# Alarms expected by chance if every true difference were 0, over the %d jobs with results"
                + " (z line %.2f, drop margin %.4f, worse streak %d); promotions are expected false promotions and all"
                + " promotions\n",
            done.size(),
            watch.alarmZ(),
            watch.margin(),
            watch.onWorse().streak()));
    text.append("# signal\texpected\tobserved\n");
    text.append(String.format(Locale.ROOT, "# promotions\t%.4f\t%d\n", budget.promotions(), promotions));
    for (Signal signal : Signal.values()) {
      text.append(
          String.format(
              Locale.ROOT, "# %s\t%.4f\t%d\n", signal, budget.alarms().get(signal), counts.getOrDefault(signal, 0)));
    }
    text.append("job\tversion\tsignal\tmean\tlow\thigh\thandling\tchampion\n");
    for (Alarm line : raised) {
      Estimate estimate = line.estimate();
      text.append(
          String.join(
              "\t",
              line.job(),
              Integer.toString(line.version()),
              line.signal().name(),
              number(estimate.mean()),
              number(estimate.low()),
              number(estimate.high()),
              line.handling().name(),
              line.champion() == null ? "-" : Integer.toString(line.champion())));
      text.append('\n');
    }
    DurableFiles.write(run.report(ALARMS), text.toString());
  }

  /** 王者の保存先が空だったので、打たずに候補を初めの王者にしたジョブなら true。計画を読むのは、初めの王者のバージョンのジョブだけ。 */
  private boolean crowning(RunDirectory.JobName name, OptionalInt crowned) throws IOException {
    return crowned.isPresent()
        && crowned.getAsInt() == name.version()
        && plan(RunDirectory.jobName(name.version(), name.table())).crowns();
  }

  /** 基準の卓の相手のバージョン。 */
  private static int base(TableSpec table) {
    return table.opponentVersion(table.from());
  }

  /** 止め方が使う区間の、両側の有意水準。 */
  private static double alpha(StopSettings stop) {
    return switch (stop) {
      case StopSettings.Fixed fixed -> EvaluationJobs.FIXED_ALPHA;
      case StopSettings.Precision precision -> precision.alpha();
      case StopSettings.Test test -> test.alpha();
    };
  }

  /**
   * 済んだ決闘のジョブ。
   *
   * @param plan ジョブの計画
   * @param result 決闘の結果
   */
  private record Measured(int version, EvaluationPlan plan, DuelResult result) {}

  private Measured measured(RunDirectory.JobName name) throws IOException {
    String job = RunDirectory.jobName(name.version(), name.table());
    return new Measured(name.version(), plan(job), Jobs.result(run.job(job), DuelResult.class));
  }

  private EvaluationPlan plan(String job) throws IOException {
    JobFile file = RecordFile.read(JobFile.class, run.job(job).resolve(Jobs.JOB)).value();
    return RecordFile.read(EvaluationPlan.class, file.plan()).value();
  }

  private static String number(double value) {
    return String.format(Locale.ROOT, "%.6f", value);
  }
}
