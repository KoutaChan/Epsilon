package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.RunReport;
import com.epsilon.agent.app.RunReport.Cell;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.run.IterationReport;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.StepReport;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.prism.eval.EvaluationJobs;
import com.epsilon.prism.eval.JobComparison;
import com.epsilon.prism.run.CollectReport;
import com.epsilon.prism.run.LearnReport;
import com.epsilon.prism.run.PrismRun;
import com.epsilon.prism.run.PrismStep;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * run の報告。ふだんは、run の反復ごとの 1 行と、卓ごとの最も新しい結果を、表か 1 行の JSON で出す（{@link RunReport}）。run のファイルは
 * 読むだけ。
 *
 * <p>PRISM が足す列は次のとおり。
 *
 * <ul>
 *   <li>反復: {@code rows}（確定したコホートの行の数）、{@code r2}（帳簿のホールドアウトの R²。帳簿を当てはめなかった反復では無い）、
 *       {@code gain}（教師のゲイン K′）。
 *   <li>卓: {@code opponent}（相手）、{@code games}（半荘の数）、{@code meanRank} と {@code meanRankError}（候補の平均順位とその
 *       標準誤差）、{@code delta} と {@code deltaError}（順位の対の量 δ とその標準誤差。δ は負が良い）、{@code verdict}（判定）、
 *       {@code stopped}（止めた理由）。王者の保存先が空で、打たずに候補を最初の王者にしたジョブは、判定だけを {@value #CROWNED} と書く。
 * </ul>
 *
 * <p>{@code --compare} では、同じ卓の 2 つのバージョンのジョブを同じ牌山で対にし、牌山ごとの δ と υ の差（2 つ目 − 1 つ目）と対の標準誤差を
 * 出す（{@link JobComparison}）。1 つ目のジョブは {@code --run} から、2 つ目は {@code --other} があればその run から読む。候補のほかの計画が
 * 違う組は拒む（相手が系譜のバージョンなら、同じ系譜の同じバージョンを相手にした組だけを対にする）。計画を決めた jar が違う組は拒まず、
 * 両方の jar を出す。
 *
 * @param run run のディレクトリ（{@code --compare} では evaluate の --out も読める）
 * @param last 最も新しい反復から数えて出す反復の数（無ければ全て）
 * @param json 表でなく 1 行の JSON で出す
 * @param compare 対にして比べる卓の名前
 * @param versions 比べる 2 つのバージョン（{@code a,b}）
 * @param other 2 つ目のバージョンのジョブを持つ run のディレクトリ
 */
@Named(
    value = "report",
    doc = "Print one line per iteration of the run and the latest result per table (--compare pairs two versions on the same walls)")
public record Report(
    @Key(doc = "Run directory (with --compare, an evaluate --out directory also works)", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Number of iterations to print, counted back from the latest (default: all)", presence = Presence.ABSENT) @Range(min = 1) Integer last,
    @Key(doc = "Print one JSON line instead of a table", value = "false") boolean json,
    @Key(doc = "Table name to compare on the same walls (use with --versions)", presence = Presence.ABSENT) String compare,
    @Key(doc = "Two versions to compare (a,b; the difference is b − a)", presence = Presence.ABSENT) String versions,
    @Key(doc = "Run directory holding the jobs of the second version (default: --run)", presence = Presence.ABSENT) Path other)
    implements PrismCommand {

  /** 打たずに候補を最初の王者にしたジョブの判定。 */
  static final String CROWNED = "CROWNED";

  /** 打たずに候補を最初の王者にしたジョブの結果の行だけが持つキー（{@link EvaluationJobs.Crowned} の成分）。 */
  private static final String CHAMPION = "champion";

  /** PRISM の run の報告の形。 */
  static final RunReport.Layout LAYOUT =
      RunReport.Layout.of(
          PrismRun.LINEAGE,
          new RunReport.Columns<>(List.of("rows", "r2", "gain"), Report::iteration),
          new RunReport.Columns<>(
              List.of("opponent", "games", "meanRank", "meanRankError", "delta", "deltaError", "verdict", "stopped"),
              Report::result));

  public Report {
    if (compare == null && (versions != null || other != null)) {
      throw new InvalidSetting(versions != null ? "versions" : "other", "--versions and --other go with --compare");
    }
    if (compare != null) {
      if (versions == null) {
        throw new InvalidSetting("compare", "--compare needs --versions a,b");
      }
      if (last != null || json) {
        throw new InvalidSetting(last != null ? "last" : "json", "--last and --json are not for --compare");
      }
      Versions.of(versions);
    }
  }

  /**
   * 比べる 2 つのバージョン。
   *
   * @param first 1 つ目のジョブのバージョン（{@code --run} の run）
   * @param second 2 つ目のジョブのバージョン（{@code --other} があればその run）
   */
  record Versions(int first, int second) {

    private static final Pattern FORM = Pattern.compile("([0-9]+),([0-9]+)");

    /**
     * @param text {@code a,b}
     * @throws InvalidSetting 書き方が違うか、どちらかが 1 より小さいとき
     */
    static Versions of(String text) {
      Matcher matcher = FORM.matcher(text);
      if (matcher.matches()) {
        try {
          int first = Integer.parseInt(matcher.group(1));
          int second = Integer.parseInt(matcher.group(2));
          if (first >= 1 && second >= 1) {
            return new Versions(first, second);
          }
        } catch (NumberFormatException overflow) {
          // 下で同じ誤りにする
        }
      }
      throw new InvalidSetting("versions", "expected two versions a,b with a, b >= 1, not " + text);
    }
  }

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    context.requireNoLayers();
    if (compare != null) {
      return compareJobs(context.out());
    }
    RunReport report = RunReport.read(new RunDirectory(run), LAYOUT, last);
    context.out().print(json ? Json.write(report.json()) + "\n" : report.text());
    return Outcome.ok(
        "iterations=" + report.closed() + " shown=" + report.iterations().size() + " tables=" + report.tables().size());
  }

  /** 2 つのジョブを同じ牌山で対にした差を出す。 */
  private Outcome compareJobs(PrintStream out) throws IOException {
    Versions pair = Versions.of(versions);
    JobComparison comparison =
        JobComparison.of(
            compare,
            new RunDirectory(run),
            pair.first(),
            new RunDirectory(other != null ? other : run),
            pair.second());
    JobComparison.Side first = comparison.first();
    JobComparison.Side second = comparison.second();
    DuelSetup duel = first.plan().duel();
    out.println(
        String.format(
            Locale.ROOT,
            "%s: %s against %s on %d paired walls (%d games); opponent %s, wall seed %d from wall %d, %s %s, %s",
            compare,
            second.plan().candidate().label(),
            first.plan().candidate().label(),
            comparison.walls(),
            (long) comparison.walls() * duel.layout().gamesPerWall(),
            duel.opponent().label(),
            duel.wallSeed(),
            duel.wallBase(),
            duel.seating(),
            duel.splits(),
            duel.draws()));
    side(out, "first", first, comparison.walls());
    side(out, "second", second, comparison.walls());
    difference(out, "rank", comparison.rank(), "negative");
    difference(out, "utility", comparison.utility(), "positive");
    return Outcome.ok(
        String.format(
            Locale.ROOT,
            "%s second - first: rank %.4f +- %.4f, utility %.4f +- %.4f on %d walls",
            compare,
            comparison.rank().mean(),
            comparison.rank().error(),
            comparison.utility().mean(),
            comparison.utility().error(),
            comparison.walls()));
  }

  /** 1 つのジョブの行: 状態、済んだ牌山と対にしなかった牌山の数、候補、jar、対にした牌山の δ と υ。 */
  private static void side(PrintStream out, String which, JobComparison.Side side, int paired) {
    out.println(
        String.format(
            Locale.ROOT,
            "%-6s %s in %s: %s, %d walls (%d unpaired), candidate %s, jar %s, rank %.6f +- %.6f, utility %.6f +- %.6f",
            which,
            side.job().name(),
            side.run().root(),
            side.job().state(),
            side.job().walls(),
            side.job().walls() - paired,
            side.plan().candidate().label(),
            side.plan().jar(),
            side.rank().mean(),
            side.rank().error(),
            side.utility().mean(),
            side.utility().error()));
  }

  /** 差の行。z は差を対の標準誤差で割った値（標準誤差が 0 なら出さない）。 */
  private static void difference(PrintStream out, String score, JobComparison.Value value, String better) {
    String z = value.error() > 0 ? String.format(Locale.ROOT, "%.2f", value.mean() / value.error()) : "-";
    out.println(
        String.format(
            Locale.ROOT,
            "%-7s second - first %.6f, paired error %.6f, z %s (%s is better for the second)",
            score,
            value.mean(),
            value.error(),
            z,
            better));
  }

  /** 確定したコホートの行の数、帳簿の R²、教師のゲイン。 */
  private static List<Cell> iteration(IterationReport report) {
    CollectReport collect = metrics(report, PrismStep.COLLECT, CollectReport.class);
    LearnReport learn = metrics(report, PrismStep.LEARN, LearnReport.class);
    return List.of(
        collect == null ? Cell.NONE : Cell.integer(collect.rows()),
        learn == null || learn.ledger() == null ? Cell.NONE : Cell.decimal(learn.ledger().r2(), 4),
        learn == null ? Cell.NONE : Cell.decimal(learn.tempo().gain(), 4));
  }

  /** 段の報告の中身。段が報告に無いか、中身の無い段なら null。 */
  private static <M extends Record> M metrics(IterationReport report, PrismStep step, Class<M> type) {
    StepReport reported = report.steps().get(step.name());
    return reported == null || reported.metrics() == null ? null : type.cast(reported.metrics());
  }

  /** ジョブの結果の行の列。 */
  private static List<Cell> result(Node.Table line) {
    if (line.get(CHAMPION) != null) {
      // ほかのフィールドもある行は、読めない行として誤りにする
      Jobs.result(line, EvaluationJobs.Crowned.class);
      return List.of(
          Cell.NONE, Cell.NONE, Cell.NONE, Cell.NONE, Cell.NONE, Cell.NONE, Cell.text(CROWNED), Cell.NONE);
    }
    DuelResult duel = Jobs.result(line, DuelResult.class);
    DuelResult.SlotValue candidate = duel.slots().get(0);
    return List.of(
        Cell.text(duel.slots().get(1).label()),
        Cell.integer(duel.games()),
        Cell.decimal(candidate.meanRank(), 4),
        Cell.decimal(candidate.rankError(), 4),
        Cell.decimal(duel.rank().mean(), 4),
        Cell.decimal(duel.rank().error(), 4),
        Cell.text(duel.verdict().name()),
        Cell.text(duel.stopped().name()));
  }
}
