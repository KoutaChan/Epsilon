package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.DuelSizing;
import com.epsilon.agent.eval.Sequential;
import com.epsilon.agent.eval.StopRule;
import com.epsilon.agent.eval.StopSettings;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.prism.eval.EvaluationPlan;
import com.epsilon.prism.settings.PrismSettings;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * 王者との決闘の記録から牌山 1 つあたりの値の標準偏差 σ を求め、王者の卓の牌山の数の上限と、互角とみなす幅を提案する（{@link DuelSizing}）。
 * σ は直近の決闘での σ の中央値にする。設定は変えない。牌山の数の上限は、1 回の決闘の半荘が上限を超えない中で最も多い数にする。判定の
 * 間隔・α・ρ・α 消費関数は、直近の決闘の計画のものをそのまま使う。
 *
 * @param run run のディレクトリ（evaluate の --out も渡せる）
 * @param table 王者の卓の名前（無ければ、王者を相手に逐次の検定で止める全ての卓）
 * @param duels σ を求めるのに使う直近の決闘の数
 * @param games 1 回の決闘の半荘の上限
 */
@Named(value = "duel-sigma", doc = "Estimate the wall σ from the champion's duels and suggest the game cap and parity margin of the champion's tables")
public record DuelSigma(
    @Key(doc = "Run directory (an evaluate --out directory also works)", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Champion's table name (default: every table that stops on a sequential test against the champion)", presence = Presence.ABSENT) String table,
    @Key(doc = "Number of recent duels used to estimate σ", value = "16") @Range(min = 1) int duels,
    @Key(doc = "Game cap of one duel", value = "25000") @Range(min = 1) long games)
    implements PrismCommand {

  /** 提案する幅の刻み。幅はこの刻みに切り上げる。 */
  private static final double MARGIN_STEP = 1e-4;

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    context.requireNoLayers();
    RunDirectory directory = new RunDirectory(run);
    List<JobFile> played = new ArrayList<>();
    for (JobFile job : new Jobs(directory.jobs(), directory.report(RunDirectory.EVALUATIONS)).list().jobs()) {
      if (job.state() != JobFile.State.DONE || (table != null && !job.table().equals(table))) {
        continue;
      }
      TableSpec spec = RecordFile.read(TableSpec.class, job.definition()).value();
      if (TableSpec.kind(spec.opponent()) == TableSpec.Opponent.CHAMPION
          && spec.stop() instanceof StopSettings.Test
          && !plan(job).crowns()) {
        played.add(job);
      }
    }
    if (played.isEmpty()) {
      String where = table == null ? "" : " on " + table;
      throw new SettingsException(List.of("The run " + directory.root() + " has no finished champion duel" + where));
    }
    List<JobFile> recent = played.subList(Math.max(0, played.size() - duels), played.size());
    PrintStream out = context.out();
    double[] sigmas = new double[recent.size()];
    for (int index = 0; index < sigmas.length; index++) {
      JobFile job = recent.get(index);
      DuelResult result = Jobs.result(directory.job(job.name()), DuelResult.class);
      sigmas[index] = DuelSizing.sigma(result);
      out.println(
          String.format(
              Locale.ROOT,
              "%-24s walls=%d games=%d verdict=%s sigma=%.4f",
              job.name(),
              result.walls(),
              result.games(),
              result.verdict(),
              sigmas[index]));
    }
    double sigma = DuelSizing.median(sigmas);
    DuelSetup duel = plan(recent.getLast()).duel();
    if (!(duel.stop() instanceof StopRule.Test test)) {
      throw new IllegalStateException(recent.getLast().name() + " was not planned as a test");
    }
    Sequential sequential = test.sequential();
    int perWall = duel.layout().gamesPerWall();
    double now = DuelSizing.detection(sequential, duel.walls(), sigma, DuelSizing.REACH * test.evenAbove());
    DuelSizing.Choice choice = DuelSizing.choose(sequential, sigma, perWall, games);
    double margin = Math.ceil(choice.margin() / MARGIN_STEP) * MARGIN_STEP;
    double suggested = DuelSizing.detection(sequential, choice.walls(), sigma, DuelSizing.REACH * margin);
    out.println(
        String.format(
            Locale.ROOT,
            "median sigma %.4f over %d duels (%s .. %s)",
            sigma,
            recent.size(),
            recent.getFirst().name(),
            recent.getLast().name()));
    out.println(
        String.format(
            Locale.ROOT,
            "planned: walls=%d games=%d checkEvery=%d alpha=%s spending=%s evenMargin=[%.4f, %.4f]"
                + " finds %.1f x the upper margin with %.4f",
            duel.walls(),
            (long) duel.walls() * perWall,
            sequential.checkEvery(),
            sequential.alpha(),
            sequential.spending(),
            test.evenBelow(),
            test.evenAbove(),
            DuelSizing.REACH,
            now));
    out.println(
        String.format(
            Locale.ROOT,
            "suggested within %d games: walls=%d games=%d evenMargin=%.4f finds %.1f x the margin with %.4f",
            games,
            choice.walls(),
            (long) choice.walls() * perWall,
            margin,
            DuelSizing.REACH,
            suggested));
    return Outcome.ok(
        String.format(
            Locale.ROOT,
            "sigma=%.4f duels=%d walls=%d evenMargin=%.4f",
            sigma,
            recent.size(),
            choice.walls(),
            margin));
  }

  private static EvaluationPlan plan(JobFile job) {
    return RecordFile.read(EvaluationPlan.class, job.plan()).value();
  }
}
