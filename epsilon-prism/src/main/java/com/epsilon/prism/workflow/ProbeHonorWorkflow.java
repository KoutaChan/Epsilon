package com.epsilon.prism.workflow;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.probe.HonorCallProbe;
import com.epsilon.prism.probe.ProbeReport;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * PRISM が字牌のポンを正しく選べているか（probe-honor）。種類（{@link HonorCallProbe.Kind}）ごとに、probe-nano と同じ自己対局（同じ半荘と牌山）を
 * 1 コホート打ち、合法手に字牌のポンがある判断で PRISM の手を数える。その種類で PRISM がスルーした判断は、ポンとスルーを同じ山の分岐で局末まで測る
 * （ΔY = Y[ポン] − Y[スルー]。正ならスルーは損）。全ての類のラベル率は 1（字牌のポンでない判断は打った手を返すので行も分岐も作らない）。
 *
 * <p>出力: DIR/&lt;種類&gt;/rows（スルーを測った行）、DIR/report.json（{@link Report}）。
 */
public final class ProbeHonorWorkflow implements CommandWorkflow<ProbeHonorWorkflow.Report> {
  private static final Logger LOG = LoggerFactory.getLogger(ProbeHonorWorkflow.class);
  private static final Gson GSON =
      new GsonBuilder().setPrettyPrinting().serializeSpecialFloatingPointValues().create();

  /**
   * 1 つの種類の結果。
   *
   * @param pon PRISM がポンした判断
   * @param pass PRISM がスルーした判断
   * @param other ほかの手（ロンなど）を打った判断
   * @param call スルーを測った行の集計（{@link ProbeReport.Line}。meanDelta は 1 判断あたりの Y[ポン] − Y[スルー]、perGame は半荘あたりの和）
   */
  public record KindLine(
      String kind, long pon, long pass, long other, double ponRate, ProbeReport.Line call) {}

  public record Report(String snapshot, int games, List<KindLine> kinds) {}

  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.range(
        "probe-honor",
        WorkflowKind.PROBE,
        "probe-honor --snapshot DIR --games N --out DIR [--kinds SEAT,ROUND,...]",
        6,
        8);
  }

  @Override
  public Report execute(WorkflowArguments arguments) throws Exception {
    WorkflowOptions options = arguments.namedOptions();
    Path snapshot = options.path("snapshot");
    String gamesText = options.optionalText("games", null);
    if (gamesText == null) throw new IllegalArgumentException("probe-honor requires --games");
    int games = Integer.parseInt(gamesText);
    Path out = options.path("out");
    String kindsText = options.optionalText("kinds", null);
    options.requireEmpty();
    if (games < 1 || games > PrismConfig.MAX_GAMES_PER_COHORT)
      throw new IllegalArgumentException("--games " + games);
    List<HonorCallProbe.Kind> kinds = new ArrayList<>();
    if (kindsText == null) kinds.addAll(Arrays.asList(HonorCallProbe.Kind.values()));
    else
      for (String name : kindsText.split(","))
        kinds.add(HonorCallProbe.Kind.valueOf(name.strip().toUpperCase(Locale.ROOT)));
    PrismConfig config = PrismConfig.fromSettings(arguments.settings());
    config.validateTraining();
    var sampling = ProbeNanoWorkflow.sampling(config, 1.0);
    double[] rates = new double[DecisionClass.COUNT];
    Arrays.fill(rates, 1.0);
    long[] zeros = new long[DecisionClass.COUNT];
    List<KindLine> lines = new ArrayList<>();
    for (HonorCallProbe.Kind kind : kinds) {
      var probe = new HonorCallProbe(kind);
      var probed =
          ProbeNanoWorkflow.collect(
              config,
              snapshot,
              games,
              out.resolve(kind.name().toLowerCase(Locale.ROOT)),
              sampling,
              probe);
      long[] measured = new long[DecisionClass.COUNT];
      for (var delta : probed.deltas()) measured[delta.decisionClass().ordinal()]++;
      // 字牌のポンの判断は鳴きの類か、立直の脅威の下なら脅威の類。測った行は字牌のポンだけなので全体で集計する
      ProbeReport.Line call =
          ProbeReport.of(games, rates, zeros, zeros, measured, probed.deltas()).overall();
      long pon = probe.count(kind, HonorCallProbe.Choice.PON);
      long pass = probe.count(kind, HonorCallProbe.Choice.PASS);
      long other = probe.count(kind, HonorCallProbe.Choice.OTHER);
      long total = pon + pass + other;
      var line =
          new KindLine(
              kind.name(), pon, pass, other, total == 0 ? Double.NaN : (double) pon / total, call);
      lines.add(line);
      LOG.info(text(line));
    }
    Report report = new Report(snapshot.toString(), games, lines);
    Files.writeString(out.resolve("report.json"), GSON.toJson(report), StandardCharsets.UTF_8);
    return report;
  }

  private static String text(KindLine line) {
    return String.format(
        Locale.ROOT,
        "%-6s pon=%d pass=%d other=%d ponRate=%.3f measured=%d dY(pon-pass)=%+.5f±%.5f dY>0=%.3f"
            + " perGame=%+.5f±%.5f",
        line.kind(),
        line.pon(),
        line.pass(),
        line.other(),
        line.ponRate(),
        line.call().disagreements(),
        line.call().meanDelta(),
        line.call().meanDeltaError(),
        line.call().positiveShare(),
        line.call().perGame(),
        line.call().perGameError());
  }

  @Override
  public String summarize(Report report) {
    StringBuilder text = new StringBuilder("games=" + report.games());
    for (KindLine line : report.kinds()) text.append('\n').append(text(line));
    return text.toString();
  }
}
