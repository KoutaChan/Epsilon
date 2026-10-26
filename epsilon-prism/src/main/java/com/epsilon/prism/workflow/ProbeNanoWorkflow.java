package com.epsilon.prism.workflow;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.prism.collect.CandidateProbe;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.collect.FixedGrpPhi;
import com.epsilon.prism.collect.HandValue;
import com.epsilon.prism.collect.Sampling;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.probe.NanoProbe;
import com.epsilon.prism.probe.ProbeReport;
import com.epsilon.prism.rows.PrismRow;
import com.epsilon.prism.rows.RowStore;
import com.epsilon.prism.rows.RowWriter;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.spi.PolicyExecutionContext;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * PRISM がどこで Nano に負けるか（probe-nano）。版 1 つを全 4 席にした PRISM の自己対局（収集と同じ b = 貪欲 + 床 ε_b）を 1 コホート打ち、ラベル行に
 * 予約した判断（率 ρ_c）で Nano の貪欲の手を同じ公開観測から求める（{@link NanoProbe}）。打った手と違えば（双子は同じ）、打った手と Nano の手の 2 手を
 * 同じ山の分岐で局末まで測る（続きは PRISM、局末の値は固定の GRP の Φ。{@link CandidateProbe}）。Nano は測定器で、打つことはない。
 *
 * <p>率: --label-rate があれば全ての類のラベル率をその値に、なければ設定の ρ_c。ロールインと写しの率は 0（根も写しも作らない）、ε_C は 0（候補集合を
 * 作らない）、保留は 0。ε_b は設定のまま。
 *
 * <p>牌山と鍵: run の種は設定の seed（Pipeline と同じ）、半荘 ID は {@link #GAME_BASE} + 番号、コホートの種は {@link
 * PrismConfig#cohortSeed}(seed, {@link #PROBE_COHORT})。train のコホート（ID (t−1)·2^20 + 番号、種 cohortSeed(seed, t)）とも評価（ID 2^40 +
 * 番号）とも重ならず、同じ seed と半荘の数なら同じ牌山と同じ硬貨になる。
 *
 * <p>出力: DIR/rows（封印した行。不一致のラベル行だけ）、DIR/report.json（{@link ProbeReport.Report} と収集の結果）。
 */
public final class ProbeNanoWorkflow implements CommandWorkflow<ProbeReport.Report> {
  private static final Logger LOG = LoggerFactory.getLogger(ProbeNanoWorkflow.class);
  private static final Gson GSON =
      new GsonBuilder().setPrettyPrinting().serializeSpecialFloatingPointValues().create();

  /** 測定の半荘 ID の基準（3·2^40）。 */
  public static final long GAME_BASE = 3L << 40;

  /** 測定のコホートの番号（"PROB"。train の反復番号には現れない）。 */
  public static final int PROBE_COHORT = 0x50524f42;

  /** Φ のメモの上限（Pipeline と同じ）。 */
  private static final int PHI_MEMO = 1 << 16;

  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.range(
        "probe-nano",
        WorkflowKind.PROBE,
        "probe-nano --snapshot DIR --nano DIR --games N --out DIR [--nano-device cpu|gpu:N]"
            + " [--label-rate R]",
        8,
        12);
  }

  @Override
  public ProbeReport.Report execute(WorkflowArguments arguments) throws Exception {
    WorkflowOptions options = arguments.namedOptions();
    Path snapshot = options.path("snapshot");
    Path nano = options.path("nano");
    String gamesText = options.optionalText("games", null);
    if (gamesText == null) throw new IllegalArgumentException("probe-nano requires --games");
    int games = Integer.parseInt(gamesText);
    Path out = options.path("out");
    String nanoDevice = nanoDevice(options.optionalText("nano-device", "cpu"));
    String labelRate = options.optionalText("label-rate", null);
    options.requireEmpty();
    if (games < 1 || games > PrismConfig.MAX_GAMES_PER_COHORT)
      throw new IllegalArgumentException("--games " + games);
    PrismConfig config = PrismConfig.fromSettings(arguments.settings());
    config.validateTraining();
    Sampling sampling = sampling(config, labelRate == null ? null : Double.parseDouble(labelRate));
    Probed probed;
    long[] floor, agreements, disagreements;
    int nanoPolicies;
    try (var probe = new NanoProbe(nano, nanoDevice)) {
      probed = collect(config, snapshot, games, out, sampling, probe);
      floor = probe.counts(CandidateProbe.Outcome.FLOOR);
      agreements = probe.counts(CandidateProbe.Outcome.AGREED);
      disagreements = probe.counts(CandidateProbe.Outcome.DISAGREED);
      nanoPolicies = probe.opened();
    }
    CollectionDriver.Result collected = probed.collected();
    LOG.info("probe-nano {} (Nano policies opened: {})", collected.summary(), nanoPolicies);
    List<ProbeReport.Disagreement> deltas = probed.deltas();
    double[] labelRates = new double[DecisionClass.COUNT];
    for (DecisionClass c : DecisionClass.values()) labelRates[c.ordinal()] = sampling.labelRate(c);
    ProbeReport.Report report =
        ProbeReport.of(games, labelRates, floor, agreements, disagreements, deltas);

    Map<String, Object> json = new LinkedHashMap<>();
    json.put("snapshot", snapshot.toString());
    json.put("nano", nano.toString());
    json.put("seed", config.seed());
    json.put("gameBase", GAME_BASE);
    json.put("cohortSeed", PrismConfig.cohortSeed(config.seed(), PROBE_COHORT));
    json.put("epsilonB", config.epsilonB());
    json.put("phi", config.phi());
    json.put("report", report);
    json.put("collection", collected);
    Files.writeString(out.resolve("report.json"), GSON.toJson(json), StandardCharsets.UTF_8);
    for (ProbeReport.Line line : report.classes()) LOG.info(text(line));
    LOG.info(text(report.overall()));
    return report;
  }

  /** 測定器つきで 1 コホート打った結果と、不一致のラベル行の ΔY = Y[測定器の手] − Y[played]。 */
  record Probed(CollectionDriver.Result collected, List<ProbeReport.Disagreement> deltas) {}

  /**
   * 版 snapshot を全 4 席にした自己対局を games 局打ち、ラベル行に予約した判断を probe に問う（半荘 ID は {@link #GAME_BASE} + 番号、コホートの種は
   * {@link #PROBE_COHORT}）。行は out/rows に封印し、読み直して ΔY を返す。
   */
  static Probed collect(
      PrismConfig config,
      Path snapshot,
      int games,
      Path out,
      Sampling sampling,
      CandidateProbe probe)
      throws Exception {
    var profile = EpsilonUtilityProfile.valueOf(config.utility());
    int version = Snapshot.params(snapshot).version();
    List<Long> gameIds = new ArrayList<>(games);
    for (int game = 0; game < games; game++) gameIds.add(GAME_BASE + game);
    Path rows = out.resolve("rows");
    if (Files.exists(rows.resolve(RowWriter.MANIFEST)))
      throw new IllegalArgumentException("Rows are already sealed (use a new --out): " + rows);
    if (Files.exists(rows)) Snapshot.deleteTree(rows);
    CollectionDriver.Result collected;
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
        var policy =
            Snapshot.open(
                snapshot,
                PrismConfig.devices(config.collectDevices()),
                execution,
                config.inferBatch(),
                config.multiTransitionBatch(),
                config.priorSettings());
        var writer = new RowWriter(rows, version, config.rowWriters())) {
      if (grp == null)
        LOG.warn(
            "phi = {}: hand ends are valued by the final score ranking (tests only)",
            config.phi());
      collected =
          driver.collect(
              new CollectionDriver.Cohort(
                  version,
                  config.seed(),
                  PrismConfig.cohortSeed(config.seed(), PROBE_COHORT),
                  gameIds,
                  profile,
                  sampling,
                  config.behaviour(),
                  null,
                  null),
              policy,
              phi,
              writer,
              probe);
      writer.seal();
    }
    // 封印した行から ΔY を読む（行は不一致のラベル行だけで、測った手は played と測定器の手の 2 つ）
    List<ProbeReport.Disagreement> deltas = new ArrayList<>();
    try (var store = new RowStore()) {
      store.addCohort(rows);
      for (int row : store.select(version, index -> true)) {
        PrismRow read = store.read(row);
        deltas.add(
            new ProbeReport.Disagreement(read.gameId(), read.decisionClass(), delta(read)));
      }
    }
    return new Probed(collected, deltas);
  }

  /** ΔY = Y[nano] − Y[played]。行の測った手は played とほかの 1 手（Nano の手）だけ。 */
  static double delta(PrismRow row) {
    float[] values = row.values();
    if (row.measuredCount() != 2 || Float.isNaN(values[row.played()]))
      throw new IllegalStateException(
          "Probe rows measure played and Nano's action: game="
              + row.gameId()
              + " decision="
              + row.decisionId());
    for (int a = 0; a < values.length; a++)
      if (a != row.played() && !Float.isNaN(values[a]))
        return (double) values[a] - values[row.played()];
    throw new IllegalStateException("Unreachable");
  }

  /** 測定の率: ラベル率（rate があれば全ての類で rate）のほかは 0。候補の上限は Sampling の検査のため設定のまま（使わない）。 */
  static Sampling sampling(PrismConfig config, Double rate) {
    Sampling base = config.sampling();
    return new Sampling(
        0,
        rate == null ? base.labelRateDiscardEarly() : rate,
        rate == null ? base.labelRateDiscardMid() : rate,
        rate == null ? base.labelRateDiscardLate() : rate,
        rate == null ? base.labelRateCommit() : rate,
        base.maxCandidates(),
        base.measureAllDiscards(),
        0,
        0,
        0,
        0,
        0,
        0);
  }

  /** PRISM の書き方（gpuN）も Nano の書き方（gpu:N）に直す。 */
  private static String nanoDevice(String name) {
    return name.matches("gpu\\d+") ? "gpu:" + name.substring(3) : name;
  }

  private static String text(ProbeReport.Line line) {
    return String.format(
        Locale.ROOT,
        "%-13s rho=%.3f probed=%d agree=%.3f disagree=%d floor=%d dY=%+.5f±%.5f dY>0=%.3f"
            + " perGame=%+.5f±%.5f perSeat=%+.5f±%.5f",
        line.decisionClass(),
        line.labelRate(),
        line.probed(),
        line.agreementRate(),
        line.disagreements(),
        line.floor(),
        line.meanDelta(),
        line.meanDeltaError(),
        line.positiveShare(),
        line.perGame(),
        line.perGameError(),
        line.perSeat(),
        line.perSeatError());
  }

  @Override
  public String summarize(ProbeReport.Report report) {
    return "games=" + report.games() + " " + text(report.overall());
  }
}
