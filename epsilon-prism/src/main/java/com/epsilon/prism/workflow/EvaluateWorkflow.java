package com.epsilon.prism.workflow;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.eval.Arena;
import com.epsilon.prism.eval.EnsembleService;
import com.epsilon.prism.eval.HybridSeat;
import com.epsilon.prism.eval.NanoService;
import com.epsilon.prism.eval.PlayStats;
import com.epsilon.prism.policy.PlayerPolicy;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.spi.PolicyExecutionContext;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import com.google.gson.Gson;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 候補の打ち手を 1 席、相手の打ち手を 3 席に置いて打たせる（{@link Arena}。候補の席は半荘 ID で回す。全員貪欲）。牌山の種と基準の半荘 ID は
 * train の評価と同じなので、同じ seed と半荘の数なら、候補や相手を替えても同じ牌山の対比較になる。対局の設定（収集の device・ワーカー・バッチ・
 * 網の推論の道・効用・seed）は [epsilon.prism]。
 *
 * <p>打ち手の書き方（--snapshot と --opponent）: 版のディレクトリ（PRISM の打つ網。{@link Snapshot#open}）、{@code heuristic}（規則の打ち手、
 * 相手だけ）、{@code nano:DIR}（Epsilon-Nano の Decision チェックポイント。測定専用で、デバイスは --nano-device）。Nano の卓は
 * {@code --snapshot 版 --opponent nano:DIR}（PRISM 1 席 対 Nano 3 席）と {@code --snapshot nano:DIR --opponent 版}（Nano 1 席 対
 * PRISM 3 席）。
 *
 * <p>混成の席（測定専用）: {@code --hybrid 類[+類...] --hybrid-nano DIR [--hybrid-open true]} で、候補の版の席のうち類の判断だけ Nano に打たせる
 * （{@link HybridSeat}。NONE なら全て版）。--hybrid-other DIR は Nano の代わりに別の版に任せる。--hybrid-open は副露がある間の判断も Nano にする。
 * --hybrid-closed true は類の判断のうち門前の間だけ、--hybrid-fraction F は類の判断のうち硬貨が F 未満のものだけを任せる。
 * --per-game FILE は半荘ごとの候補の結果（半荘 ID・着順・効用）を書く（同じ牌山の対比較のため）。
 *
 * <p>合議の席（測定専用）: {@code --ensemble DIR [--ensemble-weight W]} で、候補の版と DIR の版の s を (1 − W)·s + W·s_DIR で混ぜた貪欲で打つ
 * （{@link EnsembleService}。2 つの s は同じ単位であること）。DIR と W を読点で区切って並べると K 本の合議（s = (1 − ΣW)·s + Σ W_i·s_i）。
 */
public final class EvaluateWorkflow implements CommandWorkflow<Arena.Result> {
  private static final Logger LOG = LoggerFactory.getLogger(EvaluateWorkflow.class);

  /** Nano の打ち手の接頭辞。 */
  static final String NANO = "nano:";

  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.range(
        "evaluate",
        WorkflowKind.EVALUATION,
        "evaluate --snapshot DIR|nano:DIR [--games N] [--opponent heuristic|DIR|nano:DIR]"
            + " [--nano-device cpu|gpu:N] [--hybrid CLASSES --hybrid-nano DIR|--hybrid-other DIR"
            + " [--hybrid-open true] [--hybrid-closed true] [--hybrid-fraction F]] [--per-game FILE] [--ensemble DIR[,DIR...] --ensemble-weight W[,W...]]",
        2,
        22);
  }

  /** --hybrid の類（+ 区切りの {@link DecisionClass} の名前。NONE は空）。 */
  static Set<DecisionClass> classes(String text) {
    Set<DecisionClass> classes = EnumSet.noneOf(DecisionClass.class);
    if (text.equals("NONE")) return classes;
    for (String name : text.split("\\+")) classes.add(DecisionClass.valueOf(name));
    return classes;
  }

  @Override
  public Arena.Result execute(WorkflowArguments arguments) throws Exception {
    WorkflowOptions options = arguments.namedOptions();
    String candidateName = options.optionalText("snapshot", null);
    if (candidateName == null) throw new IllegalArgumentException("evaluate requires --snapshot");
    PrismConfig config = PrismConfig.fromSettings(arguments.settings());
    int games =
        Integer.parseInt(options.optionalText("games", Integer.toString(config.evalGames())));
    String opponentName = options.optionalText("opponent", "heuristic");
    String nanoDevice = options.optionalText("nano-device", "gpu:0");
    String hybridText = options.optionalText("hybrid", null);
    String hybridNano = options.optionalText("hybrid-nano", null);
    String hybridOther = options.optionalText("hybrid-other", null);
    boolean hybridOpen = Boolean.parseBoolean(options.optionalText("hybrid-open", "false"));
    boolean hybridClosed = Boolean.parseBoolean(options.optionalText("hybrid-closed", "false"));
    double hybridFraction = Double.parseDouble(options.optionalText("hybrid-fraction", "1"));
    String perGame = options.optionalText("per-game", null);
    String ensembleText = options.optionalText("ensemble", null);
    String ensembleWeights = options.optionalText("ensemble-weight", "0.5");
    options.requireEmpty();
    if (candidateName.equals("heuristic"))
      throw new IllegalArgumentException("The candidate must be a version or nano:DIR");
    if (hybridText != null
        && ((hybridNano == null) == (hybridOther == null) || candidateName.startsWith(NANO)))
      throw new IllegalArgumentException(
          "--hybrid needs a version as the candidate and one of --hybrid-nano DIR or --hybrid-other DIR");
    if (ensembleText != null
        && (hybridText != null || candidateName.startsWith(NANO) || ensembleText.contains(NANO)))
      throw new IllegalArgumentException("--ensemble needs versions and no --hybrid");
    String[] ensembleNames = ensembleText == null ? new String[0] : ensembleText.split(",");
    double[] weights = ensembleWeights(ensembleWeights, ensembleNames.length);
    Set<DecisionClass> hybridClasses = hybridText == null ? null : classes(hybridText);
    config.validate();
    try (var execution =
            new PolicyExecutionContext(config.encoders(), config.slots(), config.readyBatches());
        var driver =
            new CollectionDriver(
                new CollectionDriver.Settings(
                    config.maxActiveGames(),
                    config.advanceWorkers(),
                    new InferenceBatchingSettings(config.batchWaitMicros())));
        var base = open(candidateName, config, execution, nanoDevice);
        var opponent = open(opponentName, config, execution, nanoDevice);
        var other =
            hybridText == null
                ? null
                : hybridNano != null
                    ? NanoService.open(Path.of(hybridNano), nanoDevice, execution)
                    : open(hybridOther, config, execution, nanoDevice);
        var partners = new Closer()) {
      HybridSeat hybrid = other == null ? null : new HybridSeat(base, other, hybridClasses, hybridOpen, hybridClosed, hybridFraction);
      PolicyService candidate = hybrid == null ? base : hybrid;
      // K 本の合議は入れ子にする。W_i が最終の重みになるよう、後ろから a_i = W_i / (1 − Σ_{j>i} W_j) で混ぜる
      double later = 0;
      double[] steps = new double[weights.length];
      for (int i = weights.length - 1; i >= 0; i--) {
        steps[i] = weights[i] / (1 - later);
        later += weights[i];
      }
      for (int i = 0; i < ensembleNames.length; i++)
        candidate =
            new EnsembleService(
                candidate, partners.add(open(ensembleNames[i], config, execution, nanoDevice)), steps[i]);
      // 局の結果と判断の傾向の内訳（候補の席と相手の席）。打ち筋の違いを探す診断として記録する
      var stats = new PlayStats();
      Arena.Detailed detailed =
          Arena.detailed(
              driver,
              candidate,
              opponent,
              label(opponentName),
              games,
              config.seed() ^ Arena.EVALUATION_SALT,
              Arena.HEURISTIC_BASE,
              EpsilonUtilityProfile.valueOf(config.utility()),
              stats);
      LOG.info("evaluate play {}", new Gson().toJson(stats.report()));
      if (hybrid != null) {
        long[] routed = hybrid.routed();
        LOG.info(
            "evaluate hybrid classes={} open={} versionDecisions={} nanoDecisions={}",
            hybridText,
            hybridOpen,
            routed[0],
            routed[1]);
      }
      if (perGame != null) {
        StringBuilder text = new StringBuilder();
        for (Arena.Game game : detailed.games())
          text.append(game.gameId()).append('\t').append(game.rank()).append('\t').append(game.utility()).append('\n');
        Files.writeString(Path.of(perGame), text, StandardCharsets.UTF_8);
      }
      return detailed.result();
    }
  }

  /** --ensemble-weight の重み（合議の版の数だけ。各 [0, 1] で、2 本以上なら和 < 1）。 */
  static double[] ensembleWeights(String text, int count) {
    if (count == 0) return new double[0];
    String[] parts = text.split(",");
    if (parts.length != count)
      throw new IllegalArgumentException("--ensemble-weight needs one weight per --ensemble version");
    double[] weights = new double[count];
    double total = 0;
    for (int i = 0; i < count; i++) {
      weights[i] = Double.parseDouble(parts[i]);
      if (!(weights[i] >= 0 && weights[i] <= 1)) throw new IllegalArgumentException("Bad weight " + parts[i]);
      total += weights[i];
    }
    if (!(total < 1 || count == 1))
      throw new IllegalArgumentException("Ensemble weights must sum to less than 1: " + text);
    return weights;
  }

  /** 合議の版をまとめて閉じる。 */
  private static final class Closer implements AutoCloseable {
    private final List<PolicyService> opened = new ArrayList<>();

    PolicyService add(PolicyService policy) {
      opened.add(policy);
      return policy;
    }

    @Override
    public void close() throws Exception {
      Exception failure = null;
      for (PolicyService policy : opened) {
        try {
          policy.close();
        } catch (Exception e) {
          if (failure == null) failure = e;
          else failure.addSuppressed(e);
        }
      }
      if (failure != null) throw failure;
    }
  }

  /** 打ち手の書き方から方策を開く（heuristic・nano:DIR・版のディレクトリ）。 */
  private static PolicyService open(
      String name, PrismConfig config, PolicyExecutionContext execution, String nanoDevice)
      throws Exception {
    if (name.equals("heuristic")) return PlayerPolicy.heuristic();
    if (name.startsWith(NANO))
      return NanoService.open(Path.of(name.substring(NANO.length())), nanoDevice, execution);
    return Snapshot.open(
        Path.of(name),
        PrismConfig.devices(config.collectDevices()),
        execution,
        config.inferBatch(),
        config.priorSettings(),
        config.inference());
  }

  /** 結果の相手の名前（heuristic、nano:名前、snapshot:名前）。 */
  static String label(String name) {
    if (name.equals("heuristic")) return name;
    if (name.startsWith(NANO)) return NANO + Path.of(name.substring(NANO.length())).getFileName();
    return "snapshot:" + Path.of(name).getFileName();
  }

  @Override
  public String summarize(Arena.Result result) {
    return String.format(
        "opponent=%s games=%d rank=%.3f±%.3f utility=%.3f±%.3f shares=%.3f/%.3f/%.3f/%.3f",
        result.opponent(),
        result.games(),
        result.meanRank(),
        result.rankError(),
        result.meanUtility(),
        result.utilityError(),
        result.rankShares()[0],
        result.rankShares()[1],
        result.rankShares()[2],
        result.rankShares()[3]);
  }
}
