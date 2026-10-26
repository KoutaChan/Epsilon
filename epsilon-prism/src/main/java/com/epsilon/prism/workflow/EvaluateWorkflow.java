package com.epsilon.prism.workflow;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.eval.Arena;
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
import java.nio.file.Path;

/**
 * 版を 3×Heuristic か 3×別の版と打たせる（{@link Arena}。候補の席は半荘 ID で回す。どちらも貪欲）。牌山の種と基準の半荘 ID は train の評価と同じ
 * なので、同じ seed と半荘の数なら run の評価の記録と同じ牌山の対比較になる。対局の設定（収集の device・ワーカー・バッチ・効用・seed）は
 * [epsilon.prism]。
 */
public final class EvaluateWorkflow implements CommandWorkflow<Arena.Result> {
  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.range(
        "evaluate",
        WorkflowKind.EVALUATION,
        "evaluate --snapshot DIR [--games N] [--opponent heuristic|DIR]",
        2,
        6);
  }

  @Override
  public Arena.Result execute(WorkflowArguments arguments) throws Exception {
    WorkflowOptions options = arguments.namedOptions();
    Path snapshot = options.path("snapshot");
    PrismConfig config = PrismConfig.fromSettings(arguments.settings());
    int games =
        Integer.parseInt(options.optionalText("games", Integer.toString(config.evalGames())));
    String opponentName = options.optionalText("opponent", "heuristic");
    options.requireEmpty();
    // 相手は heuristic か、版のディレクトリ（貪欲で 3 席。牌山は heuristic と同じ基準の半荘 ID）
    boolean snapshotOpponent = !opponentName.equals("heuristic");
    config.validate();
    try (var execution =
            new PolicyExecutionContext(config.encoders(), config.slots(), config.readyBatches());
        var driver =
            new CollectionDriver(
                new CollectionDriver.Settings(
                    config.maxActiveGames(),
                    config.advanceWorkers(),
                    new InferenceBatchingSettings(config.batchWaitMicros())));
        var candidate =
            Snapshot.open(
                snapshot,
                PrismConfig.devices(config.collectDevices()),
                execution,
                config.inferBatch(),
                config.multiTransitionBatch(),
                config.priorSettings());
        PolicyService opponent =
            snapshotOpponent
                ? Snapshot.open(
                    Path.of(opponentName),
                    PrismConfig.devices(config.collectDevices()),
                    execution,
                    config.inferBatch(),
                    config.multiTransitionBatch(),
                    config.priorSettings())
                : PlayerPolicy.heuristic(0)) {
      return Arena.duel(
          driver,
          candidate,
          opponent,
          snapshotOpponent ? "snapshot:" + Path.of(opponentName).getFileName() : opponentName,
          games,
          config.seed() ^ Arena.EVALUATION_SALT,
          Arena.HEURISTIC_BASE,
          EpsilonUtilityProfile.valueOf(config.utility()));
    }
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
