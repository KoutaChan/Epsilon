package com.epsilon.nano.ai.decision.arena;

import com.epsilon.ai.decision.DecisionSelectionMode;
import com.epsilon.ai.grp.EpsilonGrpInference;
import com.epsilon.config.settings.DecisionTrainArenaSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionTrajectoryPayloadStore;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.nano.config.settings.EpsilonSettings;
import java.util.Locale;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 複数の保存済みモデルを使う自己対局について、入力検証、対局実行、処理速度の計測をまとめる。 */
public final class EpsilonDecisionPopulationCollector {

  private static final Logger log =
      LoggerFactory.getLogger(EpsilonDecisionPopulationCollector.class);

  private EpsilonDecisionPopulationCollector() {}

  /** 選択行動の方策勾配学習方策モデルが全合法手へ正の確率を持つことを必須にする。 */
  public static void requireSelectedPgActor(
      EpsilonDecisionPlayer.RolloutConfig actorRolloutConfig) {
    EpsilonDecisionPlayer.RolloutConfig actual =
        Objects.requireNonNull(actorRolloutConfig, "actorRolloutConfig");
    if (actual.selectionMode() != DecisionSelectionMode.FULL_SUPPORT
        || !actual.fullSupport().guaranteesFullLeafCoverage()) {
      throw new IllegalStateException(
          "production selected PG requires FULL_SUPPORT with positive exploration mass at every "
              + "policy node and a positive minimum leaf probability; actual="
              + actual.summary());
    }
  }

  /** 検証済み要求を対局へ渡し、収集処理速度を記録する。 */
  public static EpsilonDecisionArena.Metrics collect(Request request) throws Exception {
    return collect(request, EpsilonSettings.defaults());
  }

  public static EpsilonDecisionArena.Metrics collect(Request request, SettingsLoader config)
      throws Exception {
    Request actual = Objects.requireNonNull(request, "request");
    long[] totalSamples = {0L};
    EpsilonDecisionArena.CompletedGameSink countingSink =
        (gameIndex, gameSeed, game) -> {
          totalSamples[0] += game.samples().size();
          actual.sampleSink().accept(gameIndex, gameSeed, game);
        };
    int gamesInFlight =
        Math.min(actual.games(), config.bind(DecisionTrainArenaSettings.class).gamesInFlight());
    long startedNanos = System.nanoTime();
    EpsilonDecisionArena.Metrics metrics =
        EpsilonDecisionArena.collectPopulationGamesWithPayloadStore(
            actual.games(),
            actual.seedBase(),
            gamesInFlight,
            actual.actorEvaluator(),
            actual.actorSnapshotId(),
            actual.opponentIdsBySeat(),
            actual.snapshotEvaluatorProvider(),
            actual.actorRolloutConfig(),
            actual.opponentRolloutConfig(),
            countingSink,
            actual.trajectoryPayloadStore(),
            actual.grpInference(),
            actual.adaptiveExplorationSession(),
            actual.gameIndexPolicy(),
            config);
    long elapsedMillis = elapsedMillis(startedNanos);
    log.info(
        "Decision arena collection complete: games={} gamesInFlight={} samples={} "
            + "elapsedMs={} gamesPerSec={} decisionsPerSec={} "
            + "inferenceBatches={} inferenceRequests={} "
            + "avgInferenceBatch={} maxInferenceBatch={}",
        actual.games(),
        gamesInFlight,
        totalSamples[0],
        elapsedMillis,
        formatRate(actual.games(), elapsedMillis / 1000.0),
        formatRate(metrics.inferenceRequestCount(), elapsedMillis / 1000.0),
        metrics.inferenceBatchCount(),
        metrics.inferenceRequestCount(),
        String.format(Locale.ROOT, "%.3f", metrics.averageInferenceBatchSize()),
        metrics.maxInferenceBatchSize());
    return metrics;
  }

  private static long elapsedMillis(long startedNanos) {
    return (System.nanoTime() - startedNanos) / 1_000_000L;
  }

  private static String formatRate(double numerator, double denominator) {
    double value = denominator <= 0.0 ? numerator : numerator / denominator;
    return String.format(Locale.ROOT, "%.3f", value);
  }

  /** 一回の複数モデルによる収集に必要な全依存を明示する。 */
  public record Request(
      int games,
      long seedBase,
      EpsilonDecisionEvaluator actorEvaluator,
      long actorSnapshotId,
      long[][] opponentIdsBySeat,
      DecisionSnapshotEvaluatorProvider snapshotEvaluatorProvider,
      EpsilonDecisionPlayer.RolloutConfig actorRolloutConfig,
      EpsilonDecisionPlayer.RolloutConfig opponentRolloutConfig,
      EpsilonDecisionArena.CompletedGameSink sampleSink,
      EpsilonDecisionTrajectoryPayloadStore trajectoryPayloadStore,
      EpsilonGrpInference grpInference,
      DecisionAdaptiveExploration.MacroSession adaptiveExplorationSession,
      EpsilonDecisionArena.GameIndexPolicy gameIndexPolicy) {

    public Request {
      if (games <= 0) {
        throw new IllegalArgumentException("games must be positive");
      }
      actorEvaluator = Objects.requireNonNull(actorEvaluator, "actorEvaluator");
      opponentIdsBySeat = Objects.requireNonNull(opponentIdsBySeat, "opponentIdsBySeat");
      snapshotEvaluatorProvider =
          Objects.requireNonNull(snapshotEvaluatorProvider, "snapshotEvaluatorProvider");
      actorRolloutConfig = Objects.requireNonNull(actorRolloutConfig, "actorRolloutConfig");
      opponentRolloutConfig =
          Objects.requireNonNull(opponentRolloutConfig, "opponentRolloutConfig");
      sampleSink = Objects.requireNonNull(sampleSink, "sampleSink");
      trajectoryPayloadStore =
          Objects.requireNonNull(trajectoryPayloadStore, "trajectoryPayloadStore");
      gameIndexPolicy = Objects.requireNonNull(gameIndexPolicy, "gameIndexPolicy");
    }

    public static Request standard(
        int games,
        long seedBase,
        EpsilonDecisionEvaluator actorEvaluator,
        long actorSnapshotId,
        long[][] opponentIdsBySeat,
        DecisionSnapshotEvaluatorProvider snapshotEvaluatorProvider,
        EpsilonDecisionPlayer.RolloutConfig actorRolloutConfig,
        EpsilonDecisionPlayer.RolloutConfig opponentRolloutConfig,
        EpsilonDecisionArena.CompletedGameSink sampleSink,
        EpsilonDecisionTrajectoryPayloadStore trajectoryPayloadStore) {
      return new Request(
          games,
          seedBase,
          actorEvaluator,
          actorSnapshotId,
          opponentIdsBySeat,
          snapshotEvaluatorProvider,
          actorRolloutConfig,
          opponentRolloutConfig,
          sampleSink,
          trajectoryPayloadStore,
          null,
          null,
          EpsilonDecisionArena.GameIndexPolicy.standard());
    }

    public Request withGrpInference(EpsilonGrpInference inference) {
      return new Request(
          games,
          seedBase,
          actorEvaluator,
          actorSnapshotId,
          opponentIdsBySeat,
          snapshotEvaluatorProvider,
          actorRolloutConfig,
          opponentRolloutConfig,
          sampleSink,
          trajectoryPayloadStore,
          inference,
          adaptiveExplorationSession,
          gameIndexPolicy);
    }

    public Request withAdaptiveExploration(
        DecisionAdaptiveExploration.MacroSession explorationSession) {
      return new Request(
          games,
          seedBase,
          actorEvaluator,
          actorSnapshotId,
          opponentIdsBySeat,
          snapshotEvaluatorProvider,
          actorRolloutConfig,
          opponentRolloutConfig,
          sampleSink,
          trajectoryPayloadStore,
          grpInference,
          Objects.requireNonNull(explorationSession, "explorationSession"),
          gameIndexPolicy);
    }

    public Request withGameIndexPolicy(EpsilonDecisionArena.GameIndexPolicy policy) {
      return new Request(
          games,
          seedBase,
          actorEvaluator,
          actorSnapshotId,
          opponentIdsBySeat,
          snapshotEvaluatorProvider,
          actorRolloutConfig,
          opponentRolloutConfig,
          sampleSink,
          trajectoryPayloadStore,
          grpInference,
          adaptiveExplorationSession,
          policy);
    }
  }
}
