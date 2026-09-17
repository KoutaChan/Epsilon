package com.epsilon.nano.ai.decision.benchmark;

import ai.djl.Device;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTrainArenaSettings;
import com.epsilon.config.settings.GrpSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.GameState;
import com.epsilon.nano.ai.decision.arena.EpsilonDecisionArena;
import com.epsilon.nano.ai.decision.arena.EpsilonDecisionPlayer;
import com.epsilon.nano.ai.decision.arena.EpsilonDecisionPopulationCollector;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionTrajectoryPayloadStore;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionEvaluatorFactory;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.nano.config.settings.DecisionInferenceSettings;
import com.epsilon.nano.config.settings.DecisionSelectedPgCampaignSettings;
import com.epsilon.runtime.DecisionDevicePipeline;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.runtime.InferenceDispatcher;
import com.epsilon.runtime.InputBatchProfile;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * 自己対局による学習データ収集の処理速度を、本番と同じ実行経路で計測する。
 *
 * <p>行動を選ぶ学習対象モデルと対戦相手を、同じチェックポイントから別々に読み込む。学習対象モデルでは方策と価値、対戦相手では方策だけを推論する。収集データは一時ファイルに保存し、対局完了時に削除する。
 */
public final class EpsilonDecisionCollectionBenchmark {

  private static final long ACTOR_SNAPSHOT_ID = 2L;
  private static final long OPPONENT_SNAPSHOT_ID = 1L;

  private EpsilonDecisionCollectionBenchmark() {}

  /**
   * 同じ実行時でウォームアップした後、指定チェックポイントの自己対局収集を計測する。
   *
   * @param checkpointDir Decision・任意GRP チェックポイントのルートディレクトリ
   * @param games 生成する半荘数
   * @param seedBase 対局乱数シードの先頭
   * @param scratchDir 保存しない対局中の行動履歴データの一時保存先
   * @param warmupGames 計測前に別乱数シードで完走する対局数
   * @return 経過時間、判断/s、対局/s、サンプル数、推論バッチ指標
   * @throws Exception チェックポイント読込、対局の実行、または一時データ本体操作に失敗した場合
   */
  public static Report run(
      Path checkpointDir, int games, long seedBase, Path scratchDir, int warmupGames)
      throws Exception {
    return run(
        checkpointDir,
        games,
        seedBase,
        scratchDir,
        warmupGames,
        com.epsilon.nano.config.settings.EpsilonSettings.defaults());
  }

  public static Report run(
      Path checkpointDir,
      int games,
      long seedBase,
      Path scratchDir,
      int warmupGames,
      SettingsLoader config)
      throws Exception {
    if (games <= 0 || warmupGames < 0) {
      throw new IllegalArgumentException(
          "benchmark games must be positive and warmup non-negative");
    }
    Path checkpoint = requireBenchmarkCheckpoint(checkpointDir);
    Files.createDirectories(scratchDir);
    Device grpDevice =
        NetworkFactory.getGrpDevices(config.bind(com.epsilon.config.settings.DeviceSettings.class))
            .primary();
    EpsilonDecisionPlayer.RolloutConfig actorRollout =
        EpsilonDecisionPlayer.RolloutConfig.fromSettings(config)
            .withCausalTraceLambda(
                config.bind(DecisionSelectedPgCampaignSettings.class).causalTraceLambda());
    EpsilonDecisionPlayer.RolloutConfig opponentRollout =
        EpsilonDecisionPlayer.RolloutConfig.opponentFromSettings(config);
    EpsilonDecisionPopulationCollector.requireSelectedPgActor(actorRollout);
    long[][] opponentIds = opponentIds();
    long[] samples = {0L};

    try (var context = new DecisionExecutionContext();
        EpsilonDecisionEvaluatorFactory.Handle actor =
            EpsilonDecisionEvaluatorFactory.openCheckpointEvaluator(checkpoint, context, config);
        EpsilonDecisionEvaluatorFactory.Handle opponent =
            EpsilonDecisionEvaluatorFactory.openPolicyCheckpointEvaluator(
                checkpoint, context, config);
        EpsilonGrpTrainingSession grp =
            config.bind(GrpSettings.class).enabled()
                ? EpsilonGrpTrainingSession.openInferenceOnly(checkpointDir, grpDevice, config)
                : EpsilonGrpTrainingSession.disabled(checkpointDir.resolve("grp"));
        EpsilonDecisionTrajectoryPayloadStore payloadStore =
            EpsilonDecisionTrajectoryPayloadStore.fromSettings(
                scratchDir,
                false,
                config.bind(
                    com.epsilon.config.settings.DecisionTrainInFlightSpoolSettings.class))) {
      if (warmupGames > 0) {
        try (var warmupStore =
            EpsilonDecisionTrajectoryPayloadStore.fromSettings(
                scratchDir.resolve("warmup"),
                false,
                config.bind(
                    com.epsilon.config.settings.DecisionTrainInFlightSpoolSettings.class))) {
          EpsilonDecisionPopulationCollector.collect(
              EpsilonDecisionPopulationCollector.Request.standard(
                      warmupGames,
                      seedBase ^ 0x6A09E667F3BCC909L,
                      actor.evaluator(),
                      ACTOR_SNAPSHOT_ID,
                      opponentIds,
                      ignored -> opponent.evaluator(),
                      actorRollout,
                      opponentRollout,
                      (gameIndex, gameSeed, game) -> {},
                      warmupStore)
                  .withGrpInference(grp.inference()),
              config);
        }
      }
      context.awaitInferenceIdle();
      context.resetInferenceMetrics();
      EpsilonDecisionArena.Metrics metrics;
      long elapsedMillis;
      try (var event = InputBatchProfile.beginCollection(games, warmupGames)) {
        if (event != null) {
          event.series = "nano";
        }
        long startedNanos = System.nanoTime();
        metrics =
            EpsilonDecisionPopulationCollector.collect(
                EpsilonDecisionPopulationCollector.Request.standard(
                        games,
                        seedBase,
                        actor.evaluator(),
                        ACTOR_SNAPSHOT_ID,
                        opponentIds,
                        ignored -> opponent.evaluator(),
                        actorRollout,
                        opponentRollout,
                        (gameIndex, gameSeed, game) -> samples[0] += game.samples().size(),
                        payloadStore)
                    .withGrpInference(grp.inference()),
                config);
        elapsedMillis = (System.nanoTime() - startedNanos) / 1_000_000L;
        if (event != null) {
          event.success = true;
        }
      }
      int configuredGamesInFlight = config.bind(DecisionTrainArenaSettings.class).gamesInFlight();
      return new Report(
          games,
          warmupGames,
          configuredGamesInFlight,
          Math.min(games, configuredGamesInFlight),
          samples[0],
          elapsedMillis,
          1_000.0 * metrics.inferenceRequestCount() / elapsedMillis,
          1_000.0 * games / elapsedMillis,
          metrics,
          config.bind(DecisionInferenceSettings.class).computePrecision(),
          actor.devices(),
          opponent.devices(),
          context.inferenceMetrics(),
          context.inferenceDispatchMetrics());
    }
  }

  /** 処理速度測定では、対局収集に使う採用モデル昇格前でも構造互換な明示チェックポイントを利用できる。 */
  private static Path requireBenchmarkCheckpoint(Path checkpointDir) throws IOException {
    Path checkpoint = EpsilonDecisionCheckpointManager.resolveExistingStrict(checkpointDir);
    if (checkpoint == null) {
      throw new IOException("Decision benchmark checkpoint not found: " + checkpointDir);
    }
    return checkpoint;
  }

  private static long[][] opponentIds() {
    long[][] result = new long[GameState.NUM_PLAYERS][GameState.NUM_PLAYERS - 1];
    for (long[] ids : result) {
      java.util.Arrays.fill(ids, OPPONENT_SNAPSHOT_ID);
    }
    return result;
  }

  /**
   * 収集性能測定の実測値。
   *
   * @param games 完走した対局数
   * @param warmupGames 計測区間外で完走したウォームアップ対局数
   * @param configuredGamesInFlight 設定した同時進行対局数
   * @param effectiveGamesInFlight 計測対局数で制限した同時進行対局数の上限
   * @param samples 収集した Decision サンプル数
   * @param elapsedMillis 計測区間の経過ミリ秒
   * @param decisionsPerSecond 推論要求数を計測区間の経過時間で割った主処理速度指標
   * @param gamesPerSecond 完走対局数を経過時間で割った補助処理速度指標
   * @param arenaMetrics 対局内部のバッチ処理・推論指標
   * @param computePrecision モデル順伝播の実効精度
   * @param actorDevices 学習器席を実行したデバイスの表示名
   * @param opponentDevices 対戦相手席を実行したデバイスの表示名
   * @param deviceMetrics 実GPU投入バッチの計測値。ウォームアップは含めない
   * @param dispatchMetrics 共通ホスト予約の容量と最大使用数。ウォームアップは含めない
   */
  public record Report(
      int games,
      int warmupGames,
      int configuredGamesInFlight,
      int effectiveGamesInFlight,
      long samples,
      long elapsedMillis,
      double decisionsPerSecond,
      double gamesPerSecond,
      EpsilonDecisionArena.Metrics arenaMetrics,
      DecisionComputePrecision computePrecision,
      String actorDevices,
      String opponentDevices,
      List<DecisionDevicePipeline.Metrics> deviceMetrics,
      InferenceDispatcher.Metrics dispatchMetrics) {}
}
