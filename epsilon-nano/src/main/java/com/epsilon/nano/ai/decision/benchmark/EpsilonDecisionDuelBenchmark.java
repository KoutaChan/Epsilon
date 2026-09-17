package com.epsilon.nano.ai.decision.benchmark;

import com.epsilon.ai.decision.duel.DuelEvaluation;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.nano.ai.decision.arena.EpsilonDecisionDuelArena;
import com.epsilon.nano.ai.decision.duel.EpsilonDecisionDuelSession;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.nio.file.Path;

/**
 * 同じ牌山で席を入れ替える対戦評価の処理速度を、本番と同じ実行経路で計測する。
 *
 * <p>チェックポイントの読み込みとウォームアップは計測時間に含めない。計測区間では {@link EpsilonDecisionDuelArena} を使う。
 */
public final class EpsilonDecisionDuelBenchmark {

  private EpsilonDecisionDuelBenchmark() {}

  /**
   * 候補と比較元を一度だけ読み込み、ウォームアップ後の固定牌山処理速度を測る。
   *
   * @param candidateCheckpointDir 候補チェックポイントまたはチェックポイントのルートディレクトリ
   * @param parentCheckpointDir 固定した比較元チェックポイントまたはチェックポイントのルートディレクトリ
   * @param warmupWallSeeds 計測前に消費する牌山数
   * @param measuredWallSeeds 処理速度計測に使う牌山数
   * @param seedBase 牌山乱数シード系列の基点
   * @param gamesInFlight 同時進行する対局数
   * @return 処理速度とバッチ指標
   * @throws IOException チェックポイントの解決または読込に失敗した場合
   */
  public static Report run(
      Path candidateCheckpointDir,
      Path parentCheckpointDir,
      int warmupWallSeeds,
      int measuredWallSeeds,
      long seedBase,
      int gamesInFlight)
      throws IOException {
    return run(
        candidateCheckpointDir,
        parentCheckpointDir,
        warmupWallSeeds,
        measuredWallSeeds,
        seedBase,
        gamesInFlight,
        com.epsilon.nano.config.settings.EpsilonSettings.defaults());
  }

  public static Report run(
      Path candidateCheckpointDir,
      Path parentCheckpointDir,
      int warmupWallSeeds,
      int measuredWallSeeds,
      long seedBase,
      int gamesInFlight,
      SettingsLoader config)
      throws IOException {
    if (warmupWallSeeds < 0) {
      throw new IllegalArgumentException("warmupWallSeeds must be non-negative");
    }
    if (measuredWallSeeds <= 0) {
      throw new IllegalArgumentException("measuredWallSeeds must be positive");
    }

    try (var context = new DecisionExecutionContext();
        EpsilonDecisionDuelSession session =
            EpsilonDecisionDuelSession.open(
                candidateCheckpointDir, parentCheckpointDir, context, config)) {
      if (warmupWallSeeds > 0) {
        session.evaluate(Math.multiplyExact(warmupWallSeeds, 4), seedBase, 0L, gamesInFlight);
      }

      long startedAt = System.nanoTime();
      EpsilonDecisionDuelArena.Evaluation evaluation =
          session.evaluate(
              Math.multiplyExact(measuredWallSeeds, 4), seedBase, warmupWallSeeds, gamesInFlight);
      long elapsedNanos = System.nanoTime() - startedAt;

      DuelEvaluation.Metrics metrics = evaluation.metrics();
      return new Report(
          session.candidateCheckpoint(),
          session.opponentCheckpoint(),
          warmupWallSeeds,
          measuredWallSeeds,
          metrics.completedGames(),
          elapsedNanos,
          metrics.completedGames() * 1_000_000_000.0 / elapsedNanos,
          measuredWallSeeds * 1_000_000_000.0 / elapsedNanos,
          metrics.inferenceBatches(),
          metrics.inferenceRequests(),
          metrics.averageInferenceBatch(),
          metrics.maxInferenceBatch());
    }
  }

  /**
   * 一回の計測結果。
   *
   * @param candidateCheckpoint 解決済み候補チェックポイント
   * @param parentCheckpoint 解決済み比較元チェックポイント
   * @param warmupWallSeeds 計測前に消費した牌山数
   * @param measuredWallSeeds 計測した独立牌山数
   * @param games 完走した対局数
   * @param elapsedNanos 計測区間の経過時間
   * @param gamesPerSecond 対局処理速度
   * @param wallsPerSecond 独立牌山処理速度
   * @param inferenceBatches スケジューラーが投入した型付き推論バッチ数
   * @param inferenceRequests 推論したDecision行数
   * @param averageInferenceBatch 投入バッチの平均行数
   * @param maximumInferenceBatch 投入した最大バッチ行数
   */
  public record Report(
      Path candidateCheckpoint,
      Path parentCheckpoint,
      int warmupWallSeeds,
      int measuredWallSeeds,
      int games,
      long elapsedNanos,
      double gamesPerSecond,
      double wallsPerSecond,
      long inferenceBatches,
      long inferenceRequests,
      double averageInferenceBatch,
      int maximumInferenceBatch) {}
}
