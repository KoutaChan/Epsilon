package com.epsilon.nano.ai.decision.duel;

import com.epsilon.ai.decision.duel.DuelEvaluation;
import com.epsilon.ai.decision.duel.DuelEvaluationSource;
import com.epsilon.config.settings.DecisionEvalVsSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.GameState;
import com.epsilon.nano.ai.decision.arena.EpsilonDecisionDuelArena;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionEvaluatorFactory;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.function.Consumer;

/**
 * 対戦比較で使う2つのチェックポイントと、方策のみを評価する推論器を管理する。
 *
 * <p>複数段階の評価で同じモデルを使い、読み直しを避ける。乱数シードの範囲と対局の実行設定は、各 {@link #evaluate(int, long, long, int)}
 * 呼び出しで指定する。
 */
public final class EpsilonDecisionDuelSession implements DuelEvaluationSource, AutoCloseable {

  private final SettingsLoader config;
  private final EpsilonDecisionEvaluatorFactory.GreedyHandle candidate;
  private final EpsilonDecisionEvaluatorFactory.GreedyHandle opponent;

  private EpsilonDecisionDuelSession(
      EpsilonDecisionEvaluatorFactory.GreedyHandle candidate,
      EpsilonDecisionEvaluatorFactory.GreedyHandle opponent,
      SettingsLoader config) {
    this.config = config;
    this.candidate = candidate;
    this.opponent = opponent;
  }

  /**
   * 候補と対戦相手を方策のみモデルの複製として一度ずつ開く。
   *
   * @param candidateCheckpoint 候補チェックポイントまたはそのルートディレクトリ
   * @param opponentCheckpoint 重みを固定した対戦相手チェックポイントまたはそのルートディレクトリ
   * @param context ワークフローが所有するGPU実行資源
   * @return 両チェックポイントを所有する対戦比較セッション
   * @throws IOException チェックポイントの解決または読込に失敗した場合
   */
  public static EpsilonDecisionDuelSession open(
      Path candidateCheckpoint, Path opponentCheckpoint, DecisionExecutionContext context)
      throws IOException {
    return open(candidateCheckpoint, opponentCheckpoint, context, EpsilonSettings.defaults());
  }

  public static EpsilonDecisionDuelSession open(
      Path candidateCheckpoint,
      Path opponentCheckpoint,
      DecisionExecutionContext context,
      SettingsLoader config)
      throws IOException {
    int maximumInferenceBatch = config.bind(DecisionEvalVsSettings.class).maximumInferenceBatch();
    EpsilonDecisionEvaluatorFactory.GreedyHandle candidate =
        EpsilonDecisionEvaluatorFactory.openGreedyPolicyCheckpointEvaluator(
            candidateCheckpoint, maximumInferenceBatch, context, config);
    try {
      return new EpsilonDecisionDuelSession(
          candidate,
          EpsilonDecisionEvaluatorFactory.openGreedyPolicyCheckpointEvaluator(
              opponentCheckpoint, maximumInferenceBatch, context, config),
          config);
    } catch (IOException | RuntimeException | Error failure) {
      candidate.close();
      throw failure;
    }
  }

  /**
   * 指定した新しく生成した牌山範囲を4席を入れ替えた対局で評価する。
   *
   * @param games 対局数。完全な4席を入れ替えた対局へ切り上げる
   * @param seedBase 牌山乱数シード系列の基点
   * @param firstWallFamilyId 最初に使う同一牌山の対局組オフセット
   * @param gamesInFlight 同時進行する対局数
   * @return 対応をそろえた対戦比較の集約値、観測、処理速度指標
   */
  public EpsilonDecisionDuelArena.Evaluation evaluate(
      int games, long seedBase, long firstWallFamilyId, int gamesInFlight) {
    return EpsilonDecisionDuelArena.evaluateDuel(
        candidate.checkpointPath(),
        opponent.checkpointPath(),
        candidate.evaluator(),
        opponent.evaluator(),
        games,
        seedBase,
        firstWallFamilyId,
        gamesInFlight,
        config);
  }

  /**
   * 全席順を入れ替えた対局を保持せず、完了した牌山を逐次渡しながら一つの連続対局で評価する。
   *
   * @param wallSeeds 完全な4席を入れ替えた対局で評価する牌山数
   * @param seedBase 牌山乱数シード系列の基点
   * @param firstWallFamilyId 最初に使う同一牌山の対局組オフセット
   * @param gamesInFlight 同時進行する対局数
   * @param wallOutcomeSink 完了した4席を入れ替えた対局の受け取り先
   * @return 観測リストを含まない集約結果と処理速度指標
   */
  @Override
  public DuelEvaluation evaluateWalls(
      int wallSeeds,
      long seedBase,
      long firstWallFamilyId,
      int gamesInFlight,
      Consumer<DuelEvaluation.WallOutcome> wallOutcomeSink) {
    return EpsilonDecisionDuelArena.evaluateDuelStreaming(
        candidate.checkpointPath(),
        opponent.checkpointPath(),
        candidate.evaluator(),
        opponent.evaluator(),
        Math.multiplyExact(wallSeeds, GameState.NUM_PLAYERS),
        seedBase,
        firstWallFamilyId,
        gamesInFlight,
        wallOutcomeSink,
        config);
  }

  @Override
  public Path candidateCheckpoint() {
    return candidate.checkpointPath();
  }

  @Override
  public Path opponentCheckpoint() {
    return opponent.checkpointPath();
  }

  int candidateShards() {
    return candidate.shards();
  }

  int opponentShards() {
    return opponent.shards();
  }

  String candidateDevices() {
    return candidate.devices();
  }

  String opponentDevices() {
    return opponent.devices();
  }

  @Override
  public void close() {
    Throwable failure = null;
    try {
      opponent.close();
    } catch (RuntimeException | Error closeFailure) {
      failure = closeFailure;
    }
    try {
      candidate.close();
    } catch (RuntimeException | Error closeFailure) {
      if (failure == null) {
        failure = closeFailure;
      } else {
        failure.addSuppressed(closeFailure);
      }
    }
    if (failure instanceof RuntimeException runtime) {
      throw runtime;
    }
    if (failure instanceof Error error) {
      throw error;
    }
  }
}
