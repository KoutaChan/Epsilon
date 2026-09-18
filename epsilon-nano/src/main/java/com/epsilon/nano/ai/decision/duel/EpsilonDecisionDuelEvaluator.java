package com.epsilon.nano.ai.decision.duel;

import com.epsilon.ai.decision.duel.DuelEvaluation;
import com.epsilon.ai.decision.duel.EpsilonDecisionDuelArena;
import com.epsilon.config.settings.DecisionEvalVsSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 同じ牌山で評価対象を4席に入れ替え、固定した対戦相手と比較する。 */
public final class EpsilonDecisionDuelEvaluator {

  private static final Logger log = LoggerFactory.getLogger(EpsilonDecisionDuelEvaluator.class);

  private EpsilonDecisionDuelEvaluator() {}

  /**
   * 二つのチェックポイントを同一牌山・4席を入れ替えた対局で比較する。
   *
   * @param candidateCheckpointDir 評価対象チェックポイントまたはそのルートディレクトリディレクトリ
   * @param opponentCheckpointDir 固定対戦相手チェックポイントまたはそのルートディレクトリディレクトリ
   * @param games 実行する総対局数。4席を入れ替えた対局を構成できる値
   * @param seedBase 牌山乱数シード列の先頭値
   * @param context ワークフローが所有するGPU実行資源
   * @return 対応をそろえた順位差と順位・得点指標を集約した対戦比較結果
   * @throws IOException チェックポイントの解決、読込、または対局の実行に失敗した場合
   */
  public static DuelEvaluation.Result evaluate(
      Path candidateCheckpointDir,
      Path opponentCheckpointDir,
      int games,
      long seedBase,
      DecisionExecutionContext context)
      throws IOException {
    return evaluate(
        candidateCheckpointDir,
        opponentCheckpointDir,
        games,
        seedBase,
        context,
        EpsilonSettings.defaults());
  }

  public static DuelEvaluation.Result evaluate(
      Path candidateCheckpointDir,
      Path opponentCheckpointDir,
      int games,
      long seedBase,
      DecisionExecutionContext context,
      SettingsLoader config)
      throws IOException {
    try (EpsilonDecisionDuelSession session =
        EpsilonDecisionDuelSession.open(
            candidateCheckpointDir, opponentCheckpointDir, context, config)) {
      EpsilonDecisionDuelArena.Evaluation evaluation =
          session.evaluate(
              games, seedBase, 0L, config.bind(DecisionEvalVsSettings.class).gamesInFlight());
      DuelEvaluation.Metrics metrics = evaluation.metrics();
      log.info(
          "Decision duel complete: games={} gamesInFlight={} completedGames={}"
              + " candidateShards={} opponentShards={} devices={}/{} inferenceBatches={}"
              + " inferenceRequests={} avgInferenceBatch={} maxInferenceBatch={} wallSeeds={}"
              + " pairedRankDeltaMean={} pairedRankDeltaSe={} pairedRankDeltaLcb={}",
          metrics.games(),
          metrics.gamesInFlight(),
          metrics.completedGames(),
          session.candidateShards(),
          session.opponentShards(),
          session.candidateDevices(),
          session.opponentDevices(),
          metrics.inferenceBatches(),
          metrics.inferenceRequests(),
          metrics.averageInferenceBatch(),
          metrics.maxInferenceBatch(),
          evaluation.result().wallSeeds(),
          evaluation.result().pairedRankDeltaMean(),
          evaluation.result().pairedRankDeltaSe(),
          evaluation.result().pairedRankDeltaLcb());
      return evaluation.result();
    }
  }
}
