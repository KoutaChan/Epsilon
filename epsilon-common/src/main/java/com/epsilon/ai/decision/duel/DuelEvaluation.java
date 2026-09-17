package com.epsilon.ai.decision.duel;

import com.epsilon.runtime.InferenceAdmission;
import java.nio.file.Path;

/** 個々の観測を保持せずに集約した、対戦評価の一区間の結果と実行指標。 */
public record DuelEvaluation(Result result, Metrics metrics) {

  /**
   * 同一牌山の4通りの席替えを1標本として集約した比較対局結果。LCB は {@code mean - 1.96 * SE}。
   *
   * @param candidateCheckpoint 評価対象チェックポイント
   * @param opponentCheckpoint 比較対象チェックポイント
   * @param games 実行した総対局数
   * @param wallSeeds 独立標本として使った牌山シード数
   * @param pairedRankDeltaMean 対応する対局で比較した、候補と対戦相手の順位効用値の差の平均
   * @param pairedRankDeltaSe 対応する対局間の順位効用値の差の標準誤差
   * @param pairedRankDeltaLcb 対応する対局間の順位効用値の差の95%信頼下限
   * @param scoreAdvantage 候補視点の平均点差
   * @param candidateAverageRank 候補の平均順位
   * @param opponentAverageRank 比較相手の平均順位
   * @param candidateTopRate 候補の1着率
   * @param opponentTopRate 比較相手の1着率
   * @param candidateLastRate 候補の4着率
   * @param opponentLastRate 比較相手の4着率
   * @param utilityProfileAdvantages 効用値設定ごとの候補優位量
   */
  public record Result(
      Path candidateCheckpoint,
      Path opponentCheckpoint,
      int games,
      int wallSeeds,
      double pairedRankDeltaMean,
      double pairedRankDeltaSe,
      double pairedRankDeltaLcb,
      double scoreAdvantage,
      double candidateAverageRank,
      double opponentAverageRank,
      double candidateTopRate,
      double opponentTopRate,
      double candidateLastRate,
      double opponentLastRate,
      double[] utilityProfileAdvantages) {}

  /**
   * 対戦評価における対局進行と推論バッチ化の実行指標。
   *
   * @param games 設定した総対局数
   * @param gamesInFlight 同時進行対局数
   * @param completedGames 完走した対局数
   * @param inferenceBatches スケジューラーが投入した型で区別された推論バッチ数
   * @param inferenceRequests バッチへ投入した推論要求数
   * @param averageInferenceBatch 投入バッチの平均行数
   * @param maxInferenceBatch 投入した最大バッチ行数
   */
  public record Metrics(
      int games,
      int gamesInFlight,
      int completedGames,
      long inferenceBatches,
      long inferenceRequests,
      double averageInferenceBatch,
      int maxInferenceBatch,
      InferenceAdmission.Metrics batching) {}

  /**
   * 4通りの席替えをすべて含む、信頼区間の計算に使う独立観測。
   *
   * @param wallIndex 評価列内の牌山インデックス
   * @param wallSeed 牌山を再現するシード
   * @param pairedRankDelta 4席で平均した候補の平均順位差
   * @param pairedConfiguredUtilityDelta 4席で平均した有効効用値設定の差
   */
  public record WallOutcome(
      long wallIndex, long wallSeed, double pairedRankDelta, double pairedConfiguredUtilityDelta) {}
}
