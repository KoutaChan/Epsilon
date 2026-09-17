package com.epsilon.ai.decision.duel;

import java.nio.file.Path;
import java.util.function.Consumer;

/** 同じモデルと牌山を使い、指定した範囲の各牌山について4通りの席替えを同期実行するインターフェース。 */
public interface DuelEvaluationSource {

  /** 評価対象を識別するパス。共通評価処理はチェックポイントの内容を解釈しない。 */
  Path candidateCheckpoint();

  /** 比較対象を識別するパス。共通評価処理はチェックポイントの内容を解釈しない。 */
  Path opponentCheckpoint();

  /**
   * 指定区間を完走し、各牌山の4席が揃うたびに受け取り先へ通知する。資源の解放は、評価元を作成した系列側で行う。
   *
   * @param wallSeeds 実行する牌山数。各牌山につき4通りの席替えを行う
   * @param seedBase 牌山シードの導出に用いるシード
   * @param firstWallFamilyId 区間の先頭牌山インデックス
   * @param gamesInFlight 同時進行対局数
   * @param sink 4通りの席替えを終えた牌山の観測を同期的に受け取る処理
   */
  DuelEvaluation evaluateWalls(
      int wallSeeds,
      long seedBase,
      long firstWallFamilyId,
      int gamesInFlight,
      Consumer<DuelEvaluation.WallOutcome> sink);
}
