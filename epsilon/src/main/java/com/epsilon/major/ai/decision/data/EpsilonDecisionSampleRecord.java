package com.epsilon.major.ai.decision.data;

import com.epsilon.ai.decision.DecisionBranchTarget;
import com.epsilon.core.DecisionLearningRole;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;

/** 対局データの収集・保存・学習が共有する、型付きの判断サンプル。 */
public interface EpsilonDecisionSampleRecord {

  /**
   * 元の Decision 観測を返す。
   *
   * @return 1 判断行を保持する型付きホスト入力
   */
  DecisionHostBatch input();

  /**
   * 動的合法候補数を返す。
   *
   * @return この判断行の動的合法候補数
   */
  default int legalActionCount() {
    return input().legalActionCount(0);
  }

  /**
   * 指定合法格納位置の行動 ID を返す。
   *
   * @param slot コンパクト合法格納位置
   * @return 指定格納位置の安定行動 ID
   */
  default int legalActionIdBySlot(int slot) {
    return input().legalActionId(0, slot);
  }

  /**
   * 意思決定者の絶対席を返す。
   *
   * @return 意思決定者の絶対席
   */
  default int playerSeat() {
    return input().playerSeat(0);
  }

  /**
   * 応答元プレイヤーの相対席を返す。
   *
   * @return 応答元プレイヤーの自家相対席。該当しない場合はスキーマの欠損値
   */
  default int sourcePlayerRelativeSeat() {
    return input().sourcePlayerRelativeSeat(0);
  }

  /**
   * 現在手番プレイヤーの相対席を返す。
   *
   * @return 現在手番プレイヤーの自家相対席
   */
  default int currentPlayerRelativeSeat() {
    return input().currentPlayerRelativeSeat(0);
  }

  /**
   * 実際に選択した格納位置を返す。
   *
   * @return 探索適用後の方策が実際に選んだコンパクト合法格納位置
   */
  int chosenLegalSlot();

  /**
   * 実際に選択した行動 ID を返す。
   *
   * @return 選択格納位置に対応する安定行動 ID
   */
  int chosenActionId();

  /**
   * 選択行動の探索適用後の対数確率を返す。
   *
   * @return 選択行動の探索適用後の対数確率
   */
  float behaviorLogProb();

  /**
   * 価値教師値を返す。
   *
   * @return スカラー効用教師値
   */
  float valueTarget();

  /**
   * 選択行動アドバンテージを返す。
   *
   * @return スカラー効用アドバンテージ
   */
  float advantage();

  /**
   * 対局の最終順位を返す。
   *
   * @return 0始まりの最終順位
   */
  int finalRank();

  /**
   * 実際に標本化した方策を返す。
   *
   * @return 探索後探索適用後の方策の合法格納位置確率
   */
  float[] behaviorPolicy();

  /**
   * 探索前のモデル方策を返す。
   *
   * @return 探索前対局生成方策の合法格納位置確率
   */
  float[] rolloutPolicy();

  /**
   * 方策識別情報を返す。
   *
   * @return サンプルを生成した学習側モデルのスナップショット ID
   */
  long actorSnapshotId();

  /**
   * 効用の定義識別情報を返す。
   *
   * @return 対局で使った効用の定義インデックス
   */
  int ruleProfile();

  /**
   * 入力元対局識別情報を返す。
   *
   * @return サンプルを生成した対局 ID
   */
  long gameId();

  /** 半荘内の0始まり局境界インデックスを返す。 */
  int boundaryIndex();

  /** 同一席の因果時間順を検証する判断通し番号を返す。 */
  int seatDecisionOrdinal();

  /**
   * GRP の局境界列を返す。
   *
   * @return 局境界順の GRP 特徴量列
   */
  float[] grpFeatureSequence();

  /**
   * GRP の終局順位コードを返す。
   *
   * @return GRP 用に符号化した終局順位
   */
  int grpFinalRanksCode();

  /** 選択が方策遷移列、価値遷移列のどちらを進めるか返す。 */
  DecisionLearningRole learningRole();

  DecisionBranchTarget branchTarget();

  /**
   * 価値教師値とアドバンテージを置換した同一観測のレコードを返す。
   *
   * @param nextValueTarget 新しいスカラー効用教師値
   * @param nextAdvantage 新しいスカラー選択行動アドバンテージ
   * @return 教師値以外の識別情報を保ったサンプルレコード
   */
  EpsilonDecisionSampleRecord withTrainingTargets(float nextValueTarget, float nextAdvantage);

  /**
   * 選択行動の探索適用後の確率を返す。
   *
   * @return 選択格納位置の探索適用後の確率
   */
  default float behaviorProb() {
    return behaviorPolicy()[chosenLegalSlot()];
  }

  /**
   * 必要なら読み込みを遅延したデータ本体を読み、標準形式のサンプルを返す。
   *
   * @return 読み込みを遅延したデータ本体を解決した標準形式のメモリ内サンプル
   */
  EpsilonDecisionSample materialize();
}
