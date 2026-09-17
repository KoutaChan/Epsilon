package com.epsilon.major.ai.decision.training;

import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.major.ai.decision.EpsilonDecisionConstants;

/** Decision チェックポイントのマニフェスト。 */
public final class EpsilonDecisionCheckpointBundle {

  /** 現在読み書きするマニフェストスキーマバージョン。 */
  public static final int CURRENT_VERSION = 24;

  /** 方策・GRPによる事前予測付きDecision 価値を一つの更新の基準となるモデルへ格納する保存物形式。 */
  public static final String CURRENT_MODEL_FORMAT =
      "single-canonical-public-boundary-hl-gauss-utility-value-v3";

  /** 標準形式の Decision パラメーターファイル名。 */
  public static final String CURRENT_MODEL_FILE = "decision-0000.params";

  /** 保存されたマニフェストスキーマバージョン。 */
  public int checkpointVersion = CURRENT_VERSION;

  /** 新規チェックポイントの系列。識別子のない旧保存物は従来の構造契約で読む。 */
  public String series;

  /** 入力・ネットワーク・ヘッド意味を識別する構造 ID。 */
  public String architecture = EpsilonDecisionConstants.ARCHITECTURE_ID;

  /** HL-Gauss 値域と全順位効用を識別する、モデル固有の価値定義。 */
  public String valueDefinition;

  /** チェックポイントネットワークの隠れ層幅。 */
  public int hidden;

  /** 保存物の物理形式。 */
  public String modelFormat = CURRENT_MODEL_FORMAT;

  /** マニフェストから参照するパラメーターファイル名。 */
  public String modelFile = CURRENT_MODEL_FILE;

  /** 保存時点までのオプティマイザー更新回数。 */
  public int globalStep;

  /** 候補の学習反復回数。 */
  public int iteration;

  /** チェックポイントまでに生成した累積自己対局数。 */
  public int selfPlayGames;

  /** JSON 復号処理向けに既定値を持つ空マニフェストを作る。 */
  public EpsilonDecisionCheckpointBundle() {}

  /**
   * 現在スキーマ・構造で学習位置を記録するマニフェストを作る。
   *
   * @param globalStep オプティマイザー更新回数
   * @param iteration 学習反復回数
   * @param selfPlayGames 累積自己対局数
   */
  public EpsilonDecisionCheckpointBundle(
      int globalStep,
      int iteration,
      int selfPlayGames,
      int hidden,
      EpsilonUtilityProfile utilityProfile) {
    series = "epsilon";
    this.hidden = hidden;
    valueDefinition = EpsilonDecisionHlGauss.fingerprint(utilityProfile);
    this.globalStep = globalStep;
    this.iteration = iteration;
    this.selfPlayGames = selfPlayGames;
  }
}
