package com.epsilon.nano.ai.belief;

/** Belief チェックポイントの構成と学習進捗を記録するマニフェスト。 */
public final class EpsilonBeliefCheckpointBundle {

  /** 現在読み書きするマニフェストスキーマバージョン。 */
  public static final int CURRENT_VERSION = 4;

  /** 保存されたマニフェストスキーマバージョン。 */
  public int checkpointVersion = CURRENT_VERSION;

  /** パラメータの構成と意味を識別するモデル構造 ID。 */
  public String architecture = EpsilonBeliefNetwork.ARCHITECTURE_ID;

  /** 保存時点までのオプティマイザー更新回数。 */
  public int globalStep;

  /** 保存時の系列・入力とモデル幅。旧形式では未記録。 */
  public String series;

  public String inputFingerprint;
  public int hidden;

  /** 保存時点の学習全体の反復番号。 */
  public int iteration;

  /** JSON 復号処理向けに既定値を持つ空マニフェストを作る。 */
  public EpsilonBeliefCheckpointBundle() {}

  /**
   * 現在のデータ形式とモデル構造に対応するマニフェストを作り、学習の進捗を記録する。
   *
   * @param globalStep オプティマイザー更新回数
   * @param iteration 学習全体の反復番号
   */
  public EpsilonBeliefCheckpointBundle(int globalStep, int iteration) {
    this.globalStep = globalStep;
    this.iteration = iteration;
  }
}
