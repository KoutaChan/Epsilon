package com.epsilon.major.ai.belief;

/** belief チェックポイントのマニフェスト。 */
public final class EpsilonBeliefCheckpointBundle {

  /** 現在読み書きするマニフェストスキーマバージョン。 */
  public static final int CURRENT_VERSION = 4;

  /** 保存されたマニフェストスキーマバージョン。 */
  public int checkpointVersion = CURRENT_VERSION;

  /** パラメーターの意味と構造を識別するネットワーク構造 ID。 */
  public String architecture = EpsilonBeliefNetwork.ARCHITECTURE_ID;

  /** 保存時点までのオプティマイザー更新回数。 */
  public int globalStep;

  /** 保存時の系列・入力とモデル幅。旧形式では未記録。 */
  public String series;

  public String inputFingerprint;
  public int hidden;

  /** 保存時点の外側学習反復回数。 */
  public int iteration;

  /** JSON 復号処理向けに既定値を持つ空マニフェストを作る。 */
  public EpsilonBeliefCheckpointBundle() {}

  /**
   * 現在スキーマ・構造で学習位置を記録するマニフェストを作る。
   *
   * @param globalStep オプティマイザー更新回数
   * @param iteration 外側学習反復回数
   */
  public EpsilonBeliefCheckpointBundle(int globalStep, int iteration) {
    this.globalStep = globalStep;
    this.iteration = iteration;
  }
}
