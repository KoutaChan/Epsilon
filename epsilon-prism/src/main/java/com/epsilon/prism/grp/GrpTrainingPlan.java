package com.epsilon.prism.grp;

import java.time.Duration;

/**
 * GRP の学習 1 回の超パラメタ（{@link GrpGameTrainer}）。値はコードの定数（{@link #of}）。torch の演算スレッドの数は呼び手が決める（学習の
 * JVM では {@link GrpBackground} が学習の間だけ grpThreads にする）。
 *
 * @param init 始め（WARM は始めの GRP の重みから、SCRATCH は白紙）
 * @param batchGames 1 歩の半荘の数
 * @param learningRate 学習率
 * @param maxEpochs 窓を見る回数の上限
 * @param seed 並べ替えと白紙の初期値の種
 * @param deadline 学習の締め切り（過ぎたら止める）
 */
public record GrpTrainingPlan(
    GrpInit init, int batchGames, double learningRate, int maxEpochs, long seed, Duration deadline) {
  private static final int BATCH_GAMES = 128;

  public GrpTrainingPlan {
    if (batchGames < 1 || maxEpochs < 1)
      throw new IllegalArgumentException("GRP training plan needs positive batch and epochs");
    if (!(learningRate > 0)) throw new IllegalArgumentException("GRP learning rate must be positive");
    if (deadline.isNegative()) throw new IllegalArgumentException("GRP training deadline must not be negative");
  }

  /** 締め切りだけを deadline にした計画（試行の残りの時間で回す学習）。 */
  public GrpTrainingPlan within(Duration deadline) {
    return new GrpTrainingPlan(init, batchGames, learningRate, maxEpochs, seed, deadline);
  }

  /**
   * 始め init の学習の計画。締め切りは、窓の上限（8 コホート × 65,536 半荘）を上限の epoch まで回す時間（本番の負荷の下、torch の演算
   * スレッド 1 本の遅い側の速さ）の約 3 倍。
   */
  public static GrpTrainingPlan of(GrpInit init, long seed) {
    return switch (init) {
      case WARM -> new GrpTrainingPlan(init, BATCH_GAMES, 2e-4, 3, seed, Duration.ofMinutes(44));
      case SCRATCH -> new GrpTrainingPlan(init, BATCH_GAMES, 1e-3, 40, seed, Duration.ofMinutes(680));
    };
  }
}
