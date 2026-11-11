package com.epsilon.grp.train;

import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.SplittableRandom;
import java.util.function.BooleanSupplier;

/**
 * 局の結果のネットワーク q(o|s) を DJL（PyTorch、CPU）で学習する。損失は記録した遷移の周辺尤度で、行ごとに −log Σ_o q(o|s)
 * を求め、コホートの重み（新しいほど重い）で平均する。和を取る o は、記録と同じ点の動きになる結果の全て。学習の進め方は
 * GrpTrainer と同じで、TrainingLoop を使う。ゼロから学習するときの初期値は {@code OutcomeGraph.scratch} が作る。初期値と行の
 * 並べ替えは乱数の種で決まる。
 */
public final class OutcomeTrainer {

  private static final long INIT_SALT = 0x3C6E_F372_FE94_F82BL;
  private static final long SHUFFLE_SALT = 0xA54F_F53A_5F1D_36F1L;

  private OutcomeTrainer() {}

  /**
   * torch の演算スレッドの数を指定して学習する。スレッドの数はプロセス全体に効くので、学習の間だけ替えて、終われば元に戻す。
   *
   * @param threads torch の演算スレッドの数（1 以上）
   */
  public static OutcomeTrained train(
      OutcomeShape shape,
      OutcomeNetwork start,
      OutcomeRows train,
      OutcomeRows stop,
      GrpTrainingRule rule,
      int threads,
      BooleanSupplier closing) {
    return TorchThreads.with(threads, () -> train(shape, start, train, stop, rule, closing));
  }

  /**
   * 学習用の行で学習し、検証用の行の損失で最良の重みと早期終了を決める。
   *
   * @param start 始めのネットワーク。ウォームスタート（WARM）のときだけ渡し、形は学習する形と同じにする。ゼロから学習するときは null
   * @param closing 停止のフラグ。毎ステップの前に見て、立っていれば STOPPED で止める
   */
  public static OutcomeTrained train(
      OutcomeShape shape,
      OutcomeNetwork start,
      OutcomeRows train,
      OutcomeRows stop,
      GrpTrainingRule rule,
      BooleanSupplier closing) {
    rule.requireStart(start == null ? null : start.shape(), shape);
    if (train.variant() != shape.variant() || stop.variant() != shape.variant()) {
      throw new IllegalArgumentException("Rows of " + train.variant() + " and " + stop.variant() + " for " + shape);
    }
    float[] initial = start == null
        ? OutcomeGraph.scratch(shape, new SplittableRandom(SeedMixer.mix64(rule.seed() ^ INIT_SALT)))
        : start.weights();
    int outcomes = Math.max(train.maxOutcomes(), stop.maxOutcomes());
    try (OutcomeGraph graph = OutcomeGraph.of(shape, initial)) {
      TrainingLoop.Result result = TrainingLoop.run(
          "GRP outcomes", graph, new OutcomeBatch(shape, rule.batchRows(), outcomes), train, stop, initial, rule,
          new SplittableRandom(SeedMixer.mix64(rule.seed() ^ SHUFFLE_SALT)), closing);
      OutcomeNetwork network =
          result.ending() == TrainingEnding.STOPPED ? null : OutcomeNetwork.of(shape, result.weights());
      return new OutcomeTrained(network, result.halfEpochs(), result.stopNll(), result.clippedSteps(), result.ending());
    }
  }
}
