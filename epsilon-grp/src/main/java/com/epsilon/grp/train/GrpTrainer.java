package com.epsilon.grp.train;

import com.epsilon.grp.GrpKey;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpPhase;
import com.epsilon.grp.GrpShape;
import com.epsilon.grp.GrpTensor;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.SplittableRandom;
import java.util.function.BooleanSupplier;

/**
 * GRP の MLP を DJL（PyTorch、CPU）で学習する。損失は、行の目標 t との交差エントロピー −Σ_π t(π)·log p(π|s) を、コホートの重み
 * （新しいほど重い。RecencyWindow）で平均したもの。t は実際の順列の one-hot なので、損失は同時 NLL になる。学習の流れ（並べ替え、
 * 半エポックごとの検証用の行の損失、最良の重み、学習率の半減、早期終了、停止のフラグ）は TrainingLoop に任せる。最良の重みは
 * GrpNetwork（GrpTensor の順）にして返す。
 *
 * <p>ゼロから学習するときの初期値は、隠れ層の重みが He（平均 0、分散 2 / 入力の幅）、b1・b2・W3 が 0、b3 が順列ごとの頻度の log。
 * 頻度は、学習用の行の目標に行の重みを掛けて足したもの（重みは平均が 1 になるようにそろえる）。段階ごとに出力ヘッドを持つなら、
 * 段階ごとに数える。数が 0 の順列は 0.5 と数える。そのため、学習を始めたときのネットワークは、順列ごとの頻度どおりの確率を出す。
 * 初期値と並べ替えは乱数の種で決まる。torch の CPU の行列積は足す順がスレッドの数で変わるので、ビット単位で再現するのは、スレッド
 * の数が同じときだけ。
 */
public final class GrpTrainer {

  /** 数が 0 の順列の代わりに数える値。 */
  private static final double ABSENT_COUNT = 0.5;

  private static final long INIT_SALT = 0x6A09_E667_F3BC_C908L;
  private static final long SHUFFLE_SALT = 0xBB67_AE85_84CA_A73BL;

  private GrpTrainer() {}

  /**
   * torch の演算スレッドの数を指定して学習する。スレッドの数はプロセス全体に効くので、学習の間だけ替え、終われば元に戻す。
   *
   * @param threads torch の演算スレッドの数（1 以上）
   */
  public static GrpTrained train(
      GrpShape shape,
      GrpNetwork start,
      GrpRows train,
      GrpRows stop,
      GrpTrainingRule rule,
      int threads,
      BooleanSupplier closing) {
    return TorchThreads.with(threads, () -> train(shape, start, train, stop, rule, closing));
  }

  /**
   * 学習用の行で学習し、検証用の行の損失で最良の重みと早期終了を決める。
   *
   * @param start 始めのネットワーク。WARM のときだけ渡し、形は学習する形と同じ。ゼロから学習するときは null
   * @param closing 停止のフラグ。各ステップの前に見て、立っていれば STOPPED で終える
   */
  public static GrpTrained train(
      GrpShape shape, GrpNetwork start, GrpRows train, GrpRows stop, GrpTrainingRule rule, BooleanSupplier closing) {
    rule.requireStart(start == null ? null : start.shape(), shape);
    if (train.variant() != shape.variant() || stop.variant() != shape.variant()) {
      throw new IllegalArgumentException("Rows of " + train.variant() + " and " + stop.variant() + " for " + shape);
    }
    float[] initial = start == null ? scratch(shape, train, rule.seed()) : start.weights();
    try (GrpGraph graph = GrpGraph.of(shape, initial)) {
      TrainingLoop.Result result = TrainingLoop.run(
          "GRP", graph, new RowBatch(shape, rule.batchRows()), train, stop, initial, rule,
          new SplittableRandom(SeedMixer.mix64(rule.seed() ^ SHUFFLE_SALT)), closing);
      GrpNetwork network = result.ending() == TrainingEnding.STOPPED ? null : GrpNetwork.of(shape, result.weights());
      return new GrpTrained(network, result.halfEpochs(), result.stopNll(), result.clippedSteps(), result.ending());
    }
  }

  /** ゼロから学習するときの初期の重み（GrpTensor の順）。{@code GrpGraph.scratch} で作った重みの b3 に、順列ごとの頻度の log を入れる。 */
  private static float[] scratch(GrpShape shape, GrpRows train, long seed) {
    float[] weights = GrpGraph.scratch(shape, new SplittableRandom(SeedMixer.mix64(seed ^ INIT_SALT)));
    int outputs = shape.outputs();
    double[] counts = new double[shape.heads() * outputs];
    float scale = (float) (train.size() / train.totalWeight());
    for (int row = 0; row < train.size(); row++) {
      int head = shape.phaseHeads() ? GrpPhase.of(shape.variant(), GrpKey.round(train.keys[row])).ordinal() : 0;
      float weight = train.weightOf(row) * scale;
      counts[head * outputs + train.permutations[row]] += weight;
    }
    int bias = GrpTensor.starts(shape)[GrpTensor.B3.ordinal()];
    for (int head = 0; head < shape.heads(); head++) {
      double total = 0;
      for (int k = 0; k < outputs; k++) {
        total += counts[head * outputs + k] > 0 ? counts[head * outputs + k] : ABSENT_COUNT;
      }
      for (int k = 0; k < outputs; k++) {
        double count = counts[head * outputs + k] > 0 ? counts[head * outputs + k] : ABSENT_COUNT;
        weights[bias + head * outputs + k] = (float) Math.log(count / total);
      }
    }
    return weights;
  }
}
