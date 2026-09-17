package com.epsilon.ai.belief;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.PairList;
import com.epsilon.core.Tile;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 公開情報から他家状態を予測するBeliefモデルの教師あり学習を行う。
 *
 * <ul>
 *   <li>手牌: 未確認の牌の残数を反映した多項分布の交差エントロピー。
 *   <li>受け入れ牌: 牌種ごとの二値交差エントロピー。テンパイ前も学習対象とし、テンパイ時は待ち牌を学習する。
 *   <li>シャンテン数: Huber損失。テンパイの有無: 二値交差エントロピー。
 * </ul>
 */
public final class EpsilonBeliefTrainer<I> implements AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(EpsilonBeliefTrainer.class);

  private static final float HAND_LOSS_WEIGHT = 1.0f;
  private static final float WAIT_LOSS_WEIGHT = 0.5f;
  private static final float SCALAR_LOSS_WEIGHT = 0.25f;

  /** 残数 0 の牌種を softmax から実質除外するための対数マスク下限。 */
  private static final float LOG_MASK_FLOOR = -30.0f;

  private final NDManager manager;
  private final Trainer trainer;
  private final int maxBatchSize;
  private final Block network;
  private final BeliefBatchFactory<I> inputs;

  /** 系列の入力転送とオプティマイザー設定を受け取り、Beliefの学習資源を構築する。 */
  public EpsilonBeliefTrainer(
      Model model,
      BeliefBatchFactory<I> inputs,
      int batchSize,
      float learningRate,
      float weightDecay,
      float gradClip) {
    this.maxBatchSize = batchSize;
    this.manager = model.getNDManager();
    this.network = model.getBlock();
    this.inputs = inputs;
    this.trainer = createTrainer(model, learningRate, weightDecay, gradClip);
  }

  /**
   * 指定したエポック数だけサンプル群を繰り返し学習する。
   *
   * @param samples 公開局面と完全情報教師データの組
   * @param epochs サンプル群を反復する正の回数
   * @return 全バッチの損失を等重みで平均した指標
   */
  public TrainMetrics train(List<EpsilonBeliefSample<I>> samples, int epochs) {
    if (epochs <= 0) {
      throw new IllegalArgumentException("epochs must be positive");
    }
    if (samples.isEmpty()) {
      return TrainMetrics.empty();
    }
    Accumulator acc = new Accumulator();
    int batchSize = batchSize(samples.size());
    for (int epoch = 0; epoch < epochs; epoch++) {
      for (int start = 0; start < samples.size(); start += batchSize) {
        int end = Math.min(samples.size(), start + batchSize);
        acc.add(trainBatch(samples.subList(start, end), true));
      }
    }
    TrainMetrics metrics = acc.toMetrics();
    log.info(
        "Belief train update: batches={} samples={} loss={} handLoss={} waitLoss={} scalarLoss={}",
        metrics.batches(),
        metrics.samples(),
        metrics.loss(),
        metrics.handLoss(),
        metrics.waitLoss(),
        metrics.scalarLoss());
    return metrics;
  }

  /**
   * パラメーターを更新せずサンプル群の損失を評価する。
   *
   * @param samples 公開局面と完全情報教師データの組
   * @return 全バッチの損失を等重みで平均した指標
   */
  public TrainMetrics evaluate(List<EpsilonBeliefSample<I>> samples) {
    if (samples.isEmpty()) {
      return TrainMetrics.empty();
    }
    Accumulator acc = new Accumulator();
    int batchSize = batchSize(samples.size());
    for (int start = 0; start < samples.size(); start += batchSize) {
      int end = Math.min(samples.size(), start + batchSize);
      acc.add(trainBatch(samples.subList(start, end), false));
    }
    return acc.toMetrics();
  }

  private int batchSize(int sampleCount) {
    return Math.min(maxBatchSize, sampleCount);
  }

  private BatchMetrics trainBatch(List<EpsilonBeliefSample<I>> samples, boolean update) {
    try (NDManager sub = manager.newSubManager()) {
      NDList batchInputs =
          inputs.transfer(sub, samples.stream().map(EpsilonBeliefSample::input).toList());
      NDArray target =
          sub.create(
              targetData(samples), new Shape(samples.size(), EpsilonBeliefLayout.OUTPUT_SIZE));
      NDArray hiddenTiles =
          sub.create(hiddenTileData(samples), new Shape(samples.size(), Tile.NUM_TILE_TYPES));
      SectionLosses losses;
      if (update) {
        try (GradientCollector gc = trainer.newGradientCollector()) {
          NDArray prediction = forward(batchInputs, true);
          losses = sectionLosses(prediction, target, hiddenTiles);
          finite(losses.totalValue(), "loss");
          gc.backward(losses.total());
        }
        trainer.step();
      } else {
        NDArray prediction = forward(batchInputs, false);
        losses = sectionLosses(prediction, target, hiddenTiles);
        finite(losses.totalValue(), "loss");
      }
      return new BatchMetrics(
          samples.size(),
          losses.totalValue(),
          losses.handValue(),
          losses.waitValue(),
          losses.scalarValue());
    }
  }

  private SectionLosses sectionLosses(NDArray prediction, NDArray target, NDArray hiddenTiles) {
    long batch = prediction.getShape().get(0);
    int opponents = EpsilonBeliefLayout.OPPONENT_COUNT;
    int tiles = Tile.NUM_TILE_TYPES;
    int handOffset = 0;
    int waitOffset = handOffset + EpsilonBeliefLayout.OPPONENT_HAND_SIZE;
    int scalarOffset = waitOffset + EpsilonBeliefLayout.OPPONENT_WAIT_SIZE;
    int outputSize = EpsilonBeliefLayout.OUTPUT_SIZE;
    int scalarCount = EpsilonBeliefLayout.OPPONENT_SCALAR_COUNT;

    NDArray predictedHands =
        prediction.get(":, {}:{}", handOffset, waitOffset).reshape(batch, opponents, tiles);
    NDArray predictedWaits =
        prediction.get(":, {}:{}", waitOffset, scalarOffset).reshape(batch, opponents, tiles);
    NDArray predictedScalars =
        prediction.get(":, {}:{}", scalarOffset, outputSize).reshape(batch, opponents, scalarCount);

    NDArray targetHands =
        target.get(":, {}:{}", handOffset, waitOffset).reshape(batch, opponents, tiles);
    NDArray targetWaits =
        target.get(":, {}:{}", waitOffset, scalarOffset).reshape(batch, opponents, tiles);
    NDArray targetScalars =
        target.get(":, {}:{}", scalarOffset, outputSize).reshape(batch, opponents, scalarCount);

    // 観測者から見えない牌の残数。残数 0 の牌種は softmax から除外する。
    NDArray logMask = hiddenTiles.add(1.0e-12f).log().maximum(LOG_MASK_FLOOR);

    NDArray handLoss = multinomialCe(predictedHands.add(logMask.expandDims(1)), targetHands, 2);
    NDArray waitLoss = bceWithLogits(predictedWaits, targetWaits).mean();
    NDArray scalarLoss = scalarLoss(predictedScalars, targetScalars);

    NDArray total =
        handLoss
            .mul(HAND_LOSS_WEIGHT)
            .add(waitLoss.mul(WAIT_LOSS_WEIGHT))
            .add(scalarLoss.mul(SCALAR_LOSS_WEIGHT));
    return new SectionLosses(
        total, total.getFloat(), handLoss.getFloat(), waitLoss.getFloat(), scalarLoss.getFloat());
  }

  private static NDArray scalarLoss(NDArray predictedScalars, NDArray targetScalars) {
    NDArray predictedShanten = predictedScalars.get(":, :, {}", EpsilonBeliefLayout.SCALAR_SHANTEN);
    NDArray targetShanten = targetScalars.get(":, :, {}", EpsilonBeliefLayout.SCALAR_SHANTEN);
    NDArray predictedTenpai = predictedScalars.get(":, :, {}", EpsilonBeliefLayout.SCALAR_TENPAI);
    NDArray targetTenpai = targetScalars.get(":, :, {}", EpsilonBeliefLayout.SCALAR_TENPAI);
    return huberPerElement(predictedShanten, targetShanten, 1.0f)
        .mean()
        .add(bceWithLogits(predictedTenpai, targetTenpai).mean());
  }

  /** 牌種ごとの枚数を合計1に正規化し、交差エントロピーを求める。ロジットにはあらかじめ牌の残数の対数を加える。 */
  private static NDArray multinomialCe(NDArray maskedLogits, NDArray counts, int tileAxis) {
    NDArray logProb = maskedLogits.logSoftmax(tileAxis);
    NDArray total = counts.sum(new int[] {tileAxis}, true).maximum(1.0f);
    return counts.div(total).mul(logProb).sum(new int[] {tileAxis}).neg().mean();
  }

  /** 数値的に安定した式で、ロジットから二値交差エントロピーを計算する。 */
  private static NDArray bceWithLogits(NDArray logits, NDArray labels) {
    return logits
        .maximum(0.0f)
        .sub(logits.mul(labels))
        .add(logits.abs().neg().exp().add(1.0f).log());
  }

  private static NDArray huberPerElement(NDArray pred, NDArray target, float delta) {
    NDArray diff = pred.sub(target).abs();
    NDArray quadratic = diff.minimum(delta);
    NDArray linear = diff.sub(delta).maximum(0.0f);
    return quadratic.square().add(linear.mul(2.0f * delta)).mul(0.5f);
  }

  private NDArray forward(NDList inputs, boolean training) {
    return network
        .forward(
            new ParameterStore(inputs.getFirst().getManager(), training),
            inputs,
            training,
            new PairList<>())
        .singletonOrThrow();
  }

  private static <I> float[] targetData(List<EpsilonBeliefSample<I>> samples) {
    int width = EpsilonBeliefLayout.OUTPUT_SIZE;
    float[] out = new float[samples.size() * width];
    for (int i = 0; i < samples.size(); i++) {
      samples.get(i).target().writeTo(out, i * width);
    }
    return out;
  }

  private static <I> float[] hiddenTileData(List<EpsilonBeliefSample<I>> samples) {
    int width = Tile.NUM_TILE_TYPES;
    float[] out = new float[samples.size() * width];
    for (int i = 0; i < samples.size(); i++) {
      System.arraycopy(samples.get(i).target().hiddenTileCounts(), 0, out, i * width, width);
    }
    return out;
  }

  private static Trainer createTrainer(
      Model model, float learningRate, float weightDecay, float gradClip) {
    DefaultTrainingConfig config =
        new DefaultTrainingConfig(Loss.l2Loss())
            .optDevices(new Device[] {model.getNDManager().getDevice()})
            .optOptimizer(
                Optimizer.adamW()
                    .optLearningRateTracker(Tracker.fixed(learningRate))
                    .optWeightDecays(weightDecay)
                    .optClipGrad(gradClip)
                    .build());
    return model.newTrainer(config);
  }

  private static float finite(float value, String label) {
    if (!Float.isFinite(value)) {
      throw new IllegalStateException("Non-finite Belief " + label + ": " + value);
    }
    return value;
  }

  /** 所有するDJLの学習器とオプティマイザー資源を解放する。 */
  @Override
  public void close() {
    trainer.close();
  }

  private record SectionLosses(
      NDArray total, float totalValue, float handValue, float waitValue, float scalarValue) {}

  private record BatchMetrics(
      int samples, float loss, float handLoss, float waitLoss, float scalarLoss) {}

  private static final class Accumulator {
    private int samples;
    private int batches;
    private float loss;
    private float handLoss;
    private float waitLoss;
    private float scalarLoss;

    void add(BatchMetrics batch) {
      samples += batch.samples;
      batches++;
      loss += batch.loss;
      handLoss += batch.handLoss;
      waitLoss += batch.waitLoss;
      scalarLoss += batch.scalarLoss;
    }

    TrainMetrics toMetrics() {
      return new TrainMetrics(
          samples,
          batches,
          loss / batches,
          handLoss / batches,
          waitLoss / batches,
          scalarLoss / batches);
    }
  }

  /**
   * 1回の学習または評価で集計した指標。損失はバッチごとの値を等しい重みで平均する。
   *
   * @param samples 学習サンプル数
   * @param batches 学習または評価したバッチ数
   * @param loss 合計損失
   * @param handLoss 他家手牌分布損失
   * @param waitLoss 全他家の受け入れ牌に対する二値交差エントロピー。テンパイ時は待ち牌の損失
   * @param scalarLoss シャンテン数のHuber損失とテンパイ判定の二値交差エントロピーの合計
   */
  public record TrainMetrics(
      int samples, int batches, float loss, float handLoss, float waitLoss, float scalarLoss) {

    /**
     * サンプルがない場合のゼロ指標を返す。
     *
     * @return サンプル数、バッチ数、損失がすべて0の指標
     */
    public static TrainMetrics empty() {
      return new TrainMetrics(0, 0, 0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
}
