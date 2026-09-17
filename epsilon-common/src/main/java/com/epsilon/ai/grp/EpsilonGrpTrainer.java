package com.epsilon.ai.grp;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.PairList;
import com.epsilon.core.GameState;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 各席が各順位になる確率を予測する GRP モデルを、単体で学習する。 */
public final class EpsilonGrpTrainer implements AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(EpsilonGrpTrainer.class);

  private final EpsilonGrpNetwork network;
  private final int batchSize;
  private final NDManager manager;
  private final Trainer trainer;
  private final Random shuffleRandom = new Random(0x6E9B5EEDL);

  /**
   * GRP モデル専用オプティマイザーと学習器を構築する。
   *
   * @param model {@link EpsilonGrpNetwork} をブロックに持つ学習可能なモデル
   */
  public EpsilonGrpTrainer(
      Model model, int batchSize, float learningRate, float weightDecay, float gradClip) {
    this.batchSize = batchSize;
    if (!(model.getBlock() instanceof EpsilonGrpNetwork grpNetwork)) {
      throw new IllegalArgumentException(
          "Model block is not EpsilonGrpNetwork: " + model.getBlock());
    }
    this.network = grpNetwork;
    this.manager = model.getNDManager();
    this.trainer = createTrainer(model, learningRate, weightDecay, gradClip);
  }

  /**
   * GRP サンプルをエポックごとにシャッフルして学習する。
   *
   * @param examples 一意化済み GRP サンプル
   * @param epochs 全サンプルを反復する回数
   * @return オプティマイザーバッチ全体の集約指標
   */
  public TrainMetrics trainExamples(List<EpsilonGrpExample> examples, int epochs) {
    if (examples.isEmpty()) {
      return TrainMetrics.empty();
    }
    int batchSize = Math.min(this.batchSize, examples.size());
    int batches = 0;
    double marginalNllSum = 0.0;
    int rankMatches = 0;
    int totalExamples = 0;
    ArrayList<EpsilonGrpExample> shuffled = new ArrayList<>(examples);
    for (int epoch = 0; epoch < epochs; epoch++) {
      // 同じ対局のサンプルが連続してバッチを占有しないようエポック毎にシャッフルする。
      Collections.shuffle(shuffled, shuffleRandom);
      for (int start = 0; start < shuffled.size(); start += batchSize) {
        int count = Math.min(batchSize, shuffled.size() - start);
        BatchMetrics metrics = trainBatch(EpsilonGrpBatch.fromExamples(shuffled, start, count));
        batches++;
        marginalNllSum += metrics.marginalNll() * metrics.examples();
        rankMatches += metrics.rankMatches();
        totalExamples += metrics.examples();
      }
    }
    TrainMetrics metrics =
        new TrainMetrics(
            totalExamples,
            batches,
            (float) (marginalNllSum / totalExamples),
            rankMatches / (float) (totalExamples * GameState.NUM_PLAYERS));
    log.info(
        "GRP train update: examples={} batches={} marginalNll={} rankAccuracy={}",
        metrics.examples,
        metrics.batches,
        metrics.marginalNll,
        metrics.rankAccuracy);
    return metrics;
  }

  private BatchMetrics trainBatch(EpsilonGrpBatch batch) {
    try (NDManager sub = manager.newSubManager()) {
      NDArray labels = batch.labelArray(sub);
      NDArray logits;
      NDArray marginalNll;
      try (GradientCollector gc = trainer.newGradientCollector()) {
        logits =
            network.forwardLogits(
                new ParameterStore(sub, false),
                batch.sequenceArray(sub),
                batch.lengthArray(sub),
                true,
                new PairList<>());
        marginalNll = crossEntropy(logits, labels);
        gc.backward(marginalNll);
      }
      trainer.step();
      float[] marginals = EpsilonGrpSinkhorn.marginals(logits).toFloatArray();
      return batchMetrics(finite(marginalNll.getFloat(), "marginal NLL"), marginals, batch);
    }
  }

  private static NDArray crossEntropy(NDArray logits, NDArray oneHotLabels) {
    long batch = logits.getShape().get(0);
    NDArray logMarginals = EpsilonGrpSinkhorn.logMarginals(logits);
    return oneHotLabels
        .mul(logMarginals)
        .reshape(batch, EpsilonGrpRanks.MATRIX_SIZE)
        .sum(new int[] {1})
        .neg()
        .div(EpsilonGrpRanks.SEAT_COUNT)
        .mean();
  }

  private static BatchMetrics batchMetrics(
      float marginalNll, float[] rankMarginals, EpsilonGrpBatch batch) {
    int rankMatches = 0;
    for (int i = 0; i < batch.size; i++) {
      int finalRanksCode = batch.finalRanksCodes[i];
      if (EpsilonGrpRanks.isValidCode(finalRanksCode)) {
        int[] trueRanks = EpsilonGrpRanks.decode(finalRanksCode);
        for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
          int predictedRank = 0;
          int offset = i * EpsilonGrpRanks.MATRIX_SIZE + player * EpsilonGrpRanks.RANK_COUNT;
          for (int rank = 1; rank < EpsilonGrpRanks.RANK_COUNT; rank++) {
            if (rankMarginals[offset + rank] > rankMarginals[offset + predictedRank]) {
              predictedRank = rank;
            }
          }
          if (predictedRank == trueRanks[player]) {
            rankMatches++;
          }
        }
      }
    }
    return new BatchMetrics(marginalNll, rankMatches, batch.size);
  }

  private static Trainer createTrainer(
      Model model, float learningRate, float weightDecay, float gradClip) {
    // 損失は trainBatch で手動計算するため、ここの Loss は Trainer 生成に必要なだけの
    // プレースホルダーで学習には使われない。
    DefaultTrainingConfig config =
        new DefaultTrainingConfig(Loss.l2Loss())
            .optDevices(new Device[] {model.getNDManager().getDevice()})
            .optOptimizer(
                new GrpOptimizer(
                    model.getBlock().getParameters(),
                    Optimizer.adamW()
                        .optLearningRateTracker(Tracker.fixed(learningRate))
                        .optWeightDecays(weightDecay)
                        .optClipGrad(gradClip)
                        .build()));
    return model.newTrainer(config);
  }

  public void saveOptimizerState(Path file) throws IOException {
    trainer.saveOptimizerState(file);
  }

  public void loadOptimizerState(Path file) throws IOException {
    trainer.loadOptimizerState(file);
  }

  private static float finite(float value, String label) {
    if (!Float.isFinite(value)) {
      throw new IllegalStateException("Non-finite GRP " + label + ": " + value);
    }
    return value;
  }

  @Override
  public void close() {
    trainer.close();
  }

  private record BatchMetrics(float marginalNll, int rankMatches, int examples) {}

  /**
   * 一回の GRP 学習呼び出しで集約した指標。
   *
   * @param examples 学習例数
   * @param batches オプティマイザーバッチ数
   * @param marginalNll 席別の順位周辺確率の平均 NLL
   * @param rankAccuracy 予測確率が最も高い順位の正解率
   */
  public record TrainMetrics(int examples, int batches, float marginalNll, float rankAccuracy) {

    /**
     * 更新を行わなかった学習結果を返す。
     *
     * @return 全カウンターと指標が 0 の結果
     */
    public static TrainMetrics empty() {
      return new TrainMetrics(0, 0, 0.0f, 0.0f);
    }
  }
}
