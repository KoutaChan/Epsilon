package com.epsilon.nano.ai.decision.training;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionDataException;
import com.epsilon.nano.ai.model.EpsilonDecisionLoss;

/**
 * Decision 学習器の診断値をページ固定のバッファへ非同期転送し、逆伝播と重ねて回収します。
 *
 * <p>損失の構築と逆伝播は学習処理が所有し、このクラスは学習結果を変更しない観測専用です。価値・BCはサンプル重みの総和、方策モデル固有値は有効な方策モデル 確率質量で集約します。
 */
final class DecisionTrainingMetrics {

  private static final int BASE_METRIC_COUNT = 13;
  private static final int ACTOR_METRIC_COUNT = 15;
  private static final int PACKED_METRIC_COUNT = BASE_METRIC_COUNT + ACTOR_METRIC_COUNT;

  private DecisionTrainingMetrics() {}

  /** 一つの小バッチから読み出したスカラー診断値です。 */
  record Batch(
      float loss,
      float actorLoss,
      float entropyBonusLoss,
      float behaviorCloningLoss,
      float valueLoss,
      float meanRolloutPolicyKl,
      float maximumRolloutPolicyKl,
      float entropy,
      float rolloutEntropy,
      float behaviorToRolloutPolicyKl,
      float chosenProb,
      float behaviorProbMean,
      float behaviorProbMin,
      float actorWeight,
      DecisionTrainingResult.PolicyRatioMetrics policyRatios,
      float effectiveActorRatioMeanSquare) {

    /** 方策検証条件に必要なスカラーをページ固定のバッファへ非同期転送する。 */
    static PendingBatchRead enqueueForPolicyGuards(
        EpsilonDecisionLoss.TrainingLossResult losses, EpsilonDecisionDataParallel.Lane lane) {
      EpsilonDecisionLoss.PolicyRatioDiagnostics ratios = losses.policyRatioDiagnostics();
      NDList metricValues =
          new NDList(
              losses.total(),
              losses.policyGradientLoss(),
              losses.entropyBonusLoss(),
              losses.behaviorCloningLoss(),
              losses.valueLoss(),
              losses.meanRolloutPolicyKl(),
              losses.maximumRolloutPolicyKl(),
              losses.currentEntropy(),
              losses.rolloutEntropy(),
              losses.behaviorToRolloutPolicyKl(),
              losses.chosenCurrentProb(),
              losses.behaviorProbMean(),
              losses.behaviorProbMin());
      metricValues.add(losses.actorWeight());
      addRatioSummary(metricValues, ratios.rawExplorationRatio());
      addRatioSummary(metricValues, ratios.explorationCreditWeight());
      addRatioSummary(metricValues, ratios.policyUpdate());
      addRatioSummary(metricValues, ratios.effectiveActorRatio());
      metricValues.add(ratios.policyUpdateClipFraction());
      metricValues.add(ratios.effectiveActorRatioMeanSquare());

      PendingBatchRead pending = new PendingBatchRead(lane);
      try {
        try (NDArray packed = NDArrays.stack(metricValues)) {
          if (packed.size() != PACKED_METRIC_COUNT) {
            throw new IllegalStateException("Unexpected Decision metric count: " + packed.size());
          }
          lane.enqueueMetricReadback(packed);
        }
        return pending;
      } catch (RuntimeException | Error failure) {
        try {
          pending.close();
        } catch (RuntimeException | Error cleanupFailure) {
          failure.addSuppressed(cleanupFailure);
        }
        throw failure;
      }
    }

    private static BatchRead decode(float[] values) {
      if (values.length != PACKED_METRIC_COUNT) {
        throw new IllegalStateException("Unexpected Decision metric count: " + values.length);
      }

      int index = 0;
      float loss = values[index++];
      float actorLoss = values[index++];
      float entropyBonusLoss = values[index++];
      float behaviorCloningLoss = values[index++];
      float valueLoss = values[index++];
      float meanRolloutPolicyKl = values[index++];
      float maximumRolloutPolicyKl = values[index++];
      float entropy = values[index++];
      float rolloutEntropy = values[index++];
      float behaviorToRolloutPolicyKl = values[index++];
      float chosenProb = values[index++];
      float behaviorProbMean = values[index++];
      float behaviorProbMin = values[index++];

      float actorWeight = values[index++];
      DecisionTrainingResult.RatioMetrics rawExplorationRatio = readRatioMetrics(values, index);
      DecisionTrainingResult.RatioMetrics explorationCreditWeight =
          readRatioMetrics(values, index + 3);
      DecisionTrainingResult.RatioMetrics policyUpdate = readRatioMetrics(values, index + 6);
      DecisionTrainingResult.RatioMetrics effectiveActorRatio = readRatioMetrics(values, index + 9);
      float policyUpdateClipFraction = values[index + 12];
      float effectiveActorRatioMeanSquare = values[index + 13];
      Batch metrics =
          new Batch(
              loss,
              actorLoss,
              entropyBonusLoss,
              behaviorCloningLoss,
              valueLoss,
              meanRolloutPolicyKl,
              maximumRolloutPolicyKl,
              entropy,
              rolloutEntropy,
              behaviorToRolloutPolicyKl,
              chosenProb,
              behaviorProbMean,
              behaviorProbMin,
              actorWeight,
              new DecisionTrainingResult.PolicyRatioMetrics(
                  rawExplorationRatio,
                  explorationCreditWeight,
                  policyUpdate,
                  effectiveActorRatio,
                  policyUpdateClipFraction,
                  effectiveSampleFraction(
                      effectiveActorRatio.mean(), effectiveActorRatioMeanSquare)),
              effectiveActorRatioMeanSquare);
      return new BatchRead(metrics, values);
    }

    private static float effectiveSampleFraction(float mean, float meanSquare) {
      if (meanSquare == 0.0f) {
        return 0.0f;
      }
      return Math.min(1.0f, mean * mean / meanSquare);
    }

    private static void addRatioSummary(NDList values, EpsilonDecisionLoss.RatioSummary summary) {
      values.add(summary.mean());
      values.add(summary.min());
      values.add(summary.max());
    }

    private static DecisionTrainingResult.RatioMetrics readRatioMetrics(float[] values, int index) {
      return new DecisionTrainingResult.RatioMetrics(
          values[index], values[index + 1], values[index + 2]);
    }
  }

  /** 逆伝播と重ねているページ固定の指標転送。解放は未完了転送も安全に回収する。 */
  static final class PendingBatchRead implements AutoCloseable {
    private EpsilonDecisionDataParallel.Lane lane;

    private PendingBatchRead(EpsilonDecisionDataParallel.Lane lane) {
      this.lane = lane;
    }

    BatchRead await() {
      EpsilonDecisionDataParallel.Lane activeLane = requirePending();
      float[] values;
      try {
        values = activeLane.awaitMetricReadback();
      } finally {
        lane = null;
      }
      return Batch.decode(values);
    }

    private EpsilonDecisionDataParallel.Lane requirePending() {
      if (lane == null) {
        throw new IllegalStateException("Decision metric readback is already complete");
      }
      return lane;
    }

    @Override
    public void close() {
      EpsilonDecisionDataParallel.Lane activeLane = lane;
      if (activeLane != null) {
        try {
          activeLane.closeMetricReadback();
        } finally {
          lane = null;
        }
      }
    }
  }

  /** 連結したスカラーを保持し、オプティマイザー確定より前の明示的有限値検査に使います。 */
  record BatchRead(Batch metrics, float[] values) {

    void requireFinite() {
      for (int index = 0; index < values.length; index++) {
        if (!Float.isFinite(values[index])) {
          throw new IllegalStateException(
              "Non-finite Decision metric at index " + index + ": " + values[index]);
        }
      }
      if (metrics.actorWeight() > 0.0f) {
        if (!(metrics.behaviorProbMin() > 0.0f)) {
          throw new IllegalStateException(
              "Decision Actor requires a strictly positive selected behavior probability");
        }
        requirePositive(metrics.policyRatios().rawExplorationRatio(), "rawExplorationRatio");
        requirePositive(
            metrics.policyRatios().explorationCreditWeight(), "explorationCreditWeight");
        requirePositive(metrics.policyRatios().policyUpdate(), "policyUpdate");
        requirePositive(metrics.policyRatios().effectiveActorRatio(), "effectiveActorRatio");
      }
    }

    private static void requirePositive(DecisionTrainingResult.RatioMetrics ratios, String label) {
      if (!(ratios.minimum() > 0.0f)) {
        throw new IllegalStateException(
            "Decision Actor requires strictly positive " + label + ": " + ratios.minimum());
      }
    }
  }

  /** スカラー選択した方策モデルアドバンテージ符号別の件数です。 */
  record AdvantageSigns(long positive, long negative, long zero) {

    static AdvantageSigns from(EpsilonDecisionTrainingBatch batch) {
      long positive = 0L;
      long negative = 0L;
      long zero = 0L;
      for (int sample = 0; sample < batch.size(); sample++) {
        if (!(batch.actorWeight(sample) > 0.0f) || !(batch.sampleWeight(sample) > 0.0f)) {
          continue;
        }
        float scalar = batch.advantage(sample);
        if (!Float.isFinite(scalar)) {
          throw new EpsilonDecisionDataException(
              "scalar advantage must be finite at batch sample=" + sample);
        }
        if (scalar > 0.0f) {
          positive++;
        } else if (scalar < 0.0f) {
          negative++;
        } else {
          zero++;
        }
      }
      return new AdvantageSigns(positive, negative, zero);
    }
  }

  /** 小バッチ診断を学習の反復単位へ重み付き集約します。 */
  static final class Accumulator {
    private int optimizerSteps;
    private int microBatches;
    private long samples;
    private double loss;
    private double lossWeight;
    private double actorLoss;
    private double entropyBonusLoss;
    private double behaviorCloningLoss;
    private double valueLoss;
    private double entropy;
    private double rolloutEntropy;
    private double behaviorToRolloutPolicyKl;
    private double chosenProb;
    private double behaviorProbMean;
    private float behaviorProbMin = Float.POSITIVE_INFINITY;
    private double rolloutPolicyKlWeightedSum;
    private double rolloutPolicyKlActorWeight;
    private float maximumRolloutPolicyKl;
    private double actorWeight;
    private double actorMetricWeight;
    private final PolicyRatioAccumulator policyRatios = new PolicyRatioAccumulator();
    private long positiveScalarAdvantageSamples;
    private long negativeScalarAdvantageSamples;
    private long zeroScalarAdvantageSamples;

    void addAdvantageSigns(AdvantageSigns signs) {
      positiveScalarAdvantageSamples =
          Math.addExact(positiveScalarAdvantageSamples, signs.positive());
      negativeScalarAdvantageSamples =
          Math.addExact(negativeScalarAdvantageSamples, signs.negative());
      zeroScalarAdvantageSamples = Math.addExact(zeroScalarAdvantageSamples, signs.zero());
    }

    void addOptimizerStep() {
      optimizerSteps++;
    }

    void add(int sampleCount, Batch batch) {
      microBatches++;
      samples += sampleCount;
      behaviorCloningLoss += batch.behaviorCloningLoss() * sampleCount;
      valueLoss += batch.valueLoss() * sampleCount;
      double activeActorWeight = (double) batch.actorWeight() * sampleCount;
      actorWeight += activeActorWeight;
      if (activeActorWeight > 0.0) {
        loss += batch.loss() * activeActorWeight;
        lossWeight += activeActorWeight;
        actorLoss += batch.actorLoss() * activeActorWeight;
        entropyBonusLoss += batch.entropyBonusLoss() * activeActorWeight;
        entropy += batch.entropy() * activeActorWeight;
        rolloutEntropy += batch.rolloutEntropy() * activeActorWeight;
        behaviorToRolloutPolicyKl += batch.behaviorToRolloutPolicyKl() * activeActorWeight;
        chosenProb += batch.chosenProb() * activeActorWeight;
        behaviorProbMean += batch.behaviorProbMean() * activeActorWeight;
        actorMetricWeight += activeActorWeight;
        behaviorProbMin = Math.min(behaviorProbMin, batch.behaviorProbMin());
        maximumRolloutPolicyKl = Math.max(maximumRolloutPolicyKl, batch.maximumRolloutPolicyKl());
        rolloutPolicyKlWeightedSum += batch.meanRolloutPolicyKl() * activeActorWeight;
        rolloutPolicyKlActorWeight += activeActorWeight;
        policyRatios.add(
            batch.policyRatios(), batch.effectiveActorRatioMeanSquare(), activeActorWeight);
      }
    }

    void addValue(int sampleCount, Batch batch) {
      microBatches++;
      samples += sampleCount;
      loss += batch.valueLoss() * sampleCount;
      lossWeight += sampleCount;
      valueLoss += batch.valueLoss() * sampleCount;
      behaviorProbMin = 0.0f;
    }

    float meanRolloutPolicyKl() {
      return rolloutPolicyKlActorWeight == 0.0
          ? 0.0f
          : (float) (rolloutPolicyKlWeightedSum / rolloutPolicyKlActorWeight);
    }

    float maximumRolloutPolicyKl() {
      return maximumRolloutPolicyKl;
    }

    DecisionTrainingResult toMetrics(
        EpsilonDecisionTrainingRejection rejection,
        DecisionTrainingPerformanceSnapshot performance) {
      if (samples == 0L && !rejection.rejected()) {
        throw new IllegalStateException("accepted training result contains no samples");
      }
      return new DecisionTrainingResult(
          optimizerSteps,
          microBatches,
          mean(loss, lossWeight),
          mean(actorLoss, actorMetricWeight),
          mean(entropyBonusLoss, actorMetricWeight),
          mean(behaviorCloningLoss, samples),
          mean(valueLoss, samples),
          mean(entropy, actorMetricWeight),
          mean(rolloutEntropy, actorMetricWeight),
          mean(behaviorToRolloutPolicyKl, actorMetricWeight),
          mean(chosenProb, actorMetricWeight),
          mean(behaviorProbMean, actorMetricWeight),
          actorMetricWeight == 0.0 ? 0.0f : behaviorProbMin,
          meanRolloutPolicyKl(),
          maximumRolloutPolicyKl,
          mean(actorWeight, samples),
          policyRatios.toMetrics(),
          positiveScalarAdvantageSamples,
          negativeScalarAdvantageSamples,
          zeroScalarAdvantageSamples,
          rejection,
          performance);
    }

    private static float mean(double sum, double weight) {
      return weight == 0.0 ? 0.0f : (float) (sum / weight);
    }
  }

  private static final class PolicyRatioAccumulator {
    private final RatioAccumulator rawExplorationRatio = new RatioAccumulator();
    private final RatioAccumulator explorationCreditWeight = new RatioAccumulator();
    private final RatioAccumulator policyUpdate = new RatioAccumulator();
    private final RatioAccumulator effectiveActorRatio = new RatioAccumulator();
    private double policyUpdateClipFractionWeightedSum;
    private double effectiveActorRatioSquareWeightedSum;
    private double weight;

    private void add(
        DecisionTrainingResult.PolicyRatioMetrics metrics,
        float effectiveActorRatioMeanSquare,
        double activeActorWeight) {
      rawExplorationRatio.add(metrics.rawExplorationRatio(), activeActorWeight);
      explorationCreditWeight.add(metrics.explorationCreditWeight(), activeActorWeight);
      policyUpdate.add(metrics.policyUpdate(), activeActorWeight);
      effectiveActorRatio.add(metrics.effectiveActorRatio(), activeActorWeight);
      policyUpdateClipFractionWeightedSum += metrics.policyUpdateClipFraction() * activeActorWeight;
      effectiveActorRatioSquareWeightedSum += effectiveActorRatioMeanSquare * activeActorWeight;
      weight += activeActorWeight;
    }

    private DecisionTrainingResult.PolicyRatioMetrics toMetrics() {
      DecisionTrainingResult.RatioMetrics aggregatedEffectiveActorRatio =
          effectiveActorRatio.toMetrics();
      return new DecisionTrainingResult.PolicyRatioMetrics(
          rawExplorationRatio.toMetrics(),
          explorationCreditWeight.toMetrics(),
          policyUpdate.toMetrics(),
          aggregatedEffectiveActorRatio,
          weight == 0.0 ? 0.0f : (float) (policyUpdateClipFractionWeightedSum / weight),
          effectiveSampleFraction(aggregatedEffectiveActorRatio.mean()));
    }

    private float effectiveSampleFraction(float mean) {
      if (weight == 0.0 || effectiveActorRatioSquareWeightedSum == 0.0) {
        return 0.0f;
      }
      double meanSquare = effectiveActorRatioSquareWeightedSum / weight;
      return (float) Math.min(1.0, (double) mean * mean / meanSquare);
    }
  }

  private static final class RatioAccumulator {
    private double weightedSum;
    private double weight;
    private float minimum = Float.POSITIVE_INFINITY;
    private float maximum;

    private void add(DecisionTrainingResult.RatioMetrics metrics, double activeActorWeight) {
      weightedSum += metrics.mean() * activeActorWeight;
      weight += activeActorWeight;
      minimum = Math.min(minimum, metrics.minimum());
      maximum = Math.max(maximum, metrics.maximum());
    }

    private DecisionTrainingResult.RatioMetrics toMetrics() {
      return new DecisionTrainingResult.RatioMetrics(
          weight == 0.0 ? 0.0f : (float) (weightedSum / weight),
          weight == 0.0 ? 0.0f : minimum,
          maximum);
    }
  }
}
