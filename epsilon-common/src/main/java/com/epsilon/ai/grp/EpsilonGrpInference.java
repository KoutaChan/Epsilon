package com.epsilon.ai.grp;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/** GRU を使う GRP モデルのバッチ推論。各席が各順位になる確率を4行4列の行列として求める。 */
public final class EpsilonGrpInference implements EpsilonGrpRankPredictor {

  private static final int CALIBRATION_BINS = 15;
  private static final float INFERENCE_MARGINAL_SUM_TOLERANCE = 5.0e-5f;

  private final Model model;
  private final EpsilonGrpNetwork network;
  private final int maxBatch;
  private final boolean cpuSinkhorn;

  /**
   * 読み込み済み GRP モデルに同期推論 API を構築する。
   *
   * @param model {@link EpsilonGrpNetwork} をブロックに持つモデル
   * @throws IllegalArgumentException モデルブロックが GRP ネットワークでない場合
   */
  public EpsilonGrpInference(Model model) {
    this(model, 64);
  }

  public EpsilonGrpInference(Model model, int maxBatch) {
    if (!(model.getBlock() instanceof EpsilonGrpNetwork grpNetwork)) {
      throw new IllegalArgumentException(
          "Model block is not EpsilonGrpNetwork: " + model.getBlock());
    }
    this.model = model;
    this.network = grpNetwork;
    this.maxBatch = maxBatch;
    this.cpuSinkhorn = !model.getNDManager().getDevice().isGpu();
  }

  /**
   * 一つの局開始時点までの GRP 特徴量系列から指定席の4通りの最終順位の周辺確率を返す。
   *
   * @param sequence 一次元化済み GRP 特徴量系列
   * @param seat 予測対象の席
   * @return 順位 1～4 の確率。系列が空なら {@code null}
   */
  public float[] predictRankProbabilities(float[] sequence, int seat) {
    if (sequence == null || EpsilonGrpFeature.steps(sequence) <= 0) {
      return null;
    }
    if (seat < 0 || seat >= EpsilonGrpRanks.SEAT_COUNT) {
      throw new IllegalArgumentException("seat must be 0-3: " + seat);
    }
    float[] marginals = predictMarginalProbabilities(List.of(sequence)).getFirst();
    float[] seatMarginal = new float[EpsilonGrpRanks.RANK_COUNT];
    System.arraycopy(
        marginals, seat * EpsilonGrpRanks.RANK_COUNT, seatMarginal, 0, EpsilonGrpRanks.RANK_COUNT);
    return seatMarginal;
  }

  @Override
  public CompletableFuture<float[]> predictRankProbabilitiesAsync(float[] sequence, int seat) {
    try {
      return CompletableFuture.completedFuture(predictRankProbabilities(sequence, seat));
    } catch (RuntimeException | Error e) {
      return CompletableFuture.failedFuture(e);
    }
  }

  @Override
  public CompletableFuture<float[]> predictMarginalProbabilitiesAsync(float[] sequence) {
    if (sequence == null || EpsilonGrpFeature.steps(sequence) <= 0) {
      return CompletableFuture.completedFuture(null);
    }
    try {
      return CompletableFuture.completedFuture(
          predictMarginalProbabilities(List.of(sequence)).getFirst());
    } catch (RuntimeException | Error e) {
      return CompletableFuture.failedFuture(e);
    }
  }

  /**
   * 検証用データ例に対する4席周辺確率 NLL、Brier、順位付き確率スコア（RPS）、較正、順位正解率を求める。
   *
   * @param examples 評価する GRP サンプル
   * @return サンプル全体を集約した評価指標
   */
  public EvalMetrics evaluateExamples(List<EpsilonGrpExample> examples) {
    if (examples.isEmpty()) {
      return EvalMetrics.empty();
    }
    ArrayList<float[]> sequences = new ArrayList<>(examples.size());
    for (EpsilonGrpExample example : examples) {
      sequences.add(example.sequence());
    }
    List<float[]> probabilities = predictMarginalProbabilities(sequences);
    double marginalNll = 0.0;
    double marginalBrier = 0.0;
    double ordinalRps = 0.0;
    double[][] confidenceSums = new double[4][CALIBRATION_BINS];
    double[][] outcomeSums = new double[4][CALIBRATION_BINS];
    int[][] calibrationCounts = new int[4][CALIBRATION_BINS];
    int rankMatches = 0;
    float maxRowSumError = 0.0f;
    float maxColumnSumError = 0.0f;
    for (int i = 0; i < examples.size(); i++) {
      float[] marginals = probabilities.get(i);
      int[] trueRanks = EpsilonGrpRanks.decode(examples.get(i).finalRanksCode());
      for (int seat = 0; seat < EpsilonGrpRanks.SEAT_COUNT; seat++) {
        float rowSum = 0.0f;
        for (int rank = 0; rank < EpsilonGrpRanks.RANK_COUNT; rank++) {
          float probability = marginals[EpsilonGrpRanks.index(seat, rank)];
          rowSum += probability;
        }
        maxRowSumError = Math.max(maxRowSumError, Math.abs(rowSum - 1.0f));
        int trueRank = trueRanks[seat];
        int predictedRank = 0;
        for (int rank = 1; rank < EpsilonGrpRanks.RANK_COUNT; rank++) {
          if (marginals[EpsilonGrpRanks.index(seat, rank)]
              > marginals[EpsilonGrpRanks.index(seat, predictedRank)]) {
            predictedRank = rank;
          }
        }
        if (predictedRank == trueRank) {
          rankMatches++;
        }
        float trueRankProbability = marginals[EpsilonGrpRanks.index(seat, trueRank)];
        marginalNll += -Math.log(trueRankProbability);
        float predictedCumulative = 0.0f;
        for (int rank = 0; rank < EpsilonGrpRanks.RANK_COUNT; rank++) {
          float confidence = marginals[EpsilonGrpRanks.index(seat, rank)];
          float outcome = rank == trueRank ? 1.0f : 0.0f;
          double error = confidence - outcome;
          marginalBrier += error * error;
          int bin = Math.min(CALIBRATION_BINS - 1, (int) (confidence * CALIBRATION_BINS));
          confidenceSums[rank][bin] += confidence;
          outcomeSums[rank][bin] += outcome;
          calibrationCounts[rank][bin]++;
          predictedCumulative += confidence;
          if (rank < EpsilonGrpRanks.RANK_COUNT - 1) {
            float targetCumulative = trueRank <= rank ? 1.0f : 0.0f;
            double cumulativeError = predictedCumulative - targetCumulative;
            ordinalRps += cumulativeError * cumulativeError;
          }
        }
      }
      for (int rank = 0; rank < EpsilonGrpRanks.RANK_COUNT; rank++) {
        float columnSum = 0.0f;
        for (int seat = 0; seat < EpsilonGrpRanks.SEAT_COUNT; seat++) {
          columnSum += marginals[EpsilonGrpRanks.index(seat, rank)];
        }
        maxColumnSumError = Math.max(maxColumnSumError, Math.abs(columnSum - 1.0f));
      }
    }
    int count = examples.size();
    float[] rankEce = new float[EpsilonGrpRanks.RANK_COUNT];
    float macroEce = 0.0f;
    for (int rank = 0; rank < rankEce.length; rank++) {
      rankEce[rank] =
          expectedCalibrationError(
              confidenceSums[rank], outcomeSums[rank], calibrationCounts[rank]);
      macroEce += rankEce[rank];
    }
    macroEce /= rankEce.length;
    int seatTargets = count * EpsilonGrpRanks.SEAT_COUNT;
    return new EvalMetrics(
        count,
        (float) (marginalNll / seatTargets),
        (float) (marginalBrier / seatTargets),
        (float) (ordinalRps / (seatTargets * (EpsilonGrpRanks.RANK_COUNT - 1))),
        macroEce,
        rankEce[3],
        rankMatches / (float) seatTargets,
        0,
        maxRowSumError,
        maxColumnSumError);
  }

  private static float expectedCalibrationError(
      double[] confidenceSums, double[] outcomeSums, int[] counts) {
    int total = 0;
    for (int count : counts) {
      total += count;
    }
    double ece = 0.0;
    for (int bin = 0; bin < counts.length; bin++) {
      int count = counts[bin];
      if (count == 0) {
        continue;
      }
      double meanConfidence = confidenceSums[bin] / count;
      double meanOutcome = outcomeSums[bin] / count;
      ece += count / (double) total * Math.abs(meanConfidence - meanOutcome);
    }
    return (float) ece;
  }

  /**
   * 各席が各順位になる確率を、入力系列ごとに返す。各結果は行和と列和が1になる4行4列の行列を、席ごとに連結した16要素の配列である。
   *
   * @param sequences 一次元化済み GRP 特徴量系列
   * @return 入力と同順序の席別の順位周辺確率
   */
  public synchronized List<float[]> predictMarginalProbabilities(List<float[]> sequences) {
    if (sequences.isEmpty()) {
      return List.of();
    }
    ArrayList<float[]> out = new ArrayList<>(sequences.size());
    for (int start = 0; start < sequences.size(); start += maxBatch) {
      int end = Math.min(sequences.size(), start + maxBatch);
      out.addAll(predictChunk(sequences.subList(start, end)));
    }
    return out;
  }

  private List<float[]> predictChunk(List<float[]> sequences) {
    EpsilonGrpBatch batch = EpsilonGrpBatch.fromSequences(sequences);
    try (NDManager sub = model.getNDManager().newSubManager()) {
      ParameterStore parameterStore = new ParameterStore(sub, false);
      NDArray logits =
          network.forwardLogits(
              parameterStore,
              batch.sequenceArray(sub),
              batch.lengthArray(sub),
              false,
              new PairList<>());
      float[] probabilities =
          cpuSinkhorn
              ? EpsilonGrpSinkhorn.marginalsBatch(logits.toFloatArray())
              : EpsilonGrpSinkhorn.marginals(logits).toFloatArray();
      requireValidMarginals(probabilities, batch.size);
      ArrayList<float[]> out = new ArrayList<>(batch.size);
      for (int i = 0; i < batch.size; i++) {
        float[] marginals = new float[EpsilonGrpRanks.MATRIX_SIZE];
        System.arraycopy(
            probabilities,
            i * EpsilonGrpRanks.MATRIX_SIZE,
            marginals,
            0,
            EpsilonGrpRanks.MATRIX_SIZE);
        out.add(marginals);
      }
      return out;
    }
  }

  private static void requireValidMarginals(float[] probabilities, int batchSize) {
    for (int batch = 0; batch < batchSize; batch++) {
      int base = batch * EpsilonGrpRanks.MATRIX_SIZE;
      for (int seat = 0; seat < EpsilonGrpRanks.SEAT_COUNT; seat++) {
        float rowSum = 0.0f;
        for (int rank = 0; rank < EpsilonGrpRanks.RANK_COUNT; rank++) {
          float probability = probabilities[base + EpsilonGrpRanks.index(seat, rank)];
          if (!Float.isFinite(probability) || probability <= 0.0f || probability > 1.0f) {
            throw new IllegalStateException(
                "GRP inference produced invalid marginal probability: " + probability);
          }
          rowSum += probability;
        }
        if (Math.abs(rowSum - 1.0f) > INFERENCE_MARGINAL_SUM_TOLERANCE) {
          throw new IllegalStateException("GRP inference row marginal sum is " + rowSum);
        }
      }
      for (int rank = 0; rank < EpsilonGrpRanks.RANK_COUNT; rank++) {
        float columnSum = 0.0f;
        for (int seat = 0; seat < EpsilonGrpRanks.SEAT_COUNT; seat++) {
          columnSum += probabilities[base + EpsilonGrpRanks.index(seat, rank)];
        }
        if (Math.abs(columnSum - 1.0f) > INFERENCE_MARGINAL_SUM_TOLERANCE) {
          throw new IllegalStateException("GRP inference column marginal sum is " + columnSum);
        }
      }
    }
  }

  /**
   * GRP 検証の確率・較正指標。
   *
   * @param examples 評価例数
   * @param marginalNll 席別の順位周辺確率の平均 NLL
   * @param marginalBrier 席別の順位周辺確率の Brierスコア
   * @param ordinalRps 順位付き確率スコア（RPS）。予測と正解の累積確率の二乗誤差を、三つの順位境界と全席で平均した値
   * @param macroEce 各順位について全席の予測から求めた期待較正誤差（ECE）の、4順位での平均
   * @param lastRankEce 4着確率の ECE
   * @param rankAccuracy 予測確率が最も高い順位の正解率
   * @param invalidProbabilities 非有限または範囲外確率の件数
   * @param maxRowSumError 席ごとの確率和最大誤差
   * @param maxColumnSumError 順位ごとの確率和最大誤差
   */
  public record EvalMetrics(
      int examples,
      float marginalNll,
      float marginalBrier,
      float ordinalRps,
      float macroEce,
      float lastRankEce,
      float rankAccuracy,
      int invalidProbabilities,
      float maxRowSumError,
      float maxColumnSumError) {

    /**
     * サンプルを含まない評価結果を返す。
     *
     * @return 全カウンターと指標が 0 の結果
     */
    public static EvalMetrics empty() {
      return new EvalMetrics(0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0.0f, 0.0f);
    }
  }
}
