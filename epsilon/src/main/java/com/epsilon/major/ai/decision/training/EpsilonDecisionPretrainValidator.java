package com.epsilon.major.ai.decision.training;

import ai.djl.Model;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.major.ai.decision.EpsilonDecisionConstants;
import com.epsilon.major.ai.decision.data.EpsilonDecisionDataException;
import com.epsilon.major.ai.decision.data.EpsilonDecisionSample;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.major.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.major.ai.network.NetworkFactory;
import com.epsilon.major.config.settings.DecisionPretrainSettings;
import com.epsilon.major.config.settings.DecisionPretrainValidationSettings;
import com.epsilon.major.config.settings.EpsilonSettings;
import com.epsilon.major.training.EpsilonLogPretrainDataCollector;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 事前学習ホールドアウト評価。方策一致率と、スカラー効用価値の教師適合・実結果誤差を分離して測る。
 *
 * <p>合法行動が一つしかない行は選択問題ではないため、方策のtop-1、NLL、一様分布の比較基準、Decision種別件数、方策状態
 * 移動平均の分母から除外する。価値指標には全行を含め、{@link ValidationMetrics#samples()}と{@link
 * ValidationMetrics#policySamples()}で二つの母数を明示する。
 */
public final class EpsilonDecisionPretrainValidator {

  private static final Logger log = LoggerFactory.getLogger(EpsilonDecisionPretrainValidator.class);

  private EpsilonDecisionPretrainValidator() {}

  /**
   * 検証用再生ログを現在モデルで評価し、方策と価値の指標を分離集約する。
   *
   * @param checkpoint 検証する変更不可 Decision チェックポイント
   * @param files 検証用再生ログの順序付き一覧
   * @param grpSession 局境界価値教師値を作る重みを固定したGRP セッション。無効時は {@code null}
   * @return ファイル・サンプル件数、方策精度・NLL、価値校正指標
   * @throws IOException チェックポイントを読み込めない場合
   */
  public static ValidationMetrics evaluate(
      Path checkpoint, List<Path> files, EpsilonGrpTrainingSession grpSession) throws IOException {
    return evaluate(checkpoint, files, grpSession, EpsilonSettings.defaults());
  }

  public static ValidationMetrics evaluate(
      Path checkpoint,
      List<Path> files,
      EpsilonGrpTrainingSession grpSession,
      SettingsLoader config)
      throws IOException {
    if (files.isEmpty()) {
      return ValidationMetrics.empty();
    }
    int maximumDeviceBatchRows =
        config.bind(DecisionPretrainValidationSettings.class).maximumDeviceBatchRows();
    try (Model model =
            EpsilonDecisionCheckpointManager.loadForInference(
                checkpoint,
                NetworkFactory.getLearnerDevice(
                    config.bind(com.epsilon.config.settings.DeviceSettings.class)),
                config.bind(com.epsilon.major.config.settings.DecisionInferenceSettings.class));
        EpsilonDecisionInferenceServer server =
            EpsilonDecisionInferenceServer.forFrozenModel(
                model,
                maximumDeviceBatchRows,
                config.bind(com.epsilon.major.config.settings.DecisionInferenceSettings.class),
                config.bind(com.epsilon.config.settings.DecisionInferenceFusionSettings.class))) {
      log.info(
          "Decision pretrain validation opened fresh checkpoint: checkpoint={} device={} "
              + "maximumDeviceBatchRows={}",
          checkpoint,
          model.getNDManager().getDevice(),
          maximumDeviceBatchRows);
      return evaluate(server, files, grpSession, config);
    }
  }

  static ValidationMetrics evaluate(
      Model model,
      List<Path> files,
      EpsilonGrpTrainingSession grpSession,
      int maximumDeviceBatchRows) {
    try (EpsilonDecisionInferenceServer server =
        EpsilonDecisionInferenceServer.forModel(model, maximumDeviceBatchRows)) {
      return evaluate(server, files, grpSession, EpsilonSettings.defaults());
    }
  }

  private static ValidationMetrics evaluate(
      EpsilonDecisionInferenceServer server,
      List<Path> files,
      EpsilonGrpTrainingSession grpSession,
      SettingsLoader config) {
    Totals totals = new Totals();
    long progressLogIntervalNanos =
        config.bind(DecisionPretrainSettings.class).progressLogIntervalNanos();
    int compFiles = 0;
    int visitedFiles = 0;
    long started = System.nanoTime();
    long nextProgressLog = started + progressLogIntervalNanos;
    for (Path file : files) {
      List<EpsilonDecisionSample> samples;
      try {
        samples =
            EpsilonDecisionPretrainTargets.prepare(
                EpsilonLogPretrainDataCollector.collectDecisionFile(file),
                grpSession,
                config
                    .bind(com.epsilon.major.config.settings.DecisionSettings.class)
                    .utilityProfile());
      } catch (EpsilonDecisionDataException e) {
        throw e;
      } catch (Exception e) {
        log.warn("Failed to load Decision validation log: file={} error={}", file, e.getMessage());
        samples = List.of();
      }
      if (!samples.isEmpty()) {
        accumulate(server, samples, totals);
        compFiles++;
      }
      visitedFiles++;
      long now = System.nanoTime();
      if (now >= nextProgressLog) {
        double elapsedSeconds = (now - started) / 1_000_000_000.0;
        double filesPerSecond = visitedFiles / elapsedSeconds;
        double etaSeconds = (files.size() - visitedFiles) / filesPerSecond;
        log.info(
            "Decision pretrain validation progress: files={}/{} compiledFiles={}"
                + " samples={} policySamples={} elapsedSeconds={} etaSeconds={}"
                + " filesPerSecond={}",
            visitedFiles,
            files.size(),
            compFiles,
            totals.samples,
            totals.policySamples,
            elapsedSeconds,
            etaSeconds,
            filesPerSecond);
        nextProgressLog = now + progressLogIntervalNanos;
      }
    }
    return totals.toMetrics(compFiles);
  }

  private static void accumulate(
      EpsilonDecisionInferenceServer server, List<EpsilonDecisionSample> samples, Totals totals) {
    for (DecisionSampleBatcher.BatchSlice batchSlice :
        DecisionSampleBatcher.partitionByBucket(samples, server)) {
      EpsilonDecisionInferenceServer.DiagnosticBatchEvaluation evaluation =
          server.evaluateDiagnosticBatch(batchSlice.hostBatch());
      totals.addStateBatch(
          evaluation.policyStateEmbeddings(),
          evaluation.valueStateEmbeddings(),
          batchSlice.size(),
          batchSlice.hostBatch().bucket().legalActionCapacity() > 1);
      for (int batchRow = 0; batchRow < batchSlice.size(); batchRow++) {
        int sampleIndex = batchSlice.sampleIndexes()[batchRow];
        EpsilonDecisionSample sample = samples.get(sampleIndex);
        EpsilonDecisionInferenceServer.Prediction prediction =
            evaluation.predictions().get(batchRow);
        int legalActionCount = sample.input().legalActionCount(0);
        boolean policyEligible = legalActionCount > 1;
        float[] policyLogProbabilities = prediction.policyLogProbabilities();
        boolean correct =
            policyEligible && argmax(policyLogProbabilities) == sample.chosenLegalSlot();
        EpsilonDecisionPointKind kind =
            policyEligible ? EpsilonDecisionPointKind.of(sample.input(), 0) : null;
        boolean postCallDahai = policyEligible && isPostCallDahai(sample.input(), kind);
        double nll = 0.0;
        if (policyEligible) {
          float[] policyProbabilities = softmax(policyLogProbabilities);
          float chosenProb = policyProbabilities[sample.chosenLegalSlot()];
          nll = -Math.log(Math.max(1.0e-8f, chosenProb));
        }
        EpsilonUtilityProfile utilityProfile = EpsilonUtilityProfile.values()[sample.ruleProfile()];
        float actualUtility = utilityProfile.utilityForRank(sample.finalRank());
        totals.add(
            kind,
            postCallDahai,
            correct,
            nll,
            actualUtility,
            prediction.valueUtility(),
            sample.valueTarget(),
            legalActionCount,
            policyEligible);
        totals.addBoundary(
            sample.gameId(),
            sample.boundaryIndex(),
            sample.playerSeat(),
            sample.finalRank(),
            sample.input().boundaryRankPrior(0, sample.playerSeat()));
      }
    }
  }

  private static boolean isPostCallDahai(DecisionHostBatch input, EpsilonDecisionPointKind kind) {
    if (kind != EpsilonDecisionPointKind.DAHAI) {
      return false;
    }
    int afterChi = DecisionInputSchema.DiscardContext.AFTER_CHI.ordinal() + 1;
    int afterPon = DecisionInputSchema.DiscardContext.AFTER_PON.ordinal() + 1;
    for (int actionSlot = 0; actionSlot < input.legalActionCount(0); actionSlot++) {
      int context =
          input.transitionCategory(
              0, actionSlot, 0, DecisionInputSchema.ActionTransitionInt.DISCARD_CONTEXT);
      if (context == afterChi || context == afterPon) {
        return true;
      }
    }
    return false;
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }

  private static float[] softmax(float[] logits) {
    float max = Float.NEGATIVE_INFINITY;
    for (float logit : logits) {
      max = Math.max(max, logit);
    }
    float[] out = new float[logits.length];
    float sum = 0.0f;
    for (int i = 0; i < logits.length; i++) {
      out[i] = (float) Math.exp(logits[i] - max);
      sum += out[i];
    }
    if (!(sum > 0.0f) || !Float.isFinite(sum)) {
      float uniform = logits.length == 0 ? 0.0f : 1.0f / logits.length;
      Arrays.fill(out, uniform);
      return out;
    }
    for (int i = 0; i < out.length; i++) {
      out[i] /= sum;
    }
    return out;
  }

  /** 4順位分布の累積3成分と、実順位one-hotの累積3成分との平均二乗誤差。 */
  static float fSpaceRps(float[] rankProbability, int finalRank) {
    if (rankProbability == null || rankProbability.length != EpsilonDecisionConstants.PLAYERS) {
      throw new IllegalArgumentException(
          "rankProbability length must be " + EpsilonDecisionConstants.PLAYERS);
    }
    if (finalRank < 0 || finalRank >= EpsilonDecisionConstants.PLAYERS) {
      throw new IllegalArgumentException("finalRank must be 0-3: " + finalRank);
    }
    double cumulative = 0.0;
    double squaredError = 0.0;
    for (int component = 0; component < (EpsilonDecisionConstants.PLAYERS - 1); component++) {
      float probability = rankProbability[component];
      if (!Float.isFinite(probability)) {
        throw new IllegalArgumentException("rankProbability must be finite");
      }
      cumulative += probability;
      double actualCumulative = finalRank <= component ? 1.0 : 0.0;
      double error = cumulative - actualCumulative;
      squaredError += error * error;
    }
    if (!Float.isFinite(rankProbability[EpsilonDecisionConstants.PLAYERS - 1])) {
      throw new IllegalArgumentException("rankProbability must be finite");
    }
    return (float) (squaredError / (EpsilonDecisionConstants.PLAYERS - 1));
  }

  static final class Totals {
    long samples;
    long policySamples;
    long correct;
    final long[] kindSamples = new long[EpsilonDecisionPointKind.values().length];
    final long[] kindCorrect = new long[EpsilonDecisionPointKind.values().length];
    double nllSum;
    long postCallDahaiSamples;
    long postCallDahaiCorrect;
    double postCallDahaiNllSum;
    double valueTeacherErrorSum;
    double valueTeacherSquaredErrorSum;
    double valueErrorSum;
    double valueSquaredErrorSum;
    double actualUtilitySum;
    double actualUtilitySquareSum;
    double predictedUtilitySum;
    double predictedUtilitySquareSum;
    long boundarySamples;
    double grpBoundaryOutcomeNllSum;
    double grpBoundaryRpsSum;
    double policyUniformNllSum;
    final VectorMoments policyStateMoments = new VectorMoments();
    final VectorMoments valueStateMoments = new VectorMoments();
    final HashSet<BoundarySeatKey> observedBoundarySeats = new HashSet<>();

    void add(
        EpsilonDecisionPointKind kind,
        boolean postCallDahai,
        boolean isCorrect,
        double nll,
        float actualUtility,
        float predictedUtility,
        float targetUtility,
        int legalActionCount,
        boolean policyEligible) {
      if (!Float.isFinite(actualUtility)
          || !Float.isFinite(predictedUtility)
          || !Float.isFinite(targetUtility)) {
        throw new IllegalArgumentException("Value validation utilities must be finite");
      }
      samples++;
      if (policyEligible) {
        policySamples++;
        policyUniformNllSum += Math.log(legalActionCount);
        kindSamples[kind.ordinal()]++;
        if (isCorrect) {
          correct++;
          kindCorrect[kind.ordinal()]++;
        }
        nllSum += nll;
        if (postCallDahai) {
          postCallDahaiSamples++;
          postCallDahaiNllSum += nll;
          if (isCorrect) {
            postCallDahaiCorrect++;
          }
        }
      }
      double outcomeError = predictedUtility - (double) actualUtility;
      double teacherError = predictedUtility - (double) targetUtility;
      valueErrorSum += outcomeError;
      valueSquaredErrorSum += outcomeError * outcomeError;
      valueTeacherErrorSum += teacherError;
      valueTeacherSquaredErrorSum += teacherError * teacherError;
      actualUtilitySum += actualUtility;
      actualUtilitySquareSum += actualUtility * (double) actualUtility;
      predictedUtilitySum += predictedUtility;
      predictedUtilitySquareSum += predictedUtility * (double) predictedUtility;
    }

    /** 呼び出し元が切り出した独立した4順位配列を受け取り、その場で正規化して集計する。 */
    void addBoundary(
        long gameId,
        int boundaryIndex,
        int playerSeat,
        int finalRank,
        float[] grpBoundaryProbability) {
      if (!observedBoundarySeats.add(new BoundarySeatKey(gameId, boundaryIndex, playerSeat))) {
        return;
      }
      float[] prior = EpsilonGrpRanks.normalizeRankProbabilitiesInPlace(grpBoundaryProbability);
      boundarySamples++;
      grpBoundaryOutcomeNllSum -= Math.log(Math.max(1.0e-8, Math.min(1.0, prior[finalRank])));
      grpBoundaryRpsSum += fSpaceRps(prior, finalRank);
    }

    ValidationMetrics toMetrics(int files) {
      return new ValidationMetrics(
          files,
          samples,
          policySamples,
          ratio(correct, policySamples),
          ratio(
              kindCorrect[EpsilonDecisionPointKind.DAHAI.ordinal()],
              kindSamples[EpsilonDecisionPointKind.DAHAI.ordinal()]),
          ratio(
              kindCorrect[EpsilonDecisionPointKind.RIICHI.ordinal()],
              kindSamples[EpsilonDecisionPointKind.RIICHI.ordinal()]),
          ratio(
              kindCorrect[EpsilonDecisionPointKind.REACTION.ordinal()],
              kindSamples[EpsilonDecisionPointKind.REACTION.ordinal()]),
          ratio(postCallDahaiCorrect, postCallDahaiSamples),
          kindSamples[EpsilonDecisionPointKind.DAHAI.ordinal()],
          kindSamples[EpsilonDecisionPointKind.RIICHI.ordinal()],
          kindSamples[EpsilonDecisionPointKind.REACTION.ordinal()],
          postCallDahaiSamples,
          (float) (nllSum / Math.max(1L, policySamples)),
          (float) (postCallDahaiNllSum / Math.max(1L, postCallDahaiSamples)),
          mean(valueTeacherErrorSum),
          mean(valueTeacherSquaredErrorSum),
          mean(valueErrorSum),
          mean(valueSquaredErrorSum),
          boundarySamples,
          (float) (grpBoundaryOutcomeNllSum / Math.max(1L, boundarySamples)),
          (float) (grpBoundaryRpsSum / Math.max(1L, boundarySamples)),
          (float) (policyUniformNllSum / Math.max(1L, policySamples)),
          variance(actualUtilitySum, actualUtilitySquareSum),
          policyStateMoments.centeredRms(),
          valueStateMoments.centeredRms(),
          (float) Math.sqrt(variance(predictedUtilitySum, predictedUtilitySquareSum)));
    }

    void addStateBatch(float[] policyState, float[] valueState, int rows, boolean policyEligible) {
      if (policyEligible) {
        policyStateMoments.addBatch(policyState, rows);
      }
      valueStateMoments.addBatch(valueState, rows);
    }

    private float mean(double sum) {
      return (float) (sum / Math.max(1L, samples));
    }

    private float variance(double sum, double squareSum) {
      double count = Math.max(1L, samples);
      double mean = sum / count;
      return (float) Math.max(0.0, squareSum / count - mean * mean);
    }

    private static float ratio(long numerator, long denominator) {
      return denominator <= 0L ? 0.0f : (float) ((double) numerator / denominator);
    }
  }

  /** サンプル軸に沿う各特徴量の分散を集約し、定数ベクトルによる見かけのRMSを除外する。 */
  private static final class VectorMoments {
    private double[] sum = new double[0];
    private double[] squareSum = new double[0];
    private long rows;

    void addRow(float[] values) {
      addBatch(values, 1);
    }

    void addBatch(float[] values, int batchRows) {
      if (batchRows <= 0 || values.length == 0 || values.length % batchRows != 0) {
        throw new IllegalArgumentException(
            "invalid diagnostic batch: rows=" + batchRows + " values=" + values.length);
      }
      int width = values.length / batchRows;
      if (sum.length == 0) {
        sum = new double[width];
        squareSum = new double[width];
      } else if (sum.length != width) {
        throw new IllegalArgumentException(
            "diagnostic width changed: expected=" + sum.length + " actual=" + width);
      }
      for (int row = 0; row < batchRows; row++) {
        int offset = row * width;
        for (int column = 0; column < width; column++) {
          double value = values[offset + column];
          sum[column] += value;
          squareSum[column] += value * value;
        }
      }
      rows += batchRows;
    }

    float centeredRms() {
      if (rows <= 0L || sum.length == 0) {
        return 0.0f;
      }
      double variance = 0.0;
      for (int column = 0; column < sum.length; column++) {
        double mean = sum[column] / rows;
        variance += Math.max(0.0, squareSum[column] / rows - mean * mean);
      }
      return (float) Math.sqrt(variance / sum.length);
    }
  }

  /** 方策精度、スカラー価値の教師・実効用誤差、GRPの順位指標を分離した検証用結果。 valueConstantMseは、この検証用の実効用平均を用いた定数予測のMSE。 */
  public record ValidationMetrics(
      int files,
      long samples,
      long policySamples,
      float top1,
      float dahaiTop1,
      float riichiTop1,
      float reactionTop1,
      float postCallDahaiTop1,
      long dahaiSamples,
      long riichiSamples,
      long reactionSamples,
      long postCallDahaiSamples,
      float nll,
      float postCallDahaiNll,
      float valueTeacherBias,
      float valueTeacherMse,
      float valueBias,
      float valueMse,
      long boundarySamples,
      float grpBoundaryOutcomeNll,
      float grpBoundaryRps,
      float policyUniformNll,
      float valueConstantMse,
      float policyStateCenteredRms,
      float valueStateCenteredRms,
      float valueUtilityCenteredRms) {

    /** 検証用ファイルが無い場合の全ゼロ指標。 */
    public static ValidationMetrics empty() {
      return new ValidationMetrics(
          0, 0L, 0L, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0L, 0L, 0L, 0L, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
          0.0f, 0L, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
  }

  private record BoundarySeatKey(long gameId, int boundaryIndex, int playerSeat) {}
}
