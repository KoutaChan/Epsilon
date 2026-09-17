package com.epsilon.nano.ai.decision.audit;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.config.settings.DecisionTrainSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionSample;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionTrainingSampleDescriptor;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionTrainingSampleDescriptorReader;
import com.epsilon.nano.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionDeviceBatch;
import com.epsilon.nano.ai.decision.training.DecisionOnlineLossConfig;
import com.epsilon.nano.ai.decision.training.DecisionPolicySignalMultipliers;
import com.epsilon.nano.ai.decision.training.DecisionPolicyTrustRegion;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionTrainer;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionTrainingBatch;
import com.epsilon.nano.ai.model.EpsilonDecisionLoss;
import com.epsilon.nano.ai.model.EpsilonDecisionNetwork;
import com.epsilon.nano.ai.model.EpsilonDecisionOutput;
import com.epsilon.nano.config.settings.DecisionSelectedPgCampaignSettings;
import com.epsilon.nano.config.settings.EpsilonSettings;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * 学習に選んだ全サンプルで、更新前後の選択行動の確率比と方策の KL ダイバージェンスを再計測する。
 *
 * <p>モデルは更新しない。メモリ上の入力は一括で、ファイルからの入力は一定件数ずつ読み込む。方策の学習対象外のサンプルも入力と出力が有限値か検証し、方策の更新量と KL の集計からのみ除外する。
 */
public final class EpsilonDecisionFinalPolicyAudit {

  private EpsilonDecisionFinalPolicyAudit() {}

  /**
   * 与えられた順序を変えず、固定幅バッチごとに方策モデル更新と対局生成 KL 検証条件の診断量を計算する。
   *
   * <p>順伝播は推論モードで行い、逆伝播、オプティマイザー、パラメーター複製は一切実行しない。
   *
   * @param model 監査対象 Decision モデル
   * @param samples 省略せず評価する選択した入力元
   * @param batchSize 一回の論理バッチ行数
   * @param config 本番損失と同じ更新設定
   * @return 入力元全体のKL・探索学習への寄与・方策更新比率監査結果
   */
  public static Report evaluate(
      Model model,
      List<EpsilonDecisionSample> samples,
      int batchSize,
      DecisionOnlineLossConfig config) {
    return evaluate(model, samples, batchSize, config, EpsilonSettings.defaults());
  }

  /** 起動時に確定した設定スナップショットで監査する。 */
  public static Report evaluate(
      Model model,
      List<EpsilonDecisionSample> samples,
      int batchSize,
      DecisionOnlineLossConfig config,
      SettingsLoader snapshot) {
    return evaluate(
        model,
        samples,
        batchSize,
        config,
        DecisionPolicySignalMultipliers.allCategories(),
        snapshot);
  }

  /**
   * 判断機会別方策モデル重みを本番更新と同じ定義で適用する全件監査。
   *
   * @param model 監査対象 Decision モデル
   * @param samples 省略せず評価する選択した入力元
   * @param batchSize 一回の論理バッチ行数
   * @param config 本番損失と同じ更新設定
   * @param policySignalMultipliers 判断機会別方策更新量倍率
   * @return 入力元全体のKL・探索学習への寄与・方策更新比率監査結果
   */
  public static Report evaluate(
      Model model,
      List<EpsilonDecisionSample> samples,
      int batchSize,
      DecisionOnlineLossConfig config,
      DecisionPolicySignalMultipliers policySignalMultipliers) {
    return evaluate(
        model, samples, batchSize, config, policySignalMultipliers, EpsilonSettings.defaults());
  }

  /** 起動時に確定した設定スナップショットで監査する。 */
  public static Report evaluate(
      Model model,
      List<EpsilonDecisionSample> samples,
      int batchSize,
      DecisionOnlineLossConfig config,
      DecisionPolicySignalMultipliers policySignalMultipliers,
      SettingsLoader snapshot) {
    return evaluateTotals(
            model,
            samples,
            batchSize,
            snapshot.bind(DecisionSelectedPgCampaignSettings.class).maximumDeviceTransitionCells(),
            config,
            policySignalMultipliers,
            snapshot)
        .finish(
            samples.size(),
            "final policy audit must cover the supplied source and contain Actor-eligible samples");
  }

  /**
   * 学習データファイル読み取り処理が返す選択した入力元を省略せず全件監査する逐次処理版。
   *
   * <p>{@code maxSamplesPerChunk} は監査対象総数の上限ではなく、同時に保持するチャンク幅である。各チャンクの平均 KL は方策学習の重み
   * で再集約するため、入力元全体の判定を保つ。
   *
   * @param model 監査対象 Decision モデル
   * @param fragmentPaths 読み込む学習データファイルの順序付きパス
   * @param reader 学習データファイルから選択したサンプルを読むコールバック
   * @param batchSize 一回の論理バッチ行数
   * @param maxSamplesPerChunk 同時にデータを復元する最大サンプル数
   * @param config 本番損失と同じ更新設定
   * @return 全学習データファイルを集約した監査結果
   * @throws IOException 学習データファイルの読込に失敗した場合
   */
  public static Report evaluateStreaming(
      Model model,
      List<Path> fragmentPaths,
      EpsilonDecisionTrainingSampleDescriptorReader reader,
      int batchSize,
      int maxSamplesPerChunk,
      DecisionOnlineLossConfig config)
      throws IOException {
    return evaluateStreaming(
        model,
        fragmentPaths,
        reader,
        batchSize,
        maxSamplesPerChunk,
        config,
        EpsilonSettings.defaults());
  }

  /** 起動時に確定した設定スナップショットで監査する。 */
  public static Report evaluateStreaming(
      Model model,
      List<Path> fragmentPaths,
      EpsilonDecisionTrainingSampleDescriptorReader reader,
      int batchSize,
      int maxSamplesPerChunk,
      DecisionOnlineLossConfig config,
      SettingsLoader snapshot)
      throws IOException {
    return evaluateStreaming(
        model,
        fragmentPaths,
        reader,
        batchSize,
        maxSamplesPerChunk,
        config,
        DecisionPolicySignalMultipliers.allCategories(),
        snapshot);
  }

  /**
   * 判断機会別方策モデル重みを本番更新と同じ定義で適用する逐次処理全件監査。
   *
   * @param model 監査対象 Decision モデル
   * @param fragmentPaths 読み込む学習データファイルの順序付きパス
   * @param reader 学習データファイルから選択したサンプルを読むコールバック
   * @param batchSize 一回の論理バッチ行数
   * @param maxSamplesPerChunk 同時にデータを復元する最大サンプル数
   * @param config 本番損失と同じ更新設定
   * @param policySignalMultipliers 判断機会別方策更新量倍率
   * @return 全学習データファイルを集約した監査結果
   * @throws IOException 学習データファイルの読込に失敗した場合
   */
  public static Report evaluateStreaming(
      Model model,
      List<Path> fragmentPaths,
      EpsilonDecisionTrainingSampleDescriptorReader reader,
      int batchSize,
      int maxSamplesPerChunk,
      DecisionOnlineLossConfig config,
      DecisionPolicySignalMultipliers policySignalMultipliers)
      throws IOException {
    return evaluateStreaming(
        model,
        fragmentPaths,
        reader,
        batchSize,
        maxSamplesPerChunk,
        config,
        policySignalMultipliers,
        EpsilonSettings.defaults());
  }

  /** 起動時に確定した設定スナップショットで監査する。 */
  public static Report evaluateStreaming(
      Model model,
      List<Path> fragmentPaths,
      EpsilonDecisionTrainingSampleDescriptorReader reader,
      int batchSize,
      int maxSamplesPerChunk,
      DecisionOnlineLossConfig config,
      DecisionPolicySignalMultipliers policySignalMultipliers,
      SettingsLoader snapshot)
      throws IOException {
    return evaluateStreaming(
        model,
        fragmentPaths,
        reader,
        batchSize,
        maxSamplesPerChunk,
        snapshot.bind(DecisionSelectedPgCampaignSettings.class).maximumDeviceTransitionCells(),
        config,
        policySignalMultipliers,
        snapshot);
  }

  /**
   * 方策モデル重みとデバイス遷移-格納枠上限を本番更新と揃える逐次処理全件監査。
   *
   * @param model 監査対象 Decision モデル
   * @param fragmentPaths 読み込む学習データファイルの順序付きパス
   * @param reader 学習データファイルから選択したサンプルを読むコールバック
   * @param batchSize 一回の論理バッチ行数
   * @param maxSamplesPerChunk 同時にデータを復元する最大サンプル数
   * @param maximumDeviceTransitionCells 一回のデバイス側バッチに許す遷移格納枠数
   * @param config 本番損失と同じ更新設定
   * @param policySignalMultipliers 判断機会別方策更新量倍率
   * @return 全学習データファイルを集約した監査結果
   * @throws IOException 学習データファイルの読込に失敗した場合
   */
  public static Report evaluateStreaming(
      Model model,
      List<Path> fragmentPaths,
      EpsilonDecisionTrainingSampleDescriptorReader reader,
      int batchSize,
      int maxSamplesPerChunk,
      int maximumDeviceTransitionCells,
      DecisionOnlineLossConfig config,
      DecisionPolicySignalMultipliers policySignalMultipliers)
      throws IOException {
    return evaluateStreaming(
        model,
        fragmentPaths,
        reader,
        batchSize,
        maxSamplesPerChunk,
        maximumDeviceTransitionCells,
        config,
        policySignalMultipliers,
        EpsilonSettings.defaults());
  }

  /** 起動時に確定した設定スナップショットで監査する。 */
  public static Report evaluateStreaming(
      Model model,
      List<Path> fragmentPaths,
      EpsilonDecisionTrainingSampleDescriptorReader reader,
      int batchSize,
      int maxSamplesPerChunk,
      int maximumDeviceTransitionCells,
      DecisionOnlineLossConfig config,
      DecisionPolicySignalMultipliers policySignalMultipliers,
      SettingsLoader snapshot)
      throws IOException {
    if (batchSize <= 0 || maxSamplesPerChunk <= 0) {
      throw new IllegalArgumentException(
          "streaming final audit batch and chunk sizes must be positive");
    }
    long expectedSamples = 0L;
    AuditTotals totals = new AuditTotals();
    for (Path path : fragmentPaths) {
      List<EpsilonDecisionTrainingSampleDescriptor> descriptors = reader.read(path);
      expectedSamples = Math.addExact(expectedSamples, descriptors.size());
      for (int start = 0; start < descriptors.size(); start += maxSamplesPerChunk) {
        int end = Math.min(descriptors.size(), start + maxSamplesPerChunk);
        ArrayList<EpsilonDecisionSample> chunk = new ArrayList<>(end - start);
        for (int index = start; index < end; index++) {
          chunk.add(descriptors.get(index).materializeChecked());
        }
        totals.add(
            evaluateTotals(
                model,
                chunk,
                batchSize,
                maximumDeviceTransitionCells,
                config,
                policySignalMultipliers,
                snapshot));
      }
    }
    return totals.finish(
        expectedSamples, "streaming final policy audit source must contain Actor-eligible samples");
  }

  private static AuditTotals evaluateTotals(
      Model model,
      List<EpsilonDecisionSample> samples,
      int batchSize,
      int maximumDeviceTransitionCells,
      DecisionOnlineLossConfig config,
      DecisionPolicySignalMultipliers policySignalMultipliers,
      SettingsLoader snapshot) {
    if (samples.isEmpty()) {
      throw new IllegalArgumentException("final policy audit samples must not be empty");
    }
    if (batchSize <= 0 || maximumDeviceTransitionCells <= 0) {
      throw new IllegalArgumentException(
          "final policy audit batchSize and maximumDeviceTransitionCells must be positive");
    }
    EpsilonDecisionNetwork network = requireDecisionNetwork(model);
    AuditTotals totals = new AuditTotals();

    var tensorTransfer = snapshot.bind(DecisionTrainSettings.class).tensorTransfer();
    EpsilonDecisionTrainingBatch.Materializer materializer =
        new EpsilonDecisionTrainingBatch.Materializer();
    for (int start = 0; start < samples.size(); ) {
      int count = exactBucketBatchCount(samples, start, batchSize, maximumDeviceTransitionCells);
      EpsilonDecisionTrainingBatch batch = materializer.materialize(samples, start, count);
      EpsilonDecisionTrainer.applyProductionActorWeights(
          batch, samples, start, count, policySignalMultipliers);
      double batchActorWeight = requireFiniteBatchAndMeasureActorWeight(batch, start, count);
      try (NDManager sub = model.getNDManager().newSubManager()) {
        DecisionDeviceBatch deviceBatch =
            DecisionBatchTransfer.transferTrainingToDevice(
                sub, batch.host().sliceRows(0, batch.size()), null, tensorTransfer);
        EpsilonDecisionLoss.TrainingLossResult losses;
        EpsilonDecisionOutput output =
            network.forwardDecision(
                new ParameterStore(sub, false), deviceBatch, false, new PairList<>());
        NDList rawOutput = output.toNDList();
        sub.attachAll(rawOutput);
        requireFiniteOutput(rawOutput);
        losses =
            EpsilonDecisionLoss.computeOnlineTrainingLoss(
                output, deviceBatch, config, batchActorWeight > 0.0);

        double batchMeanRolloutPolicyKl =
            finiteScalar(losses.meanRolloutPolicyKl(), "meanRolloutPolicyKl");
        double batchMaximumRolloutPolicyKl =
            finiteScalar(losses.maximumRolloutPolicyKl(), "maximumRolloutPolicyKl");
        double batchMaximumPolicyUpdateRatio =
            finiteScalar(
                losses.policyRatioDiagnostics().policyUpdate().max(),
                "maximumObservedPolicyUpdateRatio");
        finiteScalar(losses.total(), "totalLoss");
        double reportedActorWeight = finiteScalar(losses.actorWeight(), "actorWeight");
        double expectedMeanActorWeight = batchActorWeight / count;
        if (reportedActorWeight < 0.0
            || Math.abs(reportedActorWeight - expectedMeanActorWeight) > 1.0e-6) {
          throw new IllegalStateException(
              "final policy audit actor weight mismatch: expected="
                  + expectedMeanActorWeight
                  + " actual="
                  + reportedActorWeight);
        }

        totals.addBatch(
            count,
            batchActorWeight,
            batchMeanRolloutPolicyKl,
            batchMaximumRolloutPolicyKl,
            batchMaximumPolicyUpdateRatio);
      }
      start += count;
    }
    return totals;
  }

  private static int exactBucketBatchCount(
      List<EpsilonDecisionSample> samples,
      int start,
      int maximumRows,
      int maximumDeviceTransitionCells) {
    DecisionBucket bucket = samples.get(start).input().bucket();
    int cellsPerRow =
        Math.multiplyExact(bucket.legalActionCapacity(), bucket.actionTransitionCapacity());
    int limit = Math.min(maximumRows, Math.max(1, maximumDeviceTransitionCells / cellsPerRow));
    int count = 1;
    while (count < limit
        && start + count < samples.size()
        && samples.get(start + count).input().bucket().equals(bucket)) {
      count++;
    }
    return count;
  }

  private static EpsilonDecisionNetwork requireDecisionNetwork(Model model) {
    if (!(model.getBlock() instanceof EpsilonDecisionNetwork network)) {
      throw new IllegalArgumentException(
          "final policy audit requires EpsilonDecisionNetwork, got " + model.getBlock());
    }
    return network;
  }

  private static double requireFiniteBatchAndMeasureActorWeight(
      EpsilonDecisionTrainingBatch batch, int absoluteStart, int count) {
    requireFinite(
        batch.host().inputs().denseNumerics(),
        batch.host().activeInputNumericElementCount(),
        "inputNumericSlab",
        absoluteStart);
    requireFinite(
        batch.host().trainingTargets().numericSlab(),
        batch.host().trainingTargets().numericSlab().length,
        "targetNumericSlab",
        absoluteStart);

    double actorWeight = 0.0;
    for (int row = 0; row < count; row++) {
      double weight = (double) batch.actorWeight(row) * batch.sampleWeight(row);
      if (batch.actorWeight(row) < 0.0f
          || batch.sampleWeight(row) < 0.0f
          || weight < 0.0
          || !Double.isFinite(weight)) {
        throw new IllegalArgumentException(
            "invalid final policy audit actor weight at index=" + (absoluteStart + row));
      }
      actorWeight += weight;
    }
    return actorWeight;
  }

  private static void requireFiniteOutput(NDList output) {
    if (output.size() != EpsilonDecisionOutput.TENSOR_COUNT) {
      throw new IllegalArgumentException(
          "Decision output NDList must have "
              + EpsilonDecisionOutput.TENSOR_COUNT
              + " arrays, got "
              + output.size());
    }
    for (int index = 0; index < output.size(); index++) {
      float[] values = output.get(index).toFloatArray();
      for (int element = 0; element < values.length; element++) {
        if (!Float.isFinite(values[element])) {
          throw new IllegalStateException(
              "non-finite Decision output at tensor=" + index + " element=" + element);
        }
      }
    }
  }

  private static void requireFinite(float[] values, int length, String label, int absoluteStart) {
    for (int index = 0; index < length; index++) {
      if (!Float.isFinite(values[index])) {
        throw new IllegalArgumentException(
            "non-finite " + label + " in batch starting at sample=" + absoluteStart);
      }
    }
  }

  private static double finiteScalar(NDArray value, String label) {
    double scalar = value.getFloat();
    requireFinite(scalar, label);
    return scalar;
  }

  private static void requireFinite(double value, String label) {
    if (!Double.isFinite(value)) {
      throw new IllegalStateException("non-finite final policy audit " + label + ": " + value);
    }
  }

  private static final class AuditTotals {
    private long sampleCount;
    private int batchCount;
    private double weightedRolloutPolicyKlSum;
    private double actorWeight;
    private double maximumRolloutPolicyKl;
    private double maximumObservedPolicyUpdateRatio;

    private void addBatch(
        int samples,
        double batchActorWeight,
        double batchMeanRolloutPolicyKl,
        double batchMaximumRolloutPolicyKl,
        double batchMaximumPolicyUpdateRatio) {
      sampleCount = Math.addExact(sampleCount, samples);
      batchCount = Math.addExact(batchCount, 1);
      weightedRolloutPolicyKlSum += batchMeanRolloutPolicyKl * batchActorWeight;
      actorWeight += batchActorWeight;
      maximumRolloutPolicyKl = Math.max(maximumRolloutPolicyKl, batchMaximumRolloutPolicyKl);
      maximumObservedPolicyUpdateRatio =
          Math.max(maximumObservedPolicyUpdateRatio, batchMaximumPolicyUpdateRatio);
    }

    private void add(AuditTotals other) {
      sampleCount = Math.addExact(sampleCount, other.sampleCount);
      batchCount = Math.addExact(batchCount, other.batchCount);
      weightedRolloutPolicyKlSum += other.weightedRolloutPolicyKlSum;
      actorWeight += other.actorWeight;
      maximumRolloutPolicyKl = Math.max(maximumRolloutPolicyKl, other.maximumRolloutPolicyKl);
      maximumObservedPolicyUpdateRatio =
          Math.max(maximumObservedPolicyUpdateRatio, other.maximumObservedPolicyUpdateRatio);
    }

    private Report finish(long expectedSamples, String errorMessage) {
      if (sampleCount != expectedSamples
          || sampleCount <= 0L
          || batchCount <= 0
          || !(actorWeight > 0.0)) {
        throw new IllegalArgumentException(errorMessage);
      }
      double meanRolloutPolicyKl = weightedRolloutPolicyKlSum / actorWeight;
      requireFinite(meanRolloutPolicyKl, "meanRolloutPolicyKl");
      requireFinite(maximumRolloutPolicyKl, "maximumRolloutPolicyKl");
      requireFinite(maximumObservedPolicyUpdateRatio, "maximumObservedPolicyUpdateRatio");
      return new Report(
          sampleCount,
          batchCount,
          meanRolloutPolicyKl,
          maximumRolloutPolicyKl,
          maximumObservedPolicyUpdateRatio,
          actorWeight);
    }
  }

  /**
   * 方策学習の重みで集約し、指定入力元全件を覆った監査レポート。
   *
   * @param sampleCount 監査した選択行動サンプル数
   * @param batchCount 監査したバッチ数
   * @param meanRolloutPolicyKl 方策学習の重みで加重した {@code KL(piRollout || piCurrent)} 平均
   * @param maximumRolloutPolicyKl サンプル単位対局生成 KLの最大値
   * @param maximumObservedPolicyUpdateRatio 観測した {@code piCurrent/piRollout} 最大値
   * @param actorWeight 集約に使った方策学習の重みの総和
   */
  public record Report(
      long sampleCount,
      int batchCount,
      double meanRolloutPolicyKl,
      double maximumRolloutPolicyKl,
      double maximumObservedPolicyUpdateRatio,
      double actorWeight) {

    /** 件数と有限値を検証して変更不可レポートを構築する。 */
    public Report {
      if (sampleCount <= 0 || batchCount <= 0 || !(actorWeight > 0.0)) {
        throw new IllegalArgumentException("final policy audit report must be non-empty");
      }
      requireFinite(meanRolloutPolicyKl, "meanRolloutPolicyKl");
      requireFinite(maximumRolloutPolicyKl, "maximumRolloutPolicyKl");
      requireFinite(maximumObservedPolicyUpdateRatio, "maximumObservedPolicyUpdateRatio");
      requireFinite(actorWeight, "actorWeight");
    }

    /** 指定した対局生成 KL 信頼領域の上限を満たすか判定する。 */
    public boolean passed(DecisionPolicyTrustRegion trustRegion) {
      return !trustRegion.enabled()
          || meanRolloutPolicyKl <= trustRegion.maximumOptimizerShardMeanKl();
    }
  }
}
