package com.epsilon.nano.ai.grp;

import ai.djl.Device;
import com.epsilon.ai.grp.EpsilonGrpExample;
import com.epsilon.ai.grp.EpsilonGrpInference;
import com.epsilon.ai.grp.EpsilonGrpTrainer;
import com.epsilon.ai.grp.EpsilonGrpValidationGate;
import com.epsilon.config.settings.GrpPretrainValidationSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionDataException;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionSample;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.nano.training.EpsilonLogPretrainDataCollector;
import com.epsilon.util.FormatUtils;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Decision の牌譜サンプルから GRP を学習する単体コマンドランナー。 */
public final class EpsilonGrpTrainingRunner {

  private static final Logger log = LoggerFactory.getLogger(EpsilonGrpTrainingRunner.class);

  private EpsilonGrpTrainingRunner() {}

  /**
   * 牌譜を試合単位で学習・検証に分け、GRP をエポックごとに評価・昇格する。
   *
   * @param checkpointDir Decision チェックポイントのルートディレクトリ。GRP 保存物はその {@code grp} 子に置く
   * @param logDir mjson または Tenhou 牌譜を探索する起点
   * @param epochs 学習エポック数
   * @param maxFiles 読み込む牌譜ファイル数の上限
   * @return 学習・検証・昇格結果の集約
   * @throws Exception 牌譜収集、学習、評価またはチェックポイント保存に失敗した場合
   */
  public static LogTrainSummary trainFromLogs(
      Path checkpointDir, Path logDir, int epochs, int maxFiles) throws Exception {
    return trainFromLogs(
        checkpointDir,
        logDir,
        epochs,
        maxFiles,
        com.epsilon.nano.config.settings.EpsilonSettings.defaults());
  }

  public static LogTrainSummary trainFromLogs(
      Path checkpointDir, Path logDir, int epochs, int maxFiles, SettingsLoader settings)
      throws Exception {
    List<Path> files = EpsilonLogPretrainDataCollector.listLogFiles(logDir, maxFiles);
    if (files.isEmpty()) {
      throw new IllegalStateException(
          "No supported GRP pretrain log files were found in logDir=" + logDir);
    }
    GrpPretrainValidationSettings validationSettings =
        settings.bind(GrpPretrainValidationSettings.class);
    float validationFraction = validationSettings.fraction();
    List<Path> trainFiles = new ArrayList<>();
    List<Path> validationFiles = new ArrayList<>();
    for (Path file : files) {
      if (isValidationFile(file, validationFraction)) {
        validationFiles.add(file);
      } else {
        trainFiles.add(file);
      }
    }
    if (trainFiles.isEmpty()) {
      trainFiles = files;
      validationFiles = List.of();
    }
    log.info(
        "GRP log pretrain file split: total={} train={} validation={} validationFraction={}",
        files.size(),
        trainFiles.size(),
        validationFiles.size(),
        validationFraction);
    // GRP例は試合×局ごとに1件へ重複排除されるため軽量で、全件をメモリへキャッシュして
    // エポック毎のログ再パースを避ける。
    CollectResult train = collectExamples(trainFiles, "train");
    if (train.examples.isEmpty()) {
      throw new IllegalStateException("No usable GRP samples were produced from logDir=" + logDir);
    }
    CollectResult validation = collectExamples(validationFiles, "validation");
    long validationGames =
        validation.examples.stream().map(EpsilonGrpExample::gameId).distinct().count();
    if (validationFraction > 0.0f
        && (validation.examples.size() < validationSettings.minExamples()
            || validationGames < validationSettings.minGames())) {
      throw new IllegalStateException(
          "GRP validation holdout is too small: examples="
              + validation.examples.size()
              + " games="
              + validationGames
              + " required="
              + validationSettings.minExamples()
              + " examples/"
              + validationSettings.minGames()
              + " games"
              + ". Supply more log files or explicitly set "
              + "epsilon.grp.pretrain.validation.fraction=0 to disable promotion validation.");
    }
    Device device =
        NetworkFactory.getGrpDevices(
                settings.bind(com.epsilon.config.settings.DeviceSettings.class))
            .primary();
    TrainTotals totals = new TrainTotals();
    totals.files = train.processedFiles;
    totals.skippedFiles = train.skippedFiles;
    EpsilonGrpInference.EvalMetrics validationMetrics = EpsilonGrpInference.EvalMetrics.empty();
    EpsilonGrpInference.EvalMetrics bestValidationMetrics = EpsilonGrpInference.EvalMetrics.empty();
    boolean hasPromotedCheckpoint =
        EpsilonGrpCheckpointManager.resolveExisting(checkpointDir.resolve("grp")) != null;
    try (EpsilonGrpTrainingSession grpSession =
        EpsilonGrpTrainingSession.openForTraining(checkpointDir, device, settings)) {
      if (hasPromotedCheckpoint && !validation.examples.isEmpty()) {
        bestValidationMetrics = grpSession.evaluateExamples(validation.examples);
        log.info(
            "GRP validation baseline: checkpointIteration={} examples={} marginalNll={} "
                + "marginalBrier={} macroEce={} lastRankEce={} rankAccuracy={}",
            grpSession.iteration(),
            bestValidationMetrics.examples(),
            FormatUtils.fixed5(bestValidationMetrics.marginalNll()),
            FormatUtils.fixed5(bestValidationMetrics.marginalBrier()),
            FormatUtils.fixed5(bestValidationMetrics.macroEce()),
            FormatUtils.fixed5(bestValidationMetrics.lastRankEce()),
            FormatUtils.fixed5(bestValidationMetrics.rankAccuracy()));
      }
      for (int epoch = 1; epoch <= epochs; epoch++) {
        EpsilonGrpTrainer.TrainMetrics metrics = grpSession.trainExamples(train.examples, 1);
        if (metrics.examples() == 0) {
          throw new IllegalStateException(
              "No usable GRP samples were produced from logDir=" + logDir);
        }
        totals.examples += metrics.examples();
        totals.batches += metrics.batches();
        totals.marginalNll += metrics.marginalNll() * metrics.examples();
        totals.rankAccuracy += metrics.rankAccuracy() * metrics.examples();
        validationMetrics = grpSession.evaluateExamples(validation.examples);
        if (!validation.examples.isEmpty()
            && !EpsilonGrpValidationGate.structurallyValid(validationMetrics)) {
          throw new IllegalStateException(
              "GRP candidate produced invalid 4x4 marginals: invalidProbabilities="
                  + validationMetrics.invalidProbabilities()
                  + " maxRowSumError="
                  + validationMetrics.maxRowSumError()
                  + " maxColumnSumError="
                  + validationMetrics.maxColumnSumError());
        }
        boolean promoted =
            validation.examples.isEmpty()
                || !hasPromotedCheckpoint
                || EpsilonGrpValidationGate.better(validationMetrics, bestValidationMetrics);
        if (promoted) {
          grpSession.saveNext("grp_pretrain_logs");
          hasPromotedCheckpoint = true;
          bestValidationMetrics = validationMetrics;
          totals.promotedEpochs++;
        } else {
          totals.rejectedEpochs++;
          grpSession.restoreLatest();
        }
        log.info(
            "GRP log pretrain epoch complete: epoch={} files={} skippedFiles={} "
                + "examples={} batches={} marginalNll={} rankAccuracy={} "
                + "validationExamples={} validationMarginalNll={} "
                + "validationMarginalBrier={} validationMacroEce={} validationLastRankEce={} "
                + "validationRankAccuracy={} validationInvalidProbabilities={} "
                + "validationMaxRowSumError={} validationMaxColumnSumError={} "
                + "promoted={} promotedEpochs={} rejectedEpochs={}",
            epoch,
            train.processedFiles,
            train.skippedFiles,
            metrics.examples(),
            metrics.batches(),
            FormatUtils.fixed5(metrics.marginalNll()),
            FormatUtils.fixed5(metrics.rankAccuracy()),
            validationMetrics.examples(),
            FormatUtils.fixed5(validationMetrics.marginalNll()),
            FormatUtils.fixed5(validationMetrics.marginalBrier()),
            FormatUtils.fixed5(validationMetrics.macroEce()),
            FormatUtils.fixed5(validationMetrics.lastRankEce()),
            FormatUtils.fixed5(validationMetrics.rankAccuracy()),
            validationMetrics.invalidProbabilities(),
            FormatUtils.fixed5(validationMetrics.maxRowSumError()),
            FormatUtils.fixed5(validationMetrics.maxColumnSumError()),
            promoted,
            totals.promotedEpochs,
            totals.rejectedEpochs);
      }
      Path grpDir = checkpointDir.resolve("grp");
      Files.createDirectories(grpDir);
      Files.writeString(
          grpDir.resolve("grp-log-pretrain.txt"),
          "logDir="
              + logDir
              + System.lineSeparator()
              + "epochs="
              + epochs
              + System.lineSeparator()
              + "maxFiles="
              + maxFiles
              + System.lineSeparator()
              + "files="
              + totals.files
              + System.lineSeparator()
              + "skippedFiles="
              + totals.skippedFiles
              + System.lineSeparator()
              + "globalStep="
              + grpSession.globalStep()
              + System.lineSeparator()
              + "iteration="
              + grpSession.iteration()
              + System.lineSeparator()
              + "examples="
              + totals.examples
              + System.lineSeparator()
              + "batches="
              + totals.batches
              + System.lineSeparator()
              + "promotedEpochs="
              + totals.promotedEpochs
              + System.lineSeparator()
              + "rejectedEpochs="
              + totals.rejectedEpochs
              + System.lineSeparator()
              + "marginalNll="
              + avg(totals.marginalNll, totals.examples)
              + System.lineSeparator()
              + "rankAccuracy="
              + avg(totals.rankAccuracy, totals.examples)
              + System.lineSeparator()
              + "validationExamples="
              + bestValidationMetrics.examples()
              + System.lineSeparator()
              + "validationRankAccuracy="
              + bestValidationMetrics.rankAccuracy()
              + System.lineSeparator()
              + "validationMarginalNll="
              + bestValidationMetrics.marginalNll()
              + System.lineSeparator()
              + "validationMarginalBrier="
              + bestValidationMetrics.marginalBrier()
              + System.lineSeparator()
              + "validationMacroEce="
              + bestValidationMetrics.macroEce()
              + System.lineSeparator()
              + "validationLastRankEce="
              + bestValidationMetrics.lastRankEce()
              + System.lineSeparator()
              + "validationInvalidProbabilities="
              + bestValidationMetrics.invalidProbabilities()
              + System.lineSeparator()
              + "validationMaxRowSumError="
              + bestValidationMetrics.maxRowSumError()
              + System.lineSeparator()
              + "validationMaxColumnSumError="
              + bestValidationMetrics.maxColumnSumError()
              + System.lineSeparator());
      return new LogTrainSummary(
          totals.files,
          totals.skippedFiles,
          totals.examples,
          totals.batches,
          grpSession.globalStep(),
          grpSession.iteration(),
          avg(totals.marginalNll, totals.examples),
          avg(totals.rankAccuracy, totals.examples),
          totals.promotedEpochs,
          totals.rejectedEpochs,
          bestValidationMetrics.examples(),
          bestValidationMetrics.marginalNll(),
          bestValidationMetrics.marginalBrier(),
          bestValidationMetrics.macroEce(),
          bestValidationMetrics.lastRankEce(),
          bestValidationMetrics.rankAccuracy(),
          bestValidationMetrics.invalidProbabilities(),
          bestValidationMetrics.maxRowSumError(),
          bestValidationMetrics.maxColumnSumError());
    }
  }

  /** ファイル名の安定ハッシュで決定的にホールドアウトへ振り分ける(判断事前学習と同じ規約)。 */
  static boolean isValidationFile(Path file, float validationFraction) {
    if (validationFraction <= 0.0f) {
      return false;
    }
    int bucket = Math.floorMod(file.getFileName().toString().hashCode(), 10_000);
    return bucket < Math.round(validationFraction * 10_000.0f);
  }

  private static CollectResult collectExamples(List<Path> files, String split) throws Exception {
    CollectResult result = new CollectResult();
    for (Path file : files) {
      List<EpsilonDecisionSample> samples;
      try {
        samples = EpsilonLogPretrainDataCollector.collectDecisionFile(file);
      } catch (EpsilonDecisionDataException e) {
        throw e;
      } catch (Exception e) {
        result.skippedFiles++;
        log.warn(
            "Failed to load GRP pretrain log: split={} file={} error={}",
            split,
            file,
            e.getMessage());
        continue;
      }
      List<EpsilonGrpExample> examples = EpsilonGrpExamples.deduped(samples);
      if (examples.isEmpty()) {
        result.skippedFiles++;
        log.warn("Skipped GRP pretrain log with no samples: split={} file={}", split, file);
        continue;
      }
      result.examples.addAll(examples);
      result.processedFiles++;
    }
    log.info(
        "GRP log pretrain examples collected: split={} files={} skippedFiles={} examples={}",
        split,
        result.processedFiles,
        result.skippedFiles,
        result.examples.size());
    return result;
  }

  private static float avg(float sum, int count) {
    return sum / count;
  }

  /**
   * GRP 牌譜事前学習と検証選択の集約結果。
   *
   * @param files 正常に処理した牌譜ファイル数
   * @param skippedFiles 除外したファイル数
   * @param examples 学習例数
   * @param batches オプティマイザーバッチ数
   * @param globalStep 保存後チェックポイントの累積更新回数
   * @param iteration 保存後チェックポイントの反復回数
   * @param marginalNll 学習分割の席-順位周辺分布 NLL
   * @param rankAccuracy 学習分割の argmax 順位正解率
   * @param promotedEpochs 検証改善で採用したエポック数
   * @param rejectedEpochs 検証非改善で棄却したエポック数
   * @param validationExamples 検証例数
   * @param validationMarginalNll 検証周辺分布 NLL
   * @param validationMarginalBrier 検証 Brier スコア
   * @param validationMacroEce 検証データにおける ECE のマクロ平均
   * @param validationLastRankEce 検証 4着 ECE
   * @param validationRankAccuracy 検証順位正解率
   * @param validationInvalidProbabilities 不正確率件数
   * @param validationMaxRowSumError 席ごとの確率和最大誤差
   * @param validationMaxColumnSumError 順位ごとの確率和最大誤差
   */
  public record LogTrainSummary(
      int files,
      int skippedFiles,
      int examples,
      int batches,
      int globalStep,
      int iteration,
      float marginalNll,
      float rankAccuracy,
      int promotedEpochs,
      int rejectedEpochs,
      int validationExamples,
      float validationMarginalNll,
      float validationMarginalBrier,
      float validationMacroEce,
      float validationLastRankEce,
      float validationRankAccuracy,
      int validationInvalidProbabilities,
      float validationMaxRowSumError,
      float validationMaxColumnSumError) {}

  private static final class CollectResult {
    final List<EpsilonGrpExample> examples = new ArrayList<>();
    int processedFiles;
    int skippedFiles;
  }

  private static final class TrainTotals {
    int files;
    int skippedFiles;
    int examples;
    int batches;
    int promotedEpochs;
    int rejectedEpochs;
    float marginalNll;
    float rankAccuracy;
  }
}
