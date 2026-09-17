package com.epsilon.major.ai.decision.training;

import ai.djl.Model;
import com.epsilon.config.settings.GrpSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.major.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.major.ai.network.NetworkFactory;
import com.epsilon.major.config.settings.DecisionPretrainSettings;
import com.epsilon.major.config.settings.DecisionPretrainValidationSettings;
import com.epsilon.major.config.settings.EpsilonSettings;
import com.epsilon.major.training.EpsilonLogPretrainDataCollector;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.util.FormatUtils;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 天鳳・mjson 形式の牌譜を使い、Decision ネットワークを教師あり学習する。 */
public final class EpsilonDecisionLogPretrainRunner {

  private static final Logger log = LoggerFactory.getLogger(EpsilonDecisionLogPretrainRunner.class);

  private final SettingsLoader config;

  private EpsilonDecisionLogPretrainRunner(SettingsLoader config) {
    this.config = config;
  }

  public static TrainSummary pretrainFromLogs(
      Path checkpointDir, Path logDir, int epochs, int maxFiles) throws Exception {
    return pretrainFromLogs(checkpointDir, logDir, epochs, maxFiles, EpsilonSettings.defaults());
  }

  public static TrainSummary pretrainFromLogs(
      Path checkpointDir, Path logDir, int epochs, int maxFiles, SettingsLoader config)
      throws Exception {
    return new EpsilonDecisionLogPretrainRunner(config)
        .runPretrain(checkpointDir, logDir, epochs, maxFiles);
  }

  public static ValidationSummary validateCheckpoint(
      Path grpCheckpointRoot, Path checkpoint, Path logDir, int maxFiles) throws Exception {
    return validateCheckpoint(
        grpCheckpointRoot, checkpoint, logDir, maxFiles, EpsilonSettings.defaults());
  }

  public static ValidationSummary validateCheckpoint(
      Path grpCheckpointRoot, Path checkpoint, Path logDir, int maxFiles, SettingsLoader config)
      throws Exception {
    return new EpsilonDecisionLogPretrainRunner(config)
        .runValidation(grpCheckpointRoot, checkpoint, logDir, maxFiles);
  }

  /**
   * 再生ログを判断行へ変換し、指定エポック数だけ教師あり事前学習する。
   *
   * @param checkpointDir 初期化チェックポイントと出力チェックポイントを管理する親ディレクトリ
   * @param logDir 天鳳または mjson 再生ログの探索親ディレクトリ
   * @param epochs 実行するエポック数
   * @param maxFiles 読み込む入力元ファイル上限。収集処理の規約に従う
   * @return 全エポックの件数・損失・出力チェックポイントをまとめた結果
   * @throws Exception ログ変換、学習、検証、チェックポイント保存のいずれかに失敗した場合
   */
  private TrainSummary runPretrain(Path checkpointDir, Path logDir, int epochs, int maxFiles)
      throws Exception {
    if (epochs <= 0) {
      throw new IllegalArgumentException("epochs must be positive");
    }
    List<Path> files = EpsilonLogPretrainDataCollector.listLogFiles(logDir, maxFiles);
    if (files.isEmpty()) {
      throw new IllegalStateException(
          "No supported Decision pretrain log files were found in logDir=" + logDir);
    }
    DecisionPretrainValidationSettings validationSettings =
        config.bind(DecisionPretrainValidationSettings.class);
    SourceSplit sourceSplit = splitSourceFiles(files, validationSettings.fraction());
    log.info(
        "Decision log pretrain file split: total={} train={} validation={} validationFraction={}",
        files.size(),
        sourceSplit.trainingFiles().size(),
        sourceSplit.validationFiles().size(),
        validationSettings.fraction());

    DecisionPretrainSettings pretrainSettings = config.bind(DecisionPretrainSettings.class);
    boolean grpTeacherEnabled = config.bind(GrpSettings.class).enabled();
    TrainingRun trainingRun;
    List<CheckpointValidation> validations;
    try (EpsilonGrpTrainingSession grpSession = openGrpSession(checkpointDir, grpTeacherEnabled)) {
      log.info(
          "Decision log pretrain GRP teacher: configured={} checkpointLoaded={} frozen=true",
          grpTeacherEnabled,
          grpSession.enabled());
      EpsilonDecisionPretrainDatasetCompiler.Compilation compilation =
          EpsilonDecisionPretrainDatasetCompiler.compileOrOpen(
              checkpointDir.resolve("compiled-decision-pretrain"),
              sourceSplit.trainingFiles(),
              grpSession,
              pretrainSettings,
              config
                  .bind(com.epsilon.major.config.settings.DecisionSettings.class)
                  .utilityProfile());
      try (compilation) {
        trainingRun =
            trainEpochs(
                checkpointDir,
                logDir,
                epochs,
                sourceSplit.trainingFiles(),
                compilation,
                pretrainSettings);
      }
      validations =
          validateEpochs(
              sourceSplit.validationFiles(), grpSession, trainingRun.epochs(), validationSettings);
    }
    Path outputCheckpoint =
        validations.stream()
            .filter(CheckpointValidation::usable)
            .map(CheckpointValidation::checkpoint)
            .reduce((first, second) -> second)
            .orElseThrow(
                () ->
                    new IllegalStateException(
                        "Decision pretrain completed without a structurally usable checkpoint"));
    log.info(
        "Decision log pretrain output selected without arena promotion: checkpoint={} "
            + "validatedCheckpoints={}",
        outputCheckpoint,
        validations.size());
    return new TrainSummary(
        trainingRun.totals().samplesSeen,
        trainingRun.totals().batches,
        trainingRun.globalStep(),
        trainingRun.iteration(),
        outputCheckpoint,
        validations);
  }

  /** 保存済み事前学習チェックポイントを新しく生成した推論モデルで副作用なく再検証する。 */
  private ValidationSummary runValidation(
      Path grpCheckpointRoot, Path checkpoint, Path logDir, int maxFiles) throws Exception {
    List<Path> files = EpsilonLogPretrainDataCollector.listLogFiles(logDir, maxFiles);
    DecisionPretrainValidationSettings settings =
        config.bind(DecisionPretrainValidationSettings.class);
    List<Path> validationFiles = splitSourceFiles(files, settings.fraction()).validationFiles();
    if (validationFiles.isEmpty()) {
      throw new IllegalStateException("Decision pretrain validation split is empty");
    }
    try (EpsilonGrpTrainingSession grpSession =
        openGrpSession(grpCheckpointRoot, config.bind(GrpSettings.class).enabled())) {
      EpsilonDecisionPretrainValidator.ValidationMetrics metrics =
          EpsilonDecisionPretrainValidator.evaluate(
              checkpoint, validationFiles, grpSession, config);
      logCheckpointValidation(checkpoint, metrics);
      EpsilonDecisionPretrainValidationAssessment.Result assessment =
          EpsilonDecisionPretrainValidationAssessment.evaluate(metrics, settings);
      if (assessment.usable()) {
        log.info(
            "Decision pretrain checkpoint validation usable: checkpoint={} assessment={}",
            checkpoint,
            assessment.summary());
      } else {
        log.warn(
            "Decision pretrain checkpoint validation invalid: checkpoint={} assessment={}",
            checkpoint,
            assessment.summary());
      }
      return new ValidationSummary(checkpoint, assessment.usable(), assessment.summary(), metrics);
    }
  }

  private TrainingRun trainEpochs(
      Path checkpointDir,
      Path logDir,
      int epochs,
      List<Path> trainingFiles,
      EpsilonDecisionPretrainDatasetCompiler.Compilation compilation,
      DecisionPretrainSettings settings)
      throws Exception {
    var decision = config.bind(com.epsilon.major.config.settings.DecisionSettings.class);
    Model model =
        NetworkFactory.createDecisionModel(
            NetworkFactory.getLearnerDevice(
                config.bind(com.epsilon.config.settings.DeviceSettings.class)),
            true,
            decision.hidden(),
            decision.utilityProfile());
    log.info("Decision log pretrain created a fresh hierarchical model and fresh optimizer state");

    int globalStep = 0;
    int iteration = 0;
    TrainTotals allEpochs = new TrainTotals();
    ArrayList<EpochArtifact> epochArtifacts = new ArrayList<>(epochs);
    try (DecisionExecutionContext executionContext = new DecisionExecutionContext();
        model;
        EpsilonDecisionPretrainer trainer =
            EpsilonDecisionPretrainer.open(model, config, executionContext)) {
      logTrainingConfiguration(trainer, settings);
      for (int epoch = 1; epoch <= epochs; epoch++) {
        trainer.setLearningRateScale((float) Math.pow(settings.lrDecayPerEpoch(), epoch - 1.0));
        EpsilonDecisionPretrainer.EpochMetrics training;
        try (EpsilonDecisionPretrainBatchCursor cursor =
            compilation.openEpoch(epoch, settings.datasetPrefetchShards())) {
          training = trainer.trainEpoch(cursor);
        }
        TrainTotals epochTotals = new TrainTotals();
        addTrainingMetrics(epochTotals, training);
        if (epochTotals.samplesSeen == 0) {
          throw new IllegalStateException(
              "No Decision samples were produced from logDir=" + logDir);
        }
        globalStep += epochTotals.batches;
        iteration++;
        Path epochCheckpoint =
            checkpointDir.resolve("decision_pretrain_logs_" + FormatUtils.zeroPad(iteration, 5));
        EpsilonDecisionCheckpointManager.save(model, epochCheckpoint, globalStep, iteration, 0);
        allEpochs.add(epochTotals);
        epochArtifacts.add(new EpochArtifact(epoch, epochCheckpoint));
        logTrainingEpoch(trainingFiles, compilation, epoch, epochTotals, training);
      }
    }
    log.info(
        "Decision pretrain training resources released before validation: epochs={} "
            + "lastIteration={}",
        epochs,
        iteration);
    return new TrainingRun(List.copyOf(epochArtifacts), allEpochs, globalStep, iteration);
  }

  private List<CheckpointValidation> validateEpochs(
      List<Path> validationFiles,
      EpsilonGrpTrainingSession grpSession,
      List<EpochArtifact> epochs,
      DecisionPretrainValidationSettings settings)
      throws IOException {
    ArrayList<CheckpointValidation> validations = new ArrayList<>(epochs.size());
    for (EpochArtifact epoch : epochs) {
      EpsilonDecisionPretrainValidator.ValidationMetrics metrics =
          validationFiles.isEmpty()
              ? EpsilonDecisionPretrainValidator.ValidationMetrics.empty()
              : EpsilonDecisionPretrainValidator.evaluate(
                  epoch.checkpoint(), validationFiles, grpSession, config);
      if (!validationFiles.isEmpty()) {
        logValidationMetrics(epoch.epoch(), metrics);
      }
      EpsilonDecisionPretrainValidationAssessment.Result assessment =
          EpsilonDecisionPretrainValidationAssessment.evaluate(metrics, settings);
      validations.add(
          new CheckpointValidation(
              epoch.epoch(),
              epoch.checkpoint(),
              assessment.usable(),
              assessment.summary(),
              metrics));
      if (assessment.usable()) {
        log.info(
            "Decision log pretrain checkpoint usable: epoch={} checkpoint={} " + "assessment={}",
            epoch.epoch(),
            epoch.checkpoint(),
            assessment.summary());
      } else {
        log.warn(
            "Decision log pretrain checkpoint invalid: epoch={} checkpoint={} " + "assessment={}",
            epoch.epoch(),
            epoch.checkpoint(),
            assessment.summary());
      }
    }
    return List.copyOf(validations);
  }

  private void logTrainingConfiguration(
      EpsilonDecisionPretrainer trainer, DecisionPretrainSettings settings) {
    log.info(
        "Decision log pretrain update mode: execution=FULL_MODEL_DATA_PARALLEL "
            + "activeDevices={} optimizerBatchRows={} "
            + "maxDeviceBatchRows={} tensorTransfer={} computePrecision={} "
            + "compilerReaderWorkers={} compilerPrefetchFiles={} "
            + "riichiDecisionWeight={} reactionDecisionWeight={} learningRate={} "
            + "lrDecayPerEpoch={} behaviorCloningCoef={} pretrainValueCoef={} "
            + "trainingPhase=warm-start-only actorFrozen=false",
        trainer.activeDevices(),
        settings.optimizerBatchRows(),
        settings.maximumDeviceBatchRows(),
        settings.tensorTransfer(),
        settings.computePrecision(),
        settings.compilerReaderWorkers(),
        settings.compilerPrefetchFiles(),
        settings.riichiDecisionWeight(),
        settings.reactionDecisionWeight(),
        settings.learningRate(),
        settings.lrDecayPerEpoch(),
        settings.behaviorCloningCoef(),
        settings.valueCoef());
  }

  private void logTrainingEpoch(
      List<Path> trainingFiles,
      EpsilonDecisionPretrainDatasetCompiler.Compilation compilation,
      int epoch,
      TrainTotals totals,
      EpsilonDecisionPretrainer.EpochMetrics training) {
    log.info(
        "Decision log pretrain epoch complete: phase={} epoch={} files={} "
            + "samples={} policySamples={} batches={} policyBatches={} "
            + "rowsPerSecond={} loss={} behaviorCloningLoss={} "
            + "valueLoss={} entropy={} chosenProb={}",
        "warm-start",
        epoch,
        compilation.reused() ? trainingFiles.size() : compilation.processedFiles(),
        totals.samplesSeen,
        totals.policySamplesSeen,
        totals.batches,
        totals.policyBatches,
        FormatUtils.fixed5(training.rowsPerSecond()),
        FormatUtils.fixed5(avg(totals.loss, totals.batches)),
        FormatUtils.fixed5(avg(totals.behaviorCloningLoss, totals.policyBatches)),
        FormatUtils.fixed5(avg(totals.valueLoss, totals.batches)),
        FormatUtils.fixed5(avg(totals.entropy, totals.policyBatches)),
        FormatUtils.fixed5(avg(totals.chosenProb, totals.policyBatches)));
  }

  private void logValidationMetrics(
      int epoch, EpsilonDecisionPretrainValidator.ValidationMetrics validation) {
    log.info(
        "Decision log pretrain validation: epoch={} files={} samples={} policySamples={} top1={}"
            + " dahaiTop1={} riichiTop1={} reactionTop1={} postCallDahaiTop1={} dahaiSamples={}"
            + " riichiSamples={} reactionSamples={} postCallDahaiSamples={} nll={}"
            + " postCallDahaiNll={} valueTeacherBias={} valueTeacherMse={} valueBias={} valueMse={}"
            + " valueConstantMse={} boundarySamples={} grpBoundaryOutcomeNll={} grpBoundaryRps={}"
            + " policyUniformNll={} policyStateCenteredRms={} valueStateCenteredRms={} "
            + "valueUtilityCenteredRms={}",
        epoch,
        validation.files(),
        validation.samples(),
        validation.policySamples(),
        FormatUtils.fixed5(validation.top1()),
        FormatUtils.fixed5(validation.dahaiTop1()),
        FormatUtils.fixed5(validation.riichiTop1()),
        FormatUtils.fixed5(validation.reactionTop1()),
        FormatUtils.fixed5(validation.postCallDahaiTop1()),
        validation.dahaiSamples(),
        validation.riichiSamples(),
        validation.reactionSamples(),
        validation.postCallDahaiSamples(),
        FormatUtils.fixed5(validation.nll()),
        FormatUtils.fixed5(validation.postCallDahaiNll()),
        FormatUtils.fixed5(validation.valueTeacherBias()),
        FormatUtils.fixed5(validation.valueTeacherMse()),
        FormatUtils.fixed5(validation.valueBias()),
        FormatUtils.fixed5(validation.valueMse()),
        FormatUtils.fixed5(validation.valueConstantMse()),
        validation.boundarySamples(),
        FormatUtils.fixed5(validation.grpBoundaryOutcomeNll()),
        FormatUtils.fixed5(validation.grpBoundaryRps()),
        FormatUtils.fixed5(validation.policyUniformNll()),
        FormatUtils.fixed5(validation.policyStateCenteredRms()),
        FormatUtils.fixed5(validation.valueStateCenteredRms()),
        FormatUtils.fixed5(validation.valueUtilityCenteredRms()));
  }

  private void logCheckpointValidation(
      Path checkpoint, EpsilonDecisionPretrainValidator.ValidationMetrics metrics) {
    log.info(
        "Decision pretrain checkpoint validation: checkpoint={} files={} samples={} "
            + "policySamples={} top1={} dahaiTop1={} riichiTop1={} reactionTop1={} "
            + "postCallDahaiTop1={} dahaiSamples={} riichiSamples={} reactionSamples={} "
            + "postCallDahaiSamples={} nll={} postCallDahaiNll={} "
            + "valueTeacherBias={} valueTeacherMse={} valueBias={} valueMse={} valueConstantMse={} "
            + "boundarySamples={} grpBoundaryOutcomeNll={} grpBoundaryRps={} "
            + "policyUniformNll={} "
            + "policyStateCenteredRms={} valueStateCenteredRms={} "
            + "valueUtilityCenteredRms={}",
        checkpoint,
        metrics.files(),
        metrics.samples(),
        metrics.policySamples(),
        FormatUtils.fixed5(metrics.top1()),
        FormatUtils.fixed5(metrics.dahaiTop1()),
        FormatUtils.fixed5(metrics.riichiTop1()),
        FormatUtils.fixed5(metrics.reactionTop1()),
        FormatUtils.fixed5(metrics.postCallDahaiTop1()),
        metrics.dahaiSamples(),
        metrics.riichiSamples(),
        metrics.reactionSamples(),
        metrics.postCallDahaiSamples(),
        FormatUtils.fixed5(metrics.nll()),
        FormatUtils.fixed5(metrics.postCallDahaiNll()),
        FormatUtils.fixed5(metrics.valueTeacherBias()),
        FormatUtils.fixed5(metrics.valueTeacherMse()),
        FormatUtils.fixed5(metrics.valueBias()),
        FormatUtils.fixed5(metrics.valueMse()),
        FormatUtils.fixed5(metrics.valueConstantMse()),
        metrics.boundarySamples(),
        FormatUtils.fixed5(metrics.grpBoundaryOutcomeNll()),
        FormatUtils.fixed5(metrics.grpBoundaryRps()),
        FormatUtils.fixed5(metrics.policyUniformNll()),
        FormatUtils.fixed5(metrics.policyStateCenteredRms()),
        FormatUtils.fixed5(metrics.valueStateCenteredRms()),
        FormatUtils.fixed5(metrics.valueUtilityCenteredRms()));
  }

  private SourceSplit splitSourceFiles(List<Path> files, float validationFraction) {
    ArrayList<Path> trainingFiles = new ArrayList<>();
    ArrayList<Path> validationFiles = new ArrayList<>();
    for (Path file : files) {
      (isValidationFile(file, validationFraction) ? validationFiles : trainingFiles).add(file);
    }
    if (trainingFiles.isEmpty()) {
      return new SourceSplit(List.copyOf(files), List.of());
    }
    return new SourceSplit(List.copyOf(trainingFiles), List.copyOf(validationFiles));
  }

  /** ファイル名の安定ハッシュで決定的にホールドアウトへ振り分ける。 */
  public static boolean isValidationFile(Path file, float validationFraction) {
    if (validationFraction <= 0.0f) {
      return false;
    }
    int bucket = Math.floorMod(file.getFileName().toString().hashCode(), 10_000);
    return bucket < Math.round(validationFraction * 10_000.0f);
  }

  private EpsilonGrpTrainingSession openGrpSession(Path checkpointDir, boolean teacherEnabled)
      throws IOException {
    if (!teacherEnabled) {
      return EpsilonGrpTrainingSession.disabled(checkpointDir.resolve("grp"));
    }
    return EpsilonGrpTrainingSession.openInferenceOnly(
        checkpointDir,
        NetworkFactory.getGrpDevices(config.bind(com.epsilon.config.settings.DeviceSettings.class))
            .primary(),
        config);
  }

  private void addTrainingMetrics(
      TrainTotals totals, EpsilonDecisionPretrainer.EpochMetrics metrics) {
    if (metrics.optimizerSteps() <= 0) {
      return;
    }
    int batches = metrics.optimizerSteps();
    int policyBatches = metrics.policyOptimizerSteps();
    totals.batches += batches;
    totals.policyBatches += policyBatches;
    totals.samplesSeen += metrics.rows();
    totals.policySamplesSeen += metrics.policyRows();
    totals.loss += (float) metrics.loss() * batches;
    totals.behaviorCloningLoss += (float) metrics.behaviorCloningLoss() * policyBatches;
    totals.valueLoss += (float) metrics.valueLoss() * batches;
    totals.entropy += (float) metrics.entropy() * policyBatches;
    totals.chosenProb += (float) metrics.chosenProbability() * policyBatches;
  }

  private float avg(float sum, int count) {
    return count == 0 ? 0.0f : sum / count;
  }

  private record SourceSplit(List<Path> trainingFiles, List<Path> validationFiles) {}

  private record EpochArtifact(int epoch, Path checkpoint) {}

  private record TrainingRun(
      List<EpochArtifact> epochs, TrainTotals totals, int globalStep, int iteration) {}

  /**
   * 保存済み事前学習チェックポイントの副作用なし検証結果。
   *
   * @param checkpoint 評価した変更不可チェックポイント
   * @param usable 十分な検証用行があり全指標が有限ならtrue
   * @param assessment 評価成立条件と診断値
   * @param metrics 検証用方策／価値指標
   */
  public record ValidationSummary(
      Path checkpoint,
      boolean usable,
      String assessment,
      EpsilonDecisionPretrainValidator.ValidationMetrics metrics) {}

  /** 一つの事前学習エポックチェックポイントに対する副作用なし検証結果。 */
  public record CheckpointValidation(
      int epoch,
      Path checkpoint,
      boolean usable,
      String assessment,
      EpsilonDecisionPretrainValidator.ValidationMetrics metrics) {}

  /**
   * Decision 牌譜事前学習の保存結果。
   *
   * @param samples 処理した全サンプル数
   * @param batches オプティマイザーバッチ数
   * @param globalStep 保存後チェックポイントの累積更新回数
   * @param iteration 保存後チェックポイントの反復回数
   * @param outputCheckpoint 最後の構造的に利用可能なチェックポイント。対局実行処理参照は変更しない
   * @param validations 各エポックチェックポイントの検証用評価
   */
  public record TrainSummary(
      int samples,
      int batches,
      int globalStep,
      int iteration,
      Path outputCheckpoint,
      List<CheckpointValidation> validations) {

    public TrainSummary {
      validations = List.copyOf(validations);
    }
  }

  private static final class TrainTotals {
    int samplesSeen;
    int policySamplesSeen;
    int batches;
    int policyBatches;
    float loss;
    float behaviorCloningLoss;
    float valueLoss;
    float entropy;
    float chosenProb;

    void add(TrainTotals other) {
      samplesSeen += other.samplesSeen;
      policySamplesSeen += other.policySamplesSeen;
      batches += other.batches;
      policyBatches += other.policyBatches;
      loss += other.loss;
      behaviorCloningLoss += other.behaviorCloningLoss;
      valueLoss += other.valueLoss;
      entropy += other.entropy;
      chosenProb += other.chosenProb;
    }
  }
}
