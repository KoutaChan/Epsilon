package com.epsilon.pico.ai.belief;

import ai.djl.Model;
import com.epsilon.ai.belief.EpsilonBeliefSample;
import com.epsilon.ai.belief.EpsilonBeliefTrainer;
import com.epsilon.config.settings.BeliefPretrainSettings;
import com.epsilon.config.settings.BeliefSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import com.epsilon.pico.ai.network.NetworkFactory;
import com.epsilon.pico.training.EpsilonLogPretrainDataCollector;
import com.epsilon.util.FormatUtils;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 天鳳・mjson 形式の牌譜から、公開情報を入力とする Belief モデルを事前学習する。 */
public final class EpsilonBeliefLogPretrainRunner {

  private static final Logger log = LoggerFactory.getLogger(EpsilonBeliefLogPretrainRunner.class);

  private EpsilonBeliefLogPretrainRunner() {}

  /**
   * 対応牌譜をまとまり単位で収集し、Belief モデルを指定エポック数事前学習する。
   *
   * @param checkpointDir 既存チェックポイントの探索先兼保存先
   * @param logDir 入力元牌譜ディレクトリ
   * @param epochs 全入力元を走査するエポック数
   * @param maxFiles 使用する入力元ファイル上限。収集処理の規約に従う
   * @return 処理ファイル数・サンプル数・損失・保存位置の要約
   * @throws Exception 牌譜収集、学習、チェックポイント保存に失敗した場合
   */
  public static TrainSummary pretrainFromLogs(
      Path checkpointDir, Path logDir, int epochs, int maxFiles) throws Exception {
    return pretrainFromLogs(
        checkpointDir,
        logDir,
        epochs,
        maxFiles,
        com.epsilon.pico.config.settings.EpsilonSettings.defaults());
  }

  public static TrainSummary pretrainFromLogs(
      Path checkpointDir, Path logDir, int epochs, int maxFiles, SettingsLoader settings)
      throws Exception {
    if (epochs <= 0) {
      throw new IllegalArgumentException("epochs must be positive");
    }
    List<Path> files = EpsilonLogPretrainDataCollector.listLogFiles(logDir, maxFiles);
    if (files.isEmpty()) {
      throw new IllegalStateException(
          "No supported Belief pretrain log files were found in logDir=" + logDir);
    }
    BeliefPretrainSettings pretrainSettings = settings.bind(BeliefPretrainSettings.class);
    int fileChunkSize = pretrainSettings.logChunkFiles();
    int sampleChunkSize = pretrainSettings.logSampleChunkSize();

    Path checkpoint = EpsilonBeliefCheckpointManager.resolveExisting(checkpointDir);
    int globalStep = 0;
    int iteration = 0;
    Model model;
    if (checkpoint == null) {
      model =
          NetworkFactory.createBeliefModel(
              NetworkFactory.getLearnerDevice(
                  settings.bind(com.epsilon.config.settings.DeviceSettings.class)),
              settings.bind(com.epsilon.config.settings.BeliefSettings.class).hidden());
      log.info("Belief log pretrain created a new model");
    } else {
      EpsilonBeliefCheckpointBundle manifest =
          EpsilonBeliefCheckpointManager.loadManifest(checkpoint);
      globalStep = manifest.globalStep;
      iteration = manifest.iteration;
      model =
          EpsilonBeliefCheckpointManager.load(
              checkpoint,
              NetworkFactory.getLearnerDevice(
                  settings.bind(com.epsilon.config.settings.DeviceSettings.class)));
      log.info(
          "Belief log pretrain loaded checkpoint: checkpoint={} step={} iteration={}",
          checkpoint,
          globalStep,
          iteration);
    }

    BeliefSettings beliefSettings = settings.bind(BeliefSettings.class);
    TrainTotals allEpochs = new TrainTotals();
    try (model;
        EpsilonBeliefTrainer<DecisionHostBatch> trainer =
            new EpsilonBeliefTrainer<>(
                model,
                new BeliefInputs(),
                pretrainSettings.batchSize(),
                beliefSettings.learningRate(),
                beliefSettings.weightDecay(),
                beliefSettings.gradClip())) {
      log.info(
          "Belief log pretrain update mode: epochsPerChunk=1 batchSize={} logChunkFiles={} "
              + "sampleChunkSize={}",
          pretrainSettings.batchSize(),
          fileChunkSize,
          sampleChunkSize);
      for (int epoch = 1; epoch <= epochs; epoch++) {
        TrainTotals epochTotals = new TrainTotals();
        List<EpsilonBeliefSample<DecisionHostBatch>> sampleBuffer =
            new ArrayList<>(sampleChunkSize);
        int skippedFiles = 0;
        int processedFiles = 0;

        for (int start = 0; start < files.size(); start += fileChunkSize) {
          int end = Math.min(start + fileChunkSize, files.size());
          for (int i = start; i < end; i++) {
            Path file = files.get(i);
            List<EpsilonBeliefSample<DecisionHostBatch>> samples;
            try {
              samples = EpsilonLogPretrainDataCollector.collectBeliefFile(file);
            } catch (Exception e) {
              skippedFiles++;
              log.warn(
                  "Failed to load Belief pretrain log: file={} error={}", file, e.getMessage());
              continue;
            }
            if (samples.isEmpty()) {
              skippedFiles++;
              log.warn("Skipped Belief pretrain log with no samples: {}", file);
              continue;
            }
            appendAndTrain(samples, sampleBuffer, sampleChunkSize, trainer, epochTotals);
            processedFiles++;
          }
        }

        flush(sampleBuffer, trainer, epochTotals);
        if (epochTotals.samples == 0) {
          throw new IllegalStateException("No Belief samples were produced from logDir=" + logDir);
        }
        globalStep += epochTotals.batches;
        iteration++;
        Path epochDir =
            checkpointDir.resolve("belief_pretrain_logs_" + FormatUtils.zeroPad(iteration, 5));
        EpsilonBeliefCheckpointManager.save(model, epochDir, globalStep, iteration);
        EpsilonBeliefCheckpointManager.save(
            model, EpsilonBeliefCheckpointManager.latest(checkpointDir), globalStep, iteration);
        epochTotals.files = processedFiles;
        epochTotals.skippedFiles = skippedFiles;
        allEpochs.add(epochTotals);
        log.info(
            "Belief log pretrain epoch complete: epoch={} files={} skippedFiles={} "
                + "samples={} batches={} loss={} handLoss={} "
                + "waitLoss={} scalarLoss={}",
            epoch,
            processedFiles,
            skippedFiles,
            epochTotals.samples,
            epochTotals.batches,
            FormatUtils.fixed5(avg(epochTotals.loss, epochTotals.batches)),
            FormatUtils.fixed5(avg(epochTotals.handLoss, epochTotals.batches)),
            FormatUtils.fixed5(avg(epochTotals.waitLoss, epochTotals.batches)),
            FormatUtils.fixed5(avg(epochTotals.scalarLoss, epochTotals.batches)));
      }
    }
    return new TrainSummary(
        allEpochs.files,
        allEpochs.skippedFiles,
        allEpochs.samples,
        allEpochs.batches,
        globalStep,
        iteration,
        avg(allEpochs.loss, allEpochs.batches),
        avg(allEpochs.handLoss, allEpochs.batches),
        avg(allEpochs.waitLoss, allEpochs.batches),
        avg(allEpochs.scalarLoss, allEpochs.batches));
  }

  private static void appendAndTrain(
      List<EpsilonBeliefSample<DecisionHostBatch>> samples,
      List<EpsilonBeliefSample<DecisionHostBatch>> sampleBuffer,
      int sampleChunkSize,
      EpsilonBeliefTrainer<DecisionHostBatch> trainer,
      TrainTotals totals) {
    for (EpsilonBeliefSample<DecisionHostBatch> sample : samples) {
      sampleBuffer.add(sample);
      if (sampleBuffer.size() >= sampleChunkSize) {
        flush(sampleBuffer, trainer, totals);
      }
    }
  }

  private static void flush(
      List<EpsilonBeliefSample<DecisionHostBatch>> sampleBuffer,
      EpsilonBeliefTrainer<DecisionHostBatch> trainer,
      TrainTotals totals) {
    if (sampleBuffer.isEmpty()) {
      return;
    }
    EpsilonBeliefTrainer.TrainMetrics metrics = trainer.train(sampleBuffer, 1);
    totals.samples += metrics.samples();
    totals.batches += metrics.batches();
    totals.loss += metrics.loss() * metrics.batches();
    totals.handLoss += metrics.handLoss() * metrics.batches();
    totals.waitLoss += metrics.waitLoss() * metrics.batches();
    totals.scalarLoss += metrics.scalarLoss() * metrics.batches();
    sampleBuffer.clear();
  }

  private static float avg(float sum, int count) {
    return sum / count;
  }

  /**
   * Belief 牌譜事前学習全体の集約結果。
   *
   * @param files 正常に処理した牌譜ファイル数
   * @param skippedFiles 入力不良などで除外したファイル数
   * @param samples 学習サンプル数
   * @param batches オプティマイザーバッチ数
   * @param globalStep 保存後チェックポイントの累積更新回数
   * @param iteration 保存後チェックポイントの反復回数
   * @param loss 合計損失
   * @param handLoss 他家手牌分布損失
   * @param waitLoss 他家待ち分布損失
   * @param scalarLoss シャンテン・テンパイスカラー損失
   */
  public record TrainSummary(
      int files,
      int skippedFiles,
      int samples,
      int batches,
      int globalStep,
      int iteration,
      float loss,
      float handLoss,
      float waitLoss,
      float scalarLoss) {}

  private static final class TrainTotals {
    int files;
    int skippedFiles;
    int samples;
    int batches;
    float loss;
    float handLoss;
    float waitLoss;
    float scalarLoss;

    void add(TrainTotals other) {
      files += other.files;
      skippedFiles += other.skippedFiles;
      samples += other.samples;
      batches += other.batches;
      loss += other.loss;
      handLoss += other.handLoss;
      waitLoss += other.waitLoss;
      scalarLoss += other.scalarLoss;
    }
  }
}
