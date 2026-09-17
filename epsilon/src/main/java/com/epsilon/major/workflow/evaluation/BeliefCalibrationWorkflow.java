package com.epsilon.major.workflow.evaluation;

import ai.djl.Model;
import com.epsilon.ai.belief.EpsilonBeliefCalibration;
import com.epsilon.ai.belief.EpsilonBeliefInferenceServer;
import com.epsilon.ai.belief.EpsilonBeliefSample;
import com.epsilon.config.settings.BeliefInferenceSettings;
import com.epsilon.major.ai.belief.BeliefInputs;
import com.epsilon.major.ai.belief.EpsilonBeliefCheckpointManager;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.training.EpsilonLogPretrainDataCollector;
import com.epsilon.workflow.BeliefCalibrationStatus;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 牌譜サンプルで Belief チェックポイントの校正を評価するワークフロー。 */
public final class BeliefCalibrationWorkflow
    implements CommandWorkflow<BeliefCalibrationWorkflow.Result> {

  private static final Logger log = LoggerFactory.getLogger(BeliefCalibrationWorkflow.class);

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.CALIBRATE_BELIEF_LOGS;
  }

  @Override
  public Result execute(WorkflowArguments arguments) throws Exception {
    Path directory = arguments.optionalPath(0, Path.of("checkpoints/epsilon/belief"));
    Path logDirectory = arguments.optionalPath(1, Path.of("data/logs"));
    int maxFiles = arguments.optionalInteger(2, "maxFiles", 0);
    int maxSamples = arguments.optionalInteger(3, "maxSamples", 20_000);
    Path checkpoint = EpsilonBeliefCheckpointManager.resolveExisting(directory);
    if (checkpoint == null) {
      return Result.failure(BeliefCalibrationStatus.CHECKPOINT_NOT_FOUND, 0);
    }
    List<Path> files = EpsilonLogPretrainDataCollector.listLogFiles(logDirectory, maxFiles);
    if (files.isEmpty()) {
      return Result.failure(BeliefCalibrationStatus.NO_SUPPORTED_LOG_FILES, 0);
    }
    List<EpsilonBeliefSample<DecisionHostBatch>> samples = new ArrayList<>();
    for (Path file : files) {
      if (samples.size() >= maxSamples) {
        break;
      }
      try {
        samples.addAll(EpsilonLogPretrainDataCollector.collectBeliefFile(file));
      } catch (Exception error) {
        log.warn(
            "Failed to load Belief calibration log: file={} error={}", file, error.getMessage());
      }
    }
    if (samples.size() > maxSamples) {
      samples = samples.subList(0, maxSamples);
    }
    if (samples.isEmpty()) {
      return Result.failure(BeliefCalibrationStatus.NO_BELIEF_SAMPLES, 0);
    }
    try (Model model =
            EpsilonBeliefCheckpointManager.load(
                checkpoint,
                com.epsilon.major.ai.network.NetworkFactory.getInferenceDevices(
                        arguments.settings().bind(com.epsilon.config.settings.DeviceSettings.class))
                    .primary());
        EpsilonBeliefInferenceServer<DecisionHostBatch> server =
            new EpsilonBeliefInferenceServer<>(
                model,
                arguments.settings().bind(BeliefInferenceSettings.class).maxBatch(),
                new BeliefInputs())) {
      return Result.success(
          checkpoint, samples.size(), EpsilonBeliefCalibration.evaluate(samples, server));
    }
  }

  @Override
  public String summarize(Result result) {
    if (!result.successful()) {
      return "status=" + result.status() + " samples=" + result.samples();
    }
    return "status=calibrated checkpoint="
        + result.checkpoint()
        + " samples="
        + result.samples()
        + " report="
        + result.report().summary();
  }

  @Override
  public boolean successful(Result result) {
    return result.successful();
  }

  public record Result(
      boolean successful,
      BeliefCalibrationStatus status,
      Path checkpoint,
      int samples,
      EpsilonBeliefCalibration.Report report) {

    static Result failure(BeliefCalibrationStatus status, int samples) {
      return new Result(false, status, null, samples, null);
    }

    static Result success(Path checkpoint, int samples, EpsilonBeliefCalibration.Report report) {
      return new Result(true, BeliefCalibrationStatus.CALIBRATED, checkpoint, samples, report);
    }
  }
}
