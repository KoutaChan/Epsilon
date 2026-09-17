package com.epsilon.pico.ai.decision.training;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.pico.config.settings.DecisionSelectedPgCampaignSettings.OptimizerSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/** 学習対象のモデル、AdamW、方策更新の KL 制御を一組として管理する。 */
final class DecisionLearner implements AutoCloseable {

  private final Device device;
  private final SettingsLoader config;
  private final DecisionExecutionContext executionContext;
  private final OptimizerSettings optimizerSettings;
  private final String learnerContractId;
  private Model model;
  private EpsilonDecisionTrainer trainer;
  private DecisionActorLearningControl controller;
  private int optimizerReinitializations;
  private int optimizerRestorations;

  DecisionLearner(
      Model model,
      Device device,
      OptimizerSettings optimizerSettings,
      String learnerContractId,
      DecisionExecutionContext executionContext,
      SettingsLoader config) {
    this(
        model,
        device,
        optimizerSettings,
        learnerContractId,
        new DecisionActorLearningControl(optimizerSettings.actorKlControl()),
        executionContext,
        config);
  }

  private DecisionLearner(
      Model model,
      Device device,
      OptimizerSettings settings,
      String contract,
      DecisionActorLearningControl control,
      DecisionExecutionContext executionContext,
      SettingsLoader config) {
    this.config = config;
    this.device = Objects.requireNonNull(device, "device");
    this.executionContext = executionContext;
    this.optimizerSettings = Objects.requireNonNull(settings, "settings");
    if (contract == null || contract.isBlank()) {
      throw new IllegalArgumentException("Decision learnerContractId must not be blank");
    }
    this.learnerContractId = contract;
    this.model = Objects.requireNonNull(model, "model");
    this.controller = control;
    this.trainer =
        new EpsilonDecisionTrainer(
            model, control, settings.valueLearningRate(), contract, executionContext, config);
  }

  SettingsLoader config() {
    return config;
  }

  Model model() {
    return model;
  }

  EpsilonDecisionTrainer trainer() {
    return trainer;
  }

  float actorLearningRate() {
    return controller.currentLearningRate();
  }

  float valueLearningRate() {
    return optimizerSettings.valueLearningRate();
  }

  DecisionActorLearningControl.State controllerState() {
    return controller.state();
  }

  DecisionActorLearningControl.Adjustment acceptMacroKl(
      float observedMeanKl, int acceptedOptimizerSteps) {
    return controller.accept(observedMeanKl, acceptedOptimizerSteps);
  }

  DecisionActorLearningControl.Adjustment rejectMacroKl(float observedMeanKl) {
    return controller.reject(observedMeanKl);
  }

  int optimizerReinitializations() {
    return optimizerReinitializations;
  }

  int optimizerRestorations() {
    return optimizerRestorations;
  }

  void saveCheckpointState(Path checkpoint, EpsilonDecisionCheckpointBundle modelCheckpoint)
      throws IOException {
    DecisionLearnerCheckpoint.save(checkpoint, modelCheckpoint, trainer, controller.state());
    DecisionLearnerCheckpoint.requireIfPresent(
        checkpoint,
        modelCheckpoint,
        optimizerSettings.valueLearningRate(),
        learnerContractId,
        optimizerSettings.actorKlControl(),
        config.bind(com.epsilon.pico.config.settings.DecisionSettings.class));
  }

  /** 同じモデルを重複loadせず、保存された学習器を直接開く。 */
  static DecisionLearner openWorking(
      Path checkpoint,
      Device device,
      OptimizerSettings settings,
      String contract,
      DecisionExecutionContext executionContext,
      SettingsLoader config)
      throws IOException {
    EpsilonDecisionCheckpointBundle manifest =
        EpsilonDecisionCheckpointManager.requireValidCheckpoint(checkpoint);
    DecisionActorLearningControl.State state =
        DecisionLearnerCheckpoint.requireIfPresent(
            checkpoint,
            manifest,
            settings.valueLearningRate(),
            contract,
            settings.actorKlControl(),
            config.bind(com.epsilon.pico.config.settings.DecisionSettings.class));
    if (state == null) {
      throw new IOException("Decision learner checkpoint not found: " + checkpoint);
    }
    Model model = EpsilonDecisionCheckpointManager.load(checkpoint, device);
    DecisionLearner learner = null;
    try {
      learner =
          new DecisionLearner(
              model,
              device,
              settings,
              contract,
              new DecisionActorLearningControl(settings.actorKlControl(), state),
              executionContext,
              config);
      learner.trainer.loadOptimizerStates(checkpoint);
      learner.optimizerRestorations = 1;
      return learner;
    } catch (Exception | Error failure) {
      if (learner == null) model.close();
      else learner.close();
      throw failure;
    }
  }

  void replanTargetKl(long endOptimizerStep, float endTargetKl) {
    controller.replan(endOptimizerStep, endTargetKl);
  }

  void holdTargetKl(float targetKl) {
    controller.holdTarget(targetKl);
  }

  /** rolling 更新中のチェックポイントからモデル、AdamW、KL 制御を一組として復元する。 */
  void reloadWorking(Path checkpoint) throws IOException {
    int nextRestorations = Math.addExact(optimizerRestorations, 1);
    try (DecisionLearner restored =
        openWorking(
            checkpoint, device, optimizerSettings, learnerContractId, executionContext, config)) {
      closeCurrent();
      model = restored.model;
      trainer = restored.trainer;
      controller = restored.controller;
      restored.model = null;
      restored.trainer = null;
      optimizerRestorations = nextRestorations;
    }
  }

  @Override
  public void close() {
    closeCurrent();
  }

  private void closeCurrent() {
    try (Model closingModel = model;
        EpsilonDecisionTrainer closingTrainer = trainer) {
      model = null;
      trainer = null;
    }
  }
}
