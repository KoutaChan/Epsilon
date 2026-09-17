package com.epsilon.major.ai.decision.training;

import com.epsilon.major.ai.decision.EpsilonDecisionConstants;
import com.epsilon.major.config.settings.DecisionSelectedPgCampaignSettings.ActorKlControlSettings;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** 学習再開用チェックポイントに保存する AdamW と KL 制御状態の形式を定義する。 */
final class DecisionLearnerCheckpoint {

  static final String STATE_FILE = "learner-state.json";
  static final String ACTOR_STATE_FILE = "actor-adamw.state";
  static final String VALUE_STATE_FILE = "value-adamw.state";

  private static final String RETIRED_OPTIMIZER_MANIFEST = "optimizer.json";
  private static final int VERSION = 1;
  private static final String FORMAT = "selected-pg-learner-state-v1";
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  private DecisionLearnerCheckpoint() {}

  static Path actorState(Path checkpoint) {
    return checkpoint.resolve(ACTOR_STATE_FILE);
  }

  static Path valueState(Path checkpoint) {
    return checkpoint.resolve(VALUE_STATE_FILE);
  }

  static void save(
      Path checkpoint,
      EpsilonDecisionCheckpointBundle model,
      EpsilonDecisionTrainer trainer,
      DecisionActorLearningControl.State controllerState)
      throws IOException {
    trainer.saveOptimizerStates(checkpoint);
    Manifest manifest =
        new Manifest(
            VERSION,
            FORMAT,
            EpsilonDecisionConstants.ARCHITECTURE_ID,
            model.globalStep,
            model.iteration,
            model.selfPlayGames,
            trainer.learnerContractId(),
            trainer.valueLearningRate(),
            trainer.weightDecay(),
            trainer.gradientClip(),
            ACTOR_STATE_FILE,
            VALUE_STATE_FILE,
            controllerState);
    Files.writeString(checkpoint.resolve(STATE_FILE), GSON.toJson(manifest));
  }

  /** 付随情報が無ければnullを返し、存在する場合は全保存契約を厳格に検証する。 */
  static DecisionActorLearningControl.State requireIfPresent(
      Path checkpoint,
      EpsilonDecisionCheckpointBundle model,
      float expectedValueLearningRate,
      String expectedLearnerContractId,
      ActorKlControlSettings controllerSettings,
      com.epsilon.major.config.settings.DecisionSettings optimizerSettings)
      throws IOException {
    if (Files.exists(checkpoint.resolve(RETIRED_OPTIMIZER_MANIFEST))) {
      throw new IOException(
          "Unsupported retired Decision optimizer checkpoint: "
              + checkpoint.resolve(RETIRED_OPTIMIZER_MANIFEST)
              + "; start from an immutable Arena model with a fresh learner state");
    }
    List<Path> artifacts = artifacts(checkpoint);
    long present = artifacts.stream().filter(Files::isRegularFile).count();
    if (present == 0L) {
      return null;
    }
    if (present != artifacts.size()) {
      throw new IOException(
          "Incomplete Decision learner checkpoint: "
              + checkpoint
              + " files="
              + present
              + "/"
              + artifacts.size());
    }

    Manifest manifest = loadManifest(checkpoint.resolve(STATE_FILE));
    if (manifest.version() != VERSION || !FORMAT.equals(manifest.format())) {
      throw new IOException(
          "Unsupported Decision learner checkpoint: version="
              + manifest.version()
              + " format="
              + manifest.format());
    }
    if (!EpsilonDecisionConstants.ARCHITECTURE_ID.equals(manifest.architecture())) {
      throw new IOException("Decision learner architecture mismatch: " + manifest.architecture());
    }
    if (manifest.globalStep() != model.globalStep
        || manifest.iteration() != model.iteration
        || manifest.selfPlayGames() != model.selfPlayGames) {
      throw new IOException(
          "Decision learner/model generation mismatch: learner="
              + manifest.globalStep()
              + "/"
              + manifest.iteration()
              + "/"
              + manifest.selfPlayGames()
              + " model="
              + model.globalStep
              + "/"
              + model.iteration
              + "/"
              + model.selfPlayGames);
    }
    if (!expectedLearnerContractId.equals(manifest.learnerContractId())) {
      throw new IOException(
          "Decision learner contract mismatch: saved="
              + manifest.learnerContractId()
              + " current="
              + expectedLearnerContractId);
    }
    if (Float.compare(manifest.valueLearningRate(), expectedValueLearningRate) != 0
        || Float.compare(manifest.weightDecay(), optimizerSettings.weightDecay()) != 0
        || Float.compare(manifest.gradientClip(), optimizerSettings.gradClip()) != 0) {
      throw new IOException(
          "Decision learner optimizer configuration mismatch: saved="
              + manifest.valueLearningRate()
              + "/"
              + manifest.weightDecay()
              + "/"
              + manifest.gradientClip()
              + " current="
              + expectedValueLearningRate
              + "/"
              + optimizerSettings.weightDecay()
              + "/"
              + optimizerSettings.gradClip());
    }
    if (!ACTOR_STATE_FILE.equals(manifest.actorStateFile())
        || !VALUE_STATE_FILE.equals(manifest.valueStateFile())) {
      throw new IOException("Unsupported Decision learner optimizer state file names");
    }
    if (Files.size(actorState(checkpoint)) == 0L || Files.size(valueState(checkpoint)) == 0L) {
      throw new IOException("Decision learner optimizer state file is empty: " + checkpoint);
    }
    try {
      DecisionActorLearningControl.validateRestored(controllerSettings, manifest.controllerState());
    } catch (IllegalArgumentException | NullPointerException error) {
      throw new IOException("Invalid Decision Actor KL controller state", error);
    }
    return manifest.controllerState();
  }

  static List<Path> artifacts(Path checkpoint) {
    return List.of(checkpoint.resolve(STATE_FILE), actorState(checkpoint), valueState(checkpoint));
  }

  private static Manifest loadManifest(Path path) throws IOException {
    try {
      JsonObject json = JsonParser.parseString(Files.readString(path)).getAsJsonObject();
      requireFields(
          json,
          "version",
          "format",
          "architecture",
          "globalStep",
          "iteration",
          "selfPlayGames",
          "learnerContractId",
          "valueLearningRate",
          "weightDecay",
          "gradientClip",
          "actorStateFile",
          "valueStateFile",
          "controllerState");
      Manifest manifest = GSON.fromJson(json, Manifest.class);
      if (manifest == null) {
        throw new IllegalArgumentException("empty learner manifest");
      }
      return manifest;
    } catch (RuntimeException error) {
      throw new IOException("Invalid Decision learner manifest: " + path, error);
    }
  }

  private static void requireFields(JsonObject json, String... fields) {
    for (String field : fields) {
      if (!json.has(field) || json.get(field).isJsonNull()) {
        throw new IllegalArgumentException("Missing Decision learner field: " + field);
      }
    }
  }

  private record Manifest(
      int version,
      String format,
      String architecture,
      int globalStep,
      int iteration,
      int selfPlayGames,
      String learnerContractId,
      float valueLearningRate,
      float weightDecay,
      float gradientClip,
      String actorStateFile,
      String valueStateFile,
      DecisionActorLearningControl.State controllerState) {}
}
