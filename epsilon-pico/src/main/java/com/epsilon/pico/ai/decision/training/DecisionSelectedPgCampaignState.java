package com.epsilon.pico.ai.decision.training;

import com.epsilon.config.settings.DecisionChampionDuelSettings;
import com.epsilon.pico.ai.decision.arena.DecisionAdaptiveExploration;
import com.epsilon.pico.ai.decision.audit.EpsilonDecisionSelectedPgDebugAudit;
import com.epsilon.pico.ai.decision.data.DecisionJsonFiles;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** 最後に保存を確定した学習状態と、次に実行する収集・更新・評価処理を保存する。 */
record DecisionSelectedPgCampaignState(
    int version,
    DecisionSelectedPgRunContext run,
    int globalStep,
    int iteration,
    int selfPlayGames,
    int completedMacros,
    int games,
    long samples,
    int lineageSequence,
    Path arenaChampion,
    Lineage lineage,
    Interval interval,
    DecisionAdaptiveExploration.State exploration,
    DecisionSelectedPgDuelResult.Status terminalStatus) {

  static final String FILE_NAME = "campaign-state.json";
  static final Gson GSON =
      new GsonBuilder()
          .serializeNulls()
          .setPrettyPrinting()
          .registerTypeHierarchyAdapter(
              Path.class,
              new TypeAdapter<Path>() {
                @Override
                public void write(JsonWriter out, Path value) throws IOException {
                  if (value == null) out.nullValue();
                  else out.value(value.toString());
                }

                @Override
                public Path read(JsonReader in) throws IOException {
                  if (in.peek() == com.google.gson.stream.JsonToken.NULL) {
                    in.nextNull();
                    return null;
                  }
                  return Path.of(in.nextString());
                }
              })
          .create();

  static DecisionSelectedPgCampaignState read(Path checkpoint) throws IOException {
    Path file = checkpoint.resolve(FILE_NAME);
    if (!Files.isRegularFile(file)) return null;
    DecisionSelectedPgCampaignState state = readJson(file, DecisionSelectedPgCampaignState.class);
    EpsilonDecisionCheckpointBundle model =
        EpsilonDecisionCheckpointManager.requireValidCheckpoint(checkpoint);
    if (state.version != 1
        || state.globalStep != model.globalStep
        || state.iteration != model.iteration
        || state.selfPlayGames != model.selfPlayGames) {
      throw new IOException("Campaign state and learner generation differ: " + checkpoint);
    }
    return state;
  }

  void write(Path checkpoint) throws IOException {
    writeJson(checkpoint.resolve(FILE_NAME), this);
  }

  static <T> T readJson(Path file, Class<T> type) throws IOException {
    try {
      return GSON.fromJson(Files.readString(file), type);
    } catch (RuntimeException invalid) {
      throw new IOException("Invalid selected-PG state: " + file, invalid);
    }
  }

  static void writeJson(Path file, Object value) throws IOException {
    DecisionJsonFiles.write(file, value, GSON);
  }

  record Lineage(
      int number,
      int candidateIteration,
      Path directory,
      Path immutableParent,
      Path parentLearner,
      long seedBase,
      int completedMacros,
      int duelRound) {}

  record Interval(
      int duelRound,
      Path directory,
      long seedBase,
      long[][] opponentIds,
      int plannedMacros,
      DecisionChampionDuelSettings duelSettings,
      int games,
      long samples,
      int completedMacros,
      String guardFailure,
      DecisionTrainingResult lastActorMetrics,
      DecisionTrainingResult lastValueMetrics,
      EpsilonDecisionSelectedPgDebugAudit.Result lastAudit,
      DecisionActorLearningControl.Adjustment lastLearningRateAdjustment,
      Path candidate) {}
}
