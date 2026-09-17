package com.epsilon.pico.ai.decision.benchmark;

import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.pico.ai.decision.input.DecisionBucket;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionEvaluatorFactory;
import com.epsilon.pico.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.pico.config.settings.DecisionInferenceSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.runtime.InferenceProfile;
import com.google.gson.GsonBuilder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/** 固定入力データの1つの形状を GPU ごとに直列再生し、指定した演算区間の所要時間を測る。 */
public final class EpsilonDecisionInferenceProfileBenchmark {

  private static final int WARMUP_BATCHES = 50;
  private static final List<String> SECTIONS =
      List.of(
          "host.stage",
          "transfer.h2d",
          "transfer.numeric_cast",
          "network.forward",
          "state.project.round",
          "state.project.player",
          "state.project.tile",
          "state.project.river",
          "state.project.meld",
          "state.entity_concat",
          "policy.player_tile_context",
          "policy.prefix_pack",
          "policy.prefix_linear",
          "policy.root_scorer",
          "transfer.d2h",
          "host.compose");

  private EpsilonDecisionInferenceProfileBenchmark() {}

  /** 同じ固定入力データに保存された方策または対戦相手の入力とモデル出力を選ぶ。 */
  public enum Role {
    ACTOR,
    OPPONENT;

    public static Role parse(String text) {
      return valueOf(text.toUpperCase(Locale.ROOT));
    }
  }

  /** 計測対象名をCLI境界で検証する。allは全区間を独立に計測する。 */
  public static List<String> sections(String text) {
    if (text.equalsIgnoreCase("all")) {
      return SECTIONS;
    }
    List<String> selected = Arrays.stream(text.split(",")).map(String::trim).toList();
    if (!SECTIONS.containsAll(selected)
        || selected.stream().distinct().count() != selected.size()) {
      throw new IllegalArgumentException("profile sections must be unique members of " + SECTIONS);
    }
    return selected;
  }

  /**
   * 同じGPUのウォームアップ完了後、各区間を一バッチずつ診断する。
   *
   * <p>区間前後の同期を含むため、この結果を通常パイプラインの処理速度やGPU カーネル時間に換算しない。 入力は固定入力データから借用し、設定を持つ処理の完了後にだけ次のバッチへ進む。
   */
  public static Report run(
      Path checkpointRoot,
      Path corpusFile,
      Path outputDirectory,
      Role role,
      int rows,
      int transitionCapacity,
      int repetitions,
      List<String> sections,
      SettingsLoader settings)
      throws Exception {
    if ((transitionCapacity != 1 && transitionCapacity != 16) || rows < 1 || repetitions < 1) {
      throw new IllegalArgumentException(
          "transitionCapacity must be 1 or 16; rows/repetitions > 0");
    }
    Path checkpoint = EpsilonDecisionCheckpointManager.resolveExistingStrict(checkpointRoot);
    Path output = outputDirectory.toAbsolutePath().normalize();
    Files.createDirectories(output);
    var corpus = EpsilonDecisionInferenceReplayCorpusStore.load(corpusFile);
    DecisionBucket bucket =
        new DecisionBucket(transitionCapacity == 1 ? 16 : 4, transitionCapacity);
    var input = (role == Role.ACTOR ? corpus.actor() : corpus.opponent()).rows(bucket, rows);
    ArrayList<Sample> samples = new ArrayList<>();
    try (var context = new DecisionExecutionContext();
        var handle =
            role == Role.ACTOR
                ? EpsilonDecisionEvaluatorFactory.openCheckpointEvaluator(
                    checkpoint, context, settings)
                : EpsilonDecisionEvaluatorFactory.openPolicyCheckpointEvaluator(
                    checkpoint, context, settings)) {
      var servers = handle.serversForReplayBenchmark();
      for (var server : servers) {
        for (int warmup = 0; warmup < WARMUP_BATCHES; warmup++) {
          server.submitProfile(input, null).join();
        }
        for (int repetition = 0; repetition < repetitions; repetition++) {
          for (int index = 0; index < sections.size(); index++) {
            String section = sections.get((index + repetition) % sections.size());
            for (int mode = 0; mode < 2; mode++) {
              boolean recordOperators = mode == 1;
              String measurement = recordOperators ? "OPERATORS" : "WALL_ONLY";
              Path trace =
                  output.resolve(
                      "gpu"
                          + server.device().getDeviceId()
                          + "-r"
                          + repetition
                          + "-"
                          + section
                          + "-"
                          + measurement
                          + ".json");
              var profile = new InferenceProfile(section, trace, server.device(), recordOperators);
              server.submitProfile(input, profile).join();
              if (profile.result() == null) {
                throw new IllegalStateException(
                    "Selected profile section was not executed: " + section);
              }
              samples.add(
                  new Sample(
                      repetition,
                      server.device().toString(),
                      section,
                      measurement,
                      profile.result()));
            }
          }
        }
      }
    }
    Report report =
        new Report(
            checkpoint.toString(),
            corpusFile.toAbsolutePath().normalize().toString(),
            corpus.schemaFingerprint(),
            corpus.seedBase(),
            role,
            rows,
            transitionCapacity,
            repetitions,
            WARMUP_BATCHES,
            settings.bind(DecisionInferenceSettings.class),
            settings.bind(DecisionInferenceFusionSettings.class),
            samples,
            "Serialized diagnostic batches with per-section synchronization. WALL_ONLY omits"
                + " operator-profiler overhead; OPERATORS records a native trace."
                + " completedWallNanos includes CPU enqueue, GPU work and diagnostic completion"
                + " wait. ATen traces may omit native Fusion kernels; nested durations must not be"
                + " summed.");
    try (var writer = Files.newBufferedWriter(output.resolve("report.json"))) {
      new GsonBuilder().setPrettyPrinting().create().toJson(report, writer);
    }
    return report;
  }

  public record Sample(
      int repetition,
      String device,
      String section,
      String measurement,
      InferenceProfile.Result result) {}

  public record Report(
      String checkpoint,
      String corpus,
      String schemaFingerprint,
      long seedBase,
      Role role,
      int rows,
      int transitionCapacity,
      int repetitions,
      int warmupBatchesPerDevice,
      DecisionInferenceSettings inference,
      DecisionInferenceFusionSettings fusion,
      List<Sample> samples,
      String interpretation) {}
}
