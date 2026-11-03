package com.epsilon.prism.grp;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.EpsilonUtilityTargets;
import com.epsilon.ai.grp.EpsilonGrpInference;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.run.PrismConfig;
import com.google.gson.Gson;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/** grp の試験の道具: 白紙の GRP の checkpoint、合成の半荘、GRP を maxBatch 行ずつ直に通した Φ、run の config.json と state.json。 */
final class GrpFixtures {
  static final EpsilonUtilityProfile PROFILE = EpsilonUtilityProfile.TENHOU;

  private GrpFixtures() {}

  /**
   * decision/grp/grp_00001 に白紙（乱数の初期値）の GRP を保存する。
   *
   * @param decision grpCheckpoint に書くディレクトリ
   * @return 保存した checkpoint の正確なディレクトリ
   */
  static Path checkpoint(Path decision) throws IOException {
    try (var session = EpsilonGrpTrainingSession.openForTraining(decision, Device.cpu())) {
      session.saveAt("grp", 1);
    }
    return EpsilonGrpCheckpointManager.resolveExisting(decision.resolve("grp")).toAbsolutePath().normalize();
  }

  /** 半荘 first.. の合成の記録（局の数 3〜9、点数の和は 100,000 のまま和了と放銃で動く）。 */
  static List<RecordedGame> games(long first, int count, long seed) {
    var random = new SplittableRandom(seed);
    List<RecordedGame> games = new ArrayList<>(count);
    for (int g = 0; g < count; g++) {
      int hands = 3 + random.nextInt(7);
      int[] scores = {25000, 25000, 25000, 25000};
      int kyoku = 0, honba = 0;
      List<HandStart> starts = new ArrayList<>();
      for (int h = 0; h < hands; h++) {
        move(scores, random);
        if (random.nextInt(3) == 0) honba++;
        else {
          honba = 0;
          kyoku = Math.min(kyoku + 1, 11);
        }
        starts.add(new HandStart(kyoku, honba, 0, scores.clone()));
      }
      move(scores, random);
      games.add(new RecordedGame(first + g, starts, scores.clone()));
    }
    return games;
  }

  private static void move(int[] scores, SplittableRandom random) {
    int winner = random.nextInt(4), loser = random.nextInt(4), points = 100 * (10 + random.nextInt(120));
    if (winner != loser) {
      scores[winner] += points;
      scores[loser] -= points;
    }
  }

  /** GRP を maxBatch 行ずつ（最後は写しで埋める）に通した Φ（温度も表もなし。学習の JVM の GrpPhi とビットで同じ値）。 */
  static HandEndPhi direct(EpsilonGrpInference inference) {
    return inputs -> {
      double[][] utilities = new double[inputs.size()][4];
      double[][] ranks = new double[inputs.size()][16];
      if (inputs.isEmpty()) return new PhiValues(utilities, ranks);
      List<float[]> prefixes = new ArrayList<>();
      for (HandEndInput input : inputs) prefixes.add(input.prefix());
      while (prefixes.size() % inference.maxBatch() != 0) prefixes.add(prefixes.getLast());
      List<float[]> marginals = inference.predictMarginalProbabilities(prefixes);
      for (int i = 0; i < inputs.size(); i++) {
        for (int seat = 0; seat < 4; seat++)
          utilities[i][seat] =
              EpsilonUtilityTargets.expectedRankUtilityTrustedDouble(PROFILE, marginals.get(i), seat * 4);
        for (int j = 0; j < 16; j++) ranks[i][j] = marginals.get(i)[j];
      }
      return new PhiValues(utilities, ranks);
    };
  }

  /** 設定の [epsilon.prism] の既定に pairs を重ねた設定。 */
  static PrismConfig config(String... pairs) {
    Map<String, String> values = new LinkedHashMap<>();
    for (int i = 0; i < pairs.length; i += 2) values.put(PrismConfig.SETTINGS + pairs[i], pairs[i + 1]);
    return PrismConfig.fromSettings(EpsilonSettings.of(values));
  }

  /** SELF_GRP・点数の順位で始める run の設定。 */
  static PrismConfig scoreStart() {
    return config("phiSource", "SELF_GRP", "phiStart", "SCORE_RANK", "phiMonitor", "RECORD", "phiDrift", "OFF");
  }

  /** SELF_GRP・grpCheckpoint の GRP で始める run の設定。 */
  static PrismConfig grpStart(Path decision) {
    return config(
        "phiSource", "SELF_GRP", "phiStart", "GRP_CHECKPOINT", "phiMonitor", "RECORD", "phiDrift", "OFF",
        "grpCheckpoint", decision.toString());
  }

  /** run/config.json を書く。 */
  static void writeConfig(Path run, PrismConfig config) throws IOException {
    Files.createDirectories(run);
    Files.writeString(run.resolve("config.json"), new Gson().toJson(config));
  }

  /** run/state.json に次のコホートだけを書く。 */
  static void writeState(Path run, int next) throws IOException {
    Files.createDirectories(run);
    Files.writeString(run.resolve("state.json"), "{\"format\": 1, \"next\": " + next + "}");
  }

  /** 取り置き h の READY の候補の行を refits.jsonl に書き、境目の点検の見本を games で作る。 */
  static RefitRecord ready(GrpFiles files, String id, Path checkpoint, HandEndPhi phi, List<RecordedGame> games)
      throws IOException {
    var record =
        new RefitRecord(
            id,
            RefitStatus.READY,
            RefitCause.MANUAL,
            new RefitWindow(3, 6, 7),
            GrpInit.WARM,
            PhiEntry.INITIAL_ID,
            checkpoint.toString(),
            EpsilonGrpCheckpointManager.checkpointSha256(checkpoint),
            null,
            "",
            1.0);
    record.append(files);
    BoundaryCheck.sample(games, phi, 7).write(files.boundary(id));
    return record;
  }
}
