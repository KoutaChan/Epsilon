package com.epsilon.cfr.training;

import ai.djl.engine.Engine;
import com.epsilon.ai.grp.EpsilonGrpExample;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import com.epsilon.cfr.collection.CfrCollector;
import com.epsilon.cfr.collection.CfrEpisodeSink;
import com.epsilon.cfr.collection.CfrPolicy;
import com.epsilon.cfr.collection.RootState;
import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.model.CfrModel;
import com.epsilon.cfr.runtime.CfrPolicyFleet;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.io.AtomicFiles;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.network.NetworkDeviceResolver;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.LongStream;

/** 実戦で使う確率的な方策の最終順位を観測し、次の学習区間で固定する局境界の価値推定を校正する。 */
public final class CfrCalibration {
  private CfrCalibration() {}

  public record Result(
      String checkpointHash,
      int trainingExamples,
      int validationExamples,
      Object training,
      Object validation,
      CfrCollector.CollectionStats collection) {}

  public static Result run(Path output, CfrConfig config, CfrPolicy policy, int games, long seed)
      throws IOException {
    Files.createDirectories(output);
    List<EpsilonGrpExample> train = new ArrayList<>(), validation = new ArrayList<>();
    CfrCollector collector =
        new CfrCollector(
            new CfrCollector.Config(
                CfrCollector.Mode.MATCH_MC,
                0,
                config.collection().inferenceBatch(),
                config.collection().concurrentGames(),
                seed,
                false,
                false,
                config.collection().multiTransitionMaxBatch(),
                config.collection().arena(),
                config.collection().batching(),
                config.utilityProfile()),
            new DecisionBranchComparisonSettings(false, false, true, 1, 8, .01));
    CfrCollector.CollectionStats stats;
    try (CfrPolicyFleet fleet =
        policy instanceof CfrModel model
            ? new CfrPolicyFleet(
                model,
                NetworkDeviceResolver.inference(config.network().devices()),
                config.collection())
            : null) {
      stats =
          collector.collect(
              LongStream.range(0, games).boxed().toList(),
              fleet == null ? policy : fleet,
              null,
              CfrEpisodeSink.consuming(
                  completed -> {
                    int code = EpsilonGrpRanks.encode(completed.finalRanks());
                    long gameId = completed.episode().episodeId();
                    // 半荘単位で分割し、同じ半荘の局境界が学習用と検証用の両方に入らないようにする。
                    List<EpsilonGrpExample> destination = gameId % 5 == 0 ? validation : train;
                    for (RootState root : completed.observedRoots()) {
                      destination.add(new EpsilonGrpExample(gameId, root.publicPrefix(), code));
                    }
                  }));
    }
    if (train.isEmpty() || validation.isEmpty())
      throw new IOException(
          "GRP calibration requires both training and held-out games (at least 2)");
    train.sort(
        Comparator.comparingLong(EpsilonGrpExample::gameId)
            .thenComparingInt(e -> e.sequence().length));
    validation.sort(
        Comparator.comparingLong(EpsilonGrpExample::gameId)
            .thenComparingInt(e -> e.sequence().length));
    Engine.getEngine("PyTorch").setRandomSeed((int) (seed ^ (seed >>> 32)));
    try (EpsilonGrpTrainingSession grp =
        EpsilonGrpTrainingSession.openForTraining(
            output, NetworkDeviceResolver.grp(config.network().devices()).primary())) {
      var training = grp.trainExamples(train, 4);
      var heldout = grp.evaluateExamples(validation);
      grp.saveNext("lcfr");
      Result result =
          new Result(
              grp.identity().checkpointSha256(),
              train.size(),
              validation.size(),
              training,
              heldout,
              stats);
      AtomicFiles.write(
          output.resolve("calibration.json"),
          CfrConfig.JSON
              .toJson(
                  Map.of(
                      "checkpointHash",
                      result.checkpointHash(),
                      "trainingExamples",
                      train.size(),
                      "validationExamples",
                      validation.size(),
                      "training",
                      training,
                      "validation",
                      heldout,
                      "collection",
                      stats))
              .getBytes(StandardCharsets.UTF_8));
      return result;
    }
  }
}
