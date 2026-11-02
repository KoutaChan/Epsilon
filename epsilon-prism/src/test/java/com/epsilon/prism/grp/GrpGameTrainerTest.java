package com.epsilon.prism.grp;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpNetwork;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.nano.ai.network.NetworkFactory;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * trainGames: 1 半荘の全ての接頭辞をまとめた順伝播のロジットと損失は、接頭辞ごとの {@link EpsilonGrpNetwork#forwardLogits} と一致する
 * （損失は 1e-6 以内）。学習は損失を下げ、正確なディレクトリに読める checkpoint を書き、同じ種なら同じ重みになる（1 スレッド）。
 */
public class GrpGameTrainerTest {
  private static final int THREADS = 1;

  @Test(groups = "native")
  public void gameForwardMatchesPerPrefixForward() {
    List<RecordedGame> games = games(new SplittableRandom(2), 24);
    Engine.getEngine("PyTorch").setRandomSeed(5);
    try (Model model =
            NetworkFactory.createGrpModel(Device.cpu(), EpsilonGrpNetwork.DEFAULT_HIDDEN, EpsilonGrpNetwork.DEFAULT_LAYERS);
        NDManager manager = model.getNDManager().newSubManager()) {
      EpsilonGrpNetwork network = (EpsilonGrpNetwork) model.getBlock();
      ParameterStore parameters = new ParameterStore(manager, false);
      NDArray batched = GrpSequenceForward.gameLogits(network, parameters, manager, games, false);
      List<float[]> prefixes = new ArrayList<>();
      for (RecordedGame game : games)
        for (int k = 0; k < game.continuingHandEnds(); k++) prefixes.add(game.prefix(k));
      NDArray single = perPrefix(network, parameters, manager, prefixes);
      Assert.assertEquals(batched.getShape(), single.getShape());
      float[] a = batched.toFloatArray(), b = single.toFloatArray();
      for (int i = 0; i < a.length; i++) Assert.assertEquals(a[i], b[i], 1e-5, "logit " + i);
      NDArray labels = GrpGameTrainer.labels(manager, games);
      double gameLoss = GrpGameTrainer.loss(batched, labels).getFloat();
      double prefixLoss = GrpGameTrainer.loss(single, labels).getFloat();
      Assert.assertEquals(gameLoss, prefixLoss, 1e-6);
    }
  }

  @Test(groups = "native")
  public void trainingLowersTheLossAndWritesAReadableCheckpoint() throws IOException {
    List<RecordedGame> games = games(new SplittableRandom(3), 400);
    Path root = Files.createTempDirectory("grp-train");
    List<Double> losses = new ArrayList<>();
    GrpTrainingPlan plan = new GrpTrainingPlan(GrpInit.SCRATCH, 32, 1e-3, 4, THREADS, 7, Duration.ofMinutes(10));
    GrpGameTrainer.Trained trained =
        GrpGameTrainer.train(
            null,
            games.subList(0, 320),
            games.subList(320, 400),
            plan,
            root.resolve("scratch"),
            (epoch, trainLoss, validationLoss, model) -> losses.add(trainLoss));
    Assert.assertFalse(trained.timedOut());
    Assert.assertTrue(losses.getLast() < losses.getFirst(), "training NLL " + losses);
    Assert.assertTrue(EpsilonGrpCheckpointManager.isValidCheckpoint(trained.directory()));
    Assert.assertEquals(trained.sha256(), EpsilonGrpCheckpointManager.checkpointSha256(trained.directory()));
    Assert.assertTrue(Double.isFinite(trained.validationLoss()) && trained.gamesPerSecond() > 0);
    Assert.assertFalse(Files.exists(root.resolve(".scratch.training")));
    try (var phi = CheckpointPhi.open(trained.directory(), new double[0], null, EpsilonUtilityProfile.TENHOU)) {
      PhiValues values = phi.gameValues(games.subList(320, 400));
      for (double[] row : values.utilities()) for (double value : row) Assert.assertTrue(Double.isFinite(value));
    }
    // 温めは保存した重みから始める
    GrpGameTrainer.Trained warm =
        GrpGameTrainer.train(
            trained.directory(),
            games.subList(0, 320),
            List.of(),
            new GrpTrainingPlan(GrpInit.WARM, 32, 2e-4, 1, THREADS, 7, Duration.ofMinutes(10)),
            root.resolve("warm"));
    Assert.assertEquals(warm.epochs(), 1);
    Assert.assertTrue(Double.isNaN(warm.validationLoss()));
    Assert.assertNotEquals(warm.sha256(), trained.sha256());
    // 同じ種・同じ入力なら同じ重み
    GrpGameTrainer.Trained again =
        GrpGameTrainer.train(null, games.subList(0, 320), games.subList(320, 400), plan, root.resolve("again"));
    Assert.assertEquals(again.sha256(), trained.sha256());
  }

  @Test(groups = "native")
  public void refusesAnExistingOutputAndStopsAtTheDeadline() throws IOException {
    List<RecordedGame> games = games(new SplittableRandom(4), 64);
    Path root = Files.createTempDirectory("grp-train");
    Path existing = Files.createDirectory(root.resolve("taken"));
    GrpTrainingPlan plan = new GrpTrainingPlan(GrpInit.SCRATCH, 16, 1e-3, 2, THREADS, 1, Duration.ofMinutes(10));
    Assert.assertThrows(IOException.class, () -> GrpGameTrainer.train(null, games, List.of(), plan, existing));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> GrpGameTrainer.train(existing, games, List.of(), plan, root.resolve("x")));
    GrpGameTrainer.Trained late =
        GrpGameTrainer.train(
            null,
            games,
            List.of(),
            new GrpTrainingPlan(GrpInit.SCRATCH, 16, 1e-3, 2, THREADS, 1, Duration.ZERO),
            root.resolve("late"));
    Assert.assertTrue(late.timedOut());
    Assert.assertNull(late.directory());
    Assert.assertFalse(Files.exists(root.resolve("late")));
  }

  /** 接頭辞ごとの {@link EpsilonGrpNetwork#forwardLogits}（詰め物と長さ付きの 1 つのバッチ）。 */
  private static NDArray perPrefix(
      EpsilonGrpNetwork network, ParameterStore parameters, NDManager manager, List<float[]> prefixes) {
    int width = EpsilonGrpFeature.FEATURE_SIZE;
    int time = 0;
    for (float[] prefix : prefixes) time = Math.max(time, prefix.length / width);
    float[] sequences = new float[prefixes.size() * time * width];
    float[] lengths = new float[prefixes.size()];
    for (int i = 0; i < prefixes.size(); i++) {
      float[] prefix = prefixes.get(i);
      System.arraycopy(prefix, 0, sequences, i * time * width, prefix.length);
      lengths[i] = prefix.length / (float) width;
    }
    return network.forwardLogits(
        parameters,
        manager.create(sequences, new Shape(prefixes.size(), time, width)),
        manager.create(lengths, new Shape(prefixes.size())),
        false,
        new PairList<>());
  }

  /**
   * 局の数 1〜12 の半荘。点数は和了と放銃で動き、最終順位は最後の点数に雑音を足した順（学べる関係）。半荘の最後の局末は最終の点数だけ。
   */
  static List<RecordedGame> games(SplittableRandom random, int count) {
    List<RecordedGame> games = new ArrayList<>(count);
    for (int g = 0; g < count; g++) {
      int hands = 1 + random.nextInt(12);
      int[] scores = {25000, 25000, 25000, 25000};
      int kyoku = 0, honba = 0;
      List<HandStart> starts = new ArrayList<>();
      for (int k = 0; k < hands; k++) {
        int winner = random.nextInt(4), loser = random.nextInt(4), points = 100 * (10 + random.nextInt(120));
        if (winner != loser) {
          scores[winner] += points;
          scores[loser] -= points;
        }
        if (random.nextInt(3) == 0) honba++;
        else {
          honba = 0;
          kyoku++;
        }
        starts.add(new HandStart(kyoku, honba, 0, scores.clone()));
      }
      int[] finals = scores.clone();
      for (int seat = 0; seat < 4; seat++) finals[seat] += 100 * (random.nextInt(81) - 40);
      int drift = 100_000 - (finals[0] + finals[1] + finals[2] + finals[3]);
      finals[0] += drift;
      games.add(new RecordedGame(g, starts.subList(0, hands - 1), finals));
    }
    return games;
  }
}
