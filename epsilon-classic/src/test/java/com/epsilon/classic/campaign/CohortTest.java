package com.epsilon.classic.campaign;

import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Settings;
import com.epsilon.classic.app.ClassicSettings;
import com.epsilon.classic.learn.GameTrajectory;
import com.epsilon.classic.learn.TrajectoryBatch;
import com.epsilon.classic.learn.UncertaintyCdf;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * コホートのファイルを書いて読み直すと、軌跡（半荘 ID・局・GRP の周辺確率・終局の点・判断ごとの O・選んだ手・振る舞いの確率・選び替えの
 * 印・席）と迷いの分布が同じになる。切れたファイル、ルールの違うファイルは読まずに IOException にする。
 */
public class CohortTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  @Test
  public void aCohortReadsBackAsWritten() throws Exception {
    Path root = Files.createTempDirectory("classic-cohort");
    try {
      Collector.Collected collected = collect(5, 3);
      Path cohort = root.resolve("cohort");
      long bytes = Cohort.write(cohort, collected);
      Assert.assertEquals(
          bytes, Files.size(cohort.resolve(Cohort.TRAJECTORIES)) + Files.size(cohort.resolve(Cohort.CALIBRATION)));
      assertSame(Cohort.trajectories(cohort, VARIANT), collected.trajectories());
      Assert.assertTrue(
          Arrays.deepEquals(Cohort.calibration(cohort).counts(), collected.calibration().counts()), "calibration");

      Path empty = root.resolve("empty");
      Cohort.write(
          empty,
          new Collector.Collected(TrajectoryBatch.of(VARIANT, List.of()), UncertaintyCdf.empty(), collected.opponent()));
      Assert.assertEquals(Cohort.trajectories(empty, VARIANT).games(), List.of());
      Assert.assertTrue(
          Arrays.deepEquals(Cohort.calibration(empty).counts(), UncertaintyCdf.empty().counts()), "empty calibration");
    } finally {
      delete(root);
    }
  }

  @Test
  public void truncatedAndForeignCohortsAreRefused() throws Exception {
    Path root = Files.createTempDirectory("classic-cohort-refused");
    try {
      Path cohort = root.resolve("cohort");
      Cohort.write(cohort, collect(2, 2));
      Assert.assertThrows(IOException.class, () -> Cohort.trajectories(cohort, Variant.TENHOU_3P));
      Path file = cohort.resolve(Cohort.TRAJECTORIES);
      byte[] whole = Files.readAllBytes(file);
      Files.write(file, Arrays.copyOf(whole, whole.length / 2));
      Assert.assertThrows(IOException.class, () -> Cohort.trajectories(cohort, VARIANT));
      Path calibration = cohort.resolve(Cohort.CALIBRATION);
      byte[] counts = Files.readAllBytes(calibration);
      Files.write(calibration, Arrays.copyOf(counts, counts.length - Long.BYTES));
      Assert.assertThrows(IOException.class, () -> Cohort.calibration(cohort));
    } finally {
      delete(root);
    }
  }

  /** 小さな作り物の macro の結果（{@link RandomCollector}）。 */
  private static Collector.Collected collect(int macro, int games) throws Exception {
    ClassicSettings settings =
        Settings.bind(
                ClassicSettings.class,
                List.of(Layer.assignments("--set", List.of("schedule.games=" + games, "run.seed=3"))))
            .value();
    List<Collector.Macro> plans = new ArrayList<>();
    ChampionStore.Champion champion =
        new ChampionStore.Champion("v00001", Path.of("/v1").toAbsolutePath(), 1, null);
    try (Collector collector = RandomCollector.factory(plans).open(settings, null, null)) {
      return collector.collect(
          new Collector.Macro(
              macro,
              17L * macro,
              settings,
              Path.of("/version").toAbsolutePath(),
              champion,
              List.of(),
              RandomCollector.calibration(macro - 1)));
    }
  }

  private static void assertSame(TrajectoryBatch actual, TrajectoryBatch expected) {
    Assert.assertEquals(actual.variant(), expected.variant());
    Assert.assertEquals(actual.games().size(), expected.games().size());
    Assert.assertEquals(actual.size(), expected.size());
    Observation left = new Observation(VARIANT);
    Observation right = new Observation(VARIANT);
    ObservationBatch leftWords = new ObservationBatch(VARIANT, 1 << 10);
    ObservationBatch rightWords = new ObservationBatch(VARIANT, 1 << 10);
    boolean priors = false;
    for (int game = 0; game < expected.games().size(); game++) {
      GameTrajectory a = actual.games().get(game);
      GameTrajectory e = expected.games().get(game);
      Assert.assertEquals(a.id(), e.id());
      Assert.assertEquals(a.hands(), e.hands());
      Assert.assertEquals(a.hasPriors(), e.hasPriors());
      priors |= e.hasPriors();
      for (int hand = 0; hand < e.hands(); hand++) {
        Assert.assertTrue(Arrays.equals(a.prior(hand), e.prior(hand)), "game " + game + " hand " + hand);
      }
      Assert.assertTrue(Arrays.equals(a.finalScores(), e.finalScores()), "game " + game);
      Assert.assertEquals(a.size(), e.size());
      for (int decision = 0; decision < e.size(); decision++) {
        String where = "game " + game + " decision " + decision;
        Assert.assertEquals(a.hand(decision), e.hand(decision), where);
        Assert.assertEquals(a.seat(decision), e.seat(decision), where);
        Assert.assertEquals(a.chosen(decision), e.chosen(decision), where);
        Assert.assertEquals(
            Float.floatToIntBits(a.behaviour(decision)), Float.floatToIntBits(e.behaviour(decision)), where);
        Assert.assertEquals(a.causal(decision), e.causal(decision), where);
        leftWords.clear();
        rightWords.clear();
        leftWords.add(a.bind(decision, left));
        rightWords.add(e.bind(decision, right));
        Assert.assertTrue(
            Arrays.equals(
                Arrays.copyOfRange(leftWords.words(), 0, leftWords.longs(0)),
                Arrays.copyOfRange(rightWords.words(), 0, rightWords.longs(0))),
            where);
      }
    }
    Assert.assertTrue(priors, "no game with GRP priors");
  }

  private static void delete(Path root) throws IOException {
    try (Stream<Path> paths = Files.walk(root)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
        Files.deleteIfExists(path);
      }
    }
  }
}
