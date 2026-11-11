package com.epsilon.classic.campaign;

import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.Ending;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.Runner;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Settings;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.RandomCheckpoints;
import com.epsilon.classic.app.ClassicSettings;
import com.epsilon.classic.learn.GameTrajectory;
import com.epsilon.classic.learn.TrajectoryBatch;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.mahjong.rules.Variant;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * この jar の campaign（{@link CampaignKind#STANDARD}）の 1 回の macro を、乱数の重みの小さなチェックポイントで CPU で回す。収集の段は
 * {@link SelfPlay} が {@link Collect} で打ち、半荘ごとに学習器の席（半荘の番号を席の数で割った余り）の軌跡をコホートに確定し、学習の段が
 * バージョン 2 を公開する。王者を替える卓を書かない run では、決闘の段と昇格の段は SKIPPED。
 */
public class StandardCampaignTest {

  private static final String JAR = "test0001";

  private static final Variant VARIANT = Variant.TENHOU_4P;

  private static final int GAMES = 4;

  private static final int STEPS = 2;

  @Test(groups = "native", timeOut = 1_800_000)
  public void oneMacroCollectsTheSelfPlayAndPublishesTheNextVersion() throws Exception {
    Path root = Files.createTempDirectory("classic-campaign-standard");
    try {
      Path start = root.resolve("start");
      RandomCheckpoints.write(start, Architecture.NANO, VARIANT, 7);
      RunDirectory run = new RunDirectory(root.resolve("run"));
      create(run, start);
      Runner<ClassicSettings, CampaignState, CampaignIteration, CampaignStep> runner =
          new Runner<>(run, CampaignKind.STANDARD, new Runner.Options(JAR, null, null, () -> false));
      Assert.assertEquals(runner.train(), Ending.FINISHED);

      TrajectoryBatch cohort = Cohort.trajectories(run.rows(1), VARIANT);
      Assert.assertEquals(cohort.games().size(), GAMES);
      for (GameTrajectory game : cohort.games()) {
        Assert.assertTrue(game.size() > 0, "game " + game.id() + " has no learner decisions");
        for (int decision = 0; decision < game.size(); decision++) {
          Assert.assertEquals(game.seat(decision), (int) (game.id() % VARIANT.seats()), "game " + game.id());
        }
      }
      Assert.assertEquals(
          ClassicCheckpoint.header(run.snapshot(2)).step(), ClassicCheckpoint.header(run.snapshot(1)).step() + STEPS);

      List<String> lines = DurableFiles.lines(run.report(RunDirectory.ITERATIONS));
      Assert.assertEquals(lines.size(), 1);
      JsonObject steps = JsonParser.parseString(lines.getFirst()).getAsJsonObject().getAsJsonObject("steps");
      JsonObject collect = steps.getAsJsonObject(CampaignStep.COLLECT.name());
      Assert.assertEquals(collect.get("games").getAsInt(), GAMES, collect.toString());
      Assert.assertEquals(collect.get("decisions").getAsLong(), cohort.size(), collect.toString());
      Assert.assertEquals(collect.get("opponent").getAsString(), CampaignSession.label(1), collect.toString());
      for (CampaignStep skipped : List.of(CampaignStep.DUEL, CampaignStep.DECIDE)) {
        Assert.assertEquals(steps.getAsJsonObject(skipped.name()).get("status").getAsString(), "SKIPPED", skipped.name());
      }
    } finally {
      delete(root);
    }
  }

  /** 小さな campaign の設定で、設定の 1 行目を書いた run を作る。 */
  private static void create(RunDirectory run, Path start) throws IOException {
    List<String> sets =
        List.of(
            "run.seed=13",
            "run.start=" + start.toAbsolutePath(),
            "schedule.macros=1",
            "schedule.games=" + GAMES,
            "ppo.steps=" + STEPS,
            "batches.rows=64",
            "table.workers=2",
            "table.activeGames=4",
            "inference.encoders=1",
            "inference.slots=1",
            "inference.readyBatches=1");
    Settings<ClassicSettings> settings =
        Settings.bind(ClassicSettings.class, List.of(Layer.assignments("--set", sets)));
    try (SettingsRows.Locked locked = new SettingsRows(run).lock()) {
      locked.append(1, JAR, SettingsRow.CREATED, List.of(), settings.tree());
    }
  }

  private static void delete(Path root) throws IOException {
    try (Stream<Path> paths = Files.walk(root)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
        Files.deleteIfExists(path);
      }
    }
  }
}
