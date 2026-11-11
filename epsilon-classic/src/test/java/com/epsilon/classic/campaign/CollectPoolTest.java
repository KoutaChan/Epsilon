package com.epsilon.classic.campaign;

import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.StoreEvent;
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
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 収集の段が照合する王者（CPU）。相手の候補は今の王者と、新しいものから {@code selfPlay.pool} 人までの過去の王者なので、照合するのもその
 * 王者だけ。候補から外れた古い王者のチェックポイントが無くなっても、収集は止まらない。
 */
public class CollectPoolTest {

  private static final String JAR = "test0001";

  @Test(groups = "native", timeOut = 600_000)
  public void aLostCheckpointOfAChampionOutsideThePoolDoesNotStopTheCollection() throws Exception {
    Path root = Files.createTempDirectory("classic-collect-pool");
    try {
      Path start = root.resolve("start");
      RandomCheckpoints.write(start, Architecture.NANO, Variant.TENHOU_4P, 7);
      RunDirectory run = new RunDirectory(root.resolve("run"));
      create(run, start);
      Champion first = champion(root, "nano-stage8", 0);
      Champion second = champion(root, "v00002", 2);
      Champion third = champion(root, "v00003", 3);
      ChampionStore store = ChampionStore.open(run.champion());
      store.append(new StoreEvent.Initial(first));
      store.append(new StoreEvent.Promoted(second, "v00002-champion"));
      store.append(new StoreEvent.Promoted(third, "v00003-champion"));
      // 過去の王者は 2 人、相手の候補は新しい 1 人だけ。候補から外れた初めの王者のチェックポイントを消す
      DurableFiles.delete(first.checkpoint());

      List<Collector.Macro> plans = new ArrayList<>();
      Runner<ClassicSettings, CampaignState, CampaignIteration, CampaignStep> runner =
          new Runner<>(
              run,
              new CampaignKind(RandomCollector.factory(plans)),
              new Runner.Options(JAR, null, null, () -> false));
      Assert.assertEquals(runner.train(), Ending.FINISHED);
      Assert.assertEquals(plans.size(), 1);
      Assert.assertEquals(plans.get(0).champion(), third);
      Assert.assertEquals(plans.get(0).pool(), List.of(second));
    } finally {
      DurableFiles.delete(root);
    }
  }

  /** macro が 1 回で、相手の候補にする過去の王者が 1 人までの、小さな campaign の run を作る。 */
  private static void create(RunDirectory run, Path start) throws IOException {
    List<String> sets =
        List.of(
            "run.seed=11",
            "run.start=" + start.toAbsolutePath(),
            "schedule.macros=1",
            "schedule.games=3",
            "selfPlay.pool=1",
            "ppo.steps=2",
            "batches.rows=64",
            "table.workers=1",
            "table.activeGames=8",
            "inference.encoders=1",
            "inference.slots=1",
            "inference.readyBatches=1");
    Settings<ClassicSettings> settings =
        Settings.bind(ClassicSettings.class, List.of(Layer.assignments("--set", sets)));
    try (SettingsRows.Locked locked = new SettingsRows(run).lock()) {
      locked.append(1, JAR, SettingsRow.CREATED, List.of(), settings.tree());
    }
  }

  /** run の外のチェックポイントを指す王者（収集のテストは重みを読まないので、中身は名前だけ）。 */
  private static Champion champion(Path root, String label, int version) throws IOException {
    Path directory = root.resolve(label);
    Files.createDirectories(directory);
    Files.writeString(directory.resolve("weights.bin"), label, StandardCharsets.UTF_8);
    Files.writeString(directory.resolve("model.json"), "{}", StandardCharsets.UTF_8);
    return Champion.of(label, directory, version);
  }
}
