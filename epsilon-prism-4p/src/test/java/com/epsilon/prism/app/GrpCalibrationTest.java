package com.epsilon.prism.app;

import com.epsilon.agent.app.App;
import com.epsilon.agent.app.Exit;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.train.GameCohort;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.grp.GrpFiles;
import com.epsilon.prism.grp.PhiEntry;
import com.epsilon.prism.run.Snapshot;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class GrpCalibrationTest {
  @Test
  public void commandUsesTheRecordedEstimator() throws Exception {
    var run = Files.createTempDirectory("grp-calibration");
    try {
      TestSettings.create(run);
      var files = new GrpFiles(run);
      var variant = Variant.TENHOU_4P;
      int[] scores = {32000, 28500, 24500, 15000};
      var game =
          new GameRecord(variant, 1, List.of(new GrpState(variant, 6, 0, 0, scores)), scores);
      Files.createDirectories(files.records(1).getParent());
      GameRecordCodec.write(
          files.records(1), new GameCohort(variant, 1, PhiEntry.SCORE_RANK_ID, List.of(game)));
      var bytes = new ByteArrayOutputStream();
      try (var out = new PrintStream(bytes, true, StandardCharsets.UTF_8)) {
        Assert.assertEquals(
            App.run(
                PrismApp.application(),
                new String[] {"grp-calibration", "--run", run.toString(), "--cohort", "1"},
                out),
            Exit.OK.code());
      }
      Assert.assertTrue(
          bytes
              .toString(StandardCharsets.UTF_8)
              .contains(
                  "BEFORE_LAST\tBELOW\tFROM_2000\t1\t1\t1.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\t"));
    } finally {
      Snapshot.deleteTree(run);
    }
  }
}
