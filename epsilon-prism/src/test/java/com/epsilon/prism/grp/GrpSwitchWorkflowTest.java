package com.epsilon.prism.grp;

import com.epsilon.prism.run.Snapshot;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.stream.Stream;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * grp-switch: READY の候補を次の空いた反復（学習の JVM が動いていればその次）から効かせ、巻き戻しは既にある項目の行を足す。始まった反復・READY
 * でない候補・知らない id・sha の変わった checkpoint・SELF_GRP でない run は拒み、既に同じ Φ が効いていれば何も足さない。
 */
public class GrpSwitchWorkflowTest {

  private static int run(Path run, String... options) throws Exception {
    String[] values = new String[options.length + 2];
    values[0] = "--run";
    values[1] = run.toString();
    System.arraycopy(options, 0, values, 2, options.length);
    return new GrpSwitchWorkflow().execute(GrpRefitterTest.arguments(values));
  }

  @Test(timeOut = 300_000)
  public void switchesRollsBackAndRefuses() throws Exception {
    Path run = Files.createTempDirectory("prism-switch");
    try {
      var files = new GrpFiles(run);
      GrpFixtures.writeConfig(run, GrpFixtures.scoreStart());
      GrpFixtures.writeState(run, 8);
      // g000 は学習の JVM が書く
      Assert.assertThrows(IllegalStateException.class, () -> run(run, "--to", "g000"));
      PhiHistory.initialize(files, GrpFixtures.scoreStart());
      HandEndPhi score = ScorePhi.offline(PhiEntryKind.SCORE_RANK, GrpFixtures.PROFILE);
      List<RecordedGame> games = GrpFixtures.games(0, 8, 1);
      Path first = GrpFixtures.checkpoint(run.resolve("first"));
      Path second = GrpFixtures.checkpoint(run.resolve("second"));
      GrpFixtures.ready(files, "g007-a1", first, score, games);
      GrpFixtures.ready(files, "g007-a2", second, score, games);
      RefitRecord rejected = GrpFixtures.ready(files, "g007-a3", second, score, games);
      rejected.withStatus(RefitStatus.REJECTED, "boundary").append(files);

      Assert.assertEquals(run(run, "--to", "g007-a1"), 0);
      PhiEntry switched = PhiHistory.read(files).getLast();
      Assert.assertEquals(switched.id(), "g007-a1");
      Assert.assertEquals(switched.kind(), PhiEntryKind.SELF_CHECKPOINT);
      Assert.assertEquals(switched.effectiveFrom(), 8);
      // 同じ Φ が効いていれば何も足さない
      Assert.assertEquals(run(run, "--to", "g007-a1"), 1);
      Assert.assertEquals(PhiHistory.read(files).size(), 2);
      // 学習の JVM が動いていれば 1 反復先から
      try (PhiHistory.Locked training = PhiHistory.holdTraining(files)) {
        Assert.assertEquals(run(run, "--to", "g007-a2", "--reason", "manual"), 0);
        Assert.assertEquals(PhiHistory.read(files).getLast().effectiveFrom(), 9);
        Assert.assertThrows(IllegalArgumentException.class, () -> run(run, "--to", "g000", "--from", "8"));
      }
      Assert.assertThrows(IllegalArgumentException.class, () -> run(run, "--to", "g007-a2", "--from", "7"));
      // 巻き戻し
      Assert.assertEquals(run(run, "--to", "g000", "--from", "12", "--reason", "x1n3"), 0);
      PhiEntry back = PhiHistory.read(files).getLast();
      Assert.assertEquals(back.id(), PhiEntry.INITIAL_ID);
      Assert.assertEquals(back.kind(), PhiEntryKind.SCORE_RANK);
      Assert.assertEquals(back.reason(), "rollback: x1n3");
      Assert.assertEquals(PhiHistory.effectiveAt(PhiHistory.read(files), 11).orElseThrow().id(), "g007-a2");
      Assert.assertEquals(PhiHistory.effectiveAt(PhiHistory.read(files), 12).orElseThrow().id(), PhiEntry.INITIAL_ID);
      // 拒んだ候補・知らない id・sha の変わった checkpoint
      Assert.assertThrows(IllegalArgumentException.class, () -> run(run, "--to", "g007-a3", "--from", "13"));
      Assert.assertThrows(IllegalArgumentException.class, () -> run(run, "--to", "g099-a1", "--from", "13"));
      try (Stream<Path> paths = Files.list(first)) {
        Path params = paths.filter(path -> path.getFileName().toString().endsWith(".params")).findFirst().orElseThrow();
        Files.writeString(params, "x", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
      }
      Assert.assertThrows(IllegalStateException.class, () -> run(run, "--to", "g007-a1", "--from", "13"));
      Assert.assertEquals(PhiHistory.read(files).size(), 4);
      // SELF_GRP でない run は切り替えない
      GrpFixtures.writeConfig(run, GrpFixtures.config());
      Assert.assertThrows(IllegalArgumentException.class, () -> run(run, "--to", "g007-a2", "--from", "13"));
    } finally {
      Snapshot.deleteTree(run);
    }
  }
}
