package com.epsilon.prism.grp;

import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.TrainLock;
import com.epsilon.prism.run.Snapshot;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Optional;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * grp/mode.jsonl（{@link ModeLine}）のテスト。LATE の行は打ち切りを持ち、GRP の行は持たない。反復 t に効くのは、効き始めが t 以下の行の
 * うち効き始めが最も遅い行で、効き始めが同じなら後に書いた行。効く行が無ければ GRP のモードになる。初めの LATE の行は、run を初めて開く
 * ときに 1 回だけ書く。後から足す行は、既に始まった反復から効かせることはできない。phi.jsonl と mode.jsonl は同じロックの中で読み、
 * grp のディレクトリが無ければ何も作らずに空として読む。
 */
public class ModeHistoryTest {

  private Path run;
  private GrpFiles files;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-mode");
    files = new GrpFiles(run);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  @Test
  public void linesCheckTheirCutoffAndStart() {
    ModeLine late = ModeLine.late(1e-5, 3, "test");
    Assert.assertEquals(late.mode(), PhiMode.LATE);
    Assert.assertEquals((double) late.cutoff(), 1e-5);
    Assert.assertNull(ModeLine.grp(3, "test").cutoff());
    Assert.assertEquals((double) ModeLine.late(0, 1, "test").cutoff(), 0.0);
    for (Double cutoff : new Double[] {null, -1e-9, 1.0, Double.NaN})
      Assert.assertThrows(IllegalArgumentException.class, () -> new ModeLine(PhiMode.LATE, cutoff, 3, "test"));
    Assert.assertThrows(IllegalArgumentException.class, () -> new ModeLine(PhiMode.GRP, 1e-5, 3, "test"));
    Assert.assertThrows(IllegalArgumentException.class, () -> ModeLine.grp(0, "test"));
    Assert.assertThrows(NullPointerException.class, () -> new ModeLine(null, null, 3, "test"));
    Assert.assertEquals(PhiMode.LATE.alias(), "late");
    Assert.assertEquals(PhiMode.GRP.alias(), "grp");
  }

  @Test
  public void theLatestStartedLineIsInEffect() {
    ModeLine first = ModeLine.late(1e-5, 4, "first");
    ModeLine back = ModeLine.grp(8, "back");
    ModeLine again = ModeLine.late(1e-6, 8, "again");
    List<ModeLine> lines = List.of(first, back, again, ModeLine.grp(12, "later"));
    Assert.assertEquals(ModeLine.effectiveAt(lines, 3), Optional.empty());
    Assert.assertEquals(ModeLine.effectiveAt(lines, 4).orElseThrow(), first);
    Assert.assertEquals(ModeLine.effectiveAt(lines, 7).orElseThrow(), first);
    // 効き始めが同じなら、後に書いた行が効く
    Assert.assertEquals(ModeLine.effectiveAt(lines, 8).orElseThrow(), again);
    Assert.assertEquals(ModeLine.effectiveAt(lines, 11).orElseThrow(), again);
    Assert.assertEquals(ModeLine.effectiveAt(lines, 12).orElseThrow().mode(), PhiMode.GRP);
  }

  @Test
  public void theFirstLateLineIsWrittenOnceWhenTheRunOpens() throws IOException {
    GrpFixtures.writeState(run, 5);
    PhiHistory.initialize(files, GrpFixtures.scoreStart("phi.late.from=6", "phi.late.cutoff=1e-6"));
    List<ModeLine> written = PhiHistory.readModes(files);
    Assert.assertEquals(written, List.of(new ModeLine(PhiMode.LATE, 1e-6, 6, "phi.late.from = 6")));
    PhiHistory.initialize(files, GrpFixtures.scoreStart("phi.late.from=6", "phi.late.cutoff=1e-6"));
    Assert.assertEquals(PhiHistory.readModes(files), written, "a second open writes nothing");

    // 使い始める反復が 0 なら、行を書かない
    Path other = run.resolve("other");
    PhiHistory.initialize(new GrpFiles(other), GrpFixtures.scoreStart());
    Assert.assertFalse(Files.exists(new GrpFiles(other).modeHistory()));
    // 固定の GRP の run でも同じ
    GrpFiles fixed = new GrpFiles(run.resolve("fixed"));
    PhiHistory.initializeModes(fixed, new LateSettings(0, 1e-5, 16_384));
    Assert.assertFalse(Files.exists(fixed.directory()));
    PhiHistory.initializeModes(fixed, new LateSettings(3, 1e-4, 16_384));
    PhiHistory.initializeModes(fixed, new LateSettings(3, 1e-4, 16_384));
    Assert.assertEquals(PhiHistory.readModes(fixed), List.of(ModeLine.late(1e-4, 3, "phi.late.from = 3")));
  }

  @Test
  public void appendedLinesStartAfterTheLastStartedIteration() throws IOException {
    GrpFixtures.writeState(run, 5);
    PhiHistory.initialize(files, GrpFixtures.scoreStart());
    Assert.assertThrows(IllegalArgumentException.class, () -> PhiHistory.appendMode(files, ModeLine.late(1e-5, 4, "x")));
    PhiHistory.appendMode(files, ModeLine.late(1e-5, 5, "switch: late"));
    try (TrainLock training = TrainLock.hold(new RunDirectory(run).trainLock())) {
      Assert.assertThrows(IllegalArgumentException.class, () -> PhiHistory.appendMode(files, ModeLine.grp(5, "x")));
      PhiHistory.appendMode(files, ModeLine.grp(6, "switch: grp"));
    }
    // コホート 6 を確定した後は、反復 6 も始まっているとみなす
    GrpFixtures.writeState(run, 6);
    GrpFixtures.seal(files, 6);
    Assert.assertThrows(IllegalArgumentException.class, () -> PhiHistory.appendMode(files, ModeLine.grp(6, "x")));
    PhiHistory.appendMode(files, ModeLine.late(0, 7, "switch: late"));
    Assert.assertEquals(
        PhiHistory.readModes(files),
        List.of(ModeLine.late(1e-5, 5, "switch: late"), ModeLine.grp(6, "switch: grp"), ModeLine.late(0, 7, "switch: late")));

    // GRP の行には打ち切りのフィールドを書かない
    List<String> lines = Files.readAllLines(files.modeHistory());
    JsonObject grp = JsonParser.parseString(lines.get(1)).getAsJsonObject();
    Assert.assertFalse(grp.has("cutoff"), lines.get(1));
    Assert.assertEquals(JsonParser.parseString(lines.get(0)).getAsJsonObject().get("cutoff").getAsDouble(), 1e-5);

    PhiHistory.Lines both = PhiHistory.readLocked(files);
    Assert.assertEquals(both.entries(), PhiHistory.read(files));
    Assert.assertEquals(both.modes(), PhiHistory.readModes(files));
    Assert.assertEquals(both.entries().getFirst().id(), PhiEntry.INITIAL_ID);
  }

  @Test
  public void brokenAndPartialLines() throws IOException {
    GrpFixtures.writeState(run, 1);
    PhiHistory.initialize(files, GrpFixtures.scoreStart("phi.late.from=2"));
    // 改行の無い最後の行は、書きかけとみなして読まず、次の追記で切り捨てる
    Files.writeString(files.modeHistory(), "{\"mode\":\"GRP\"", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
    Assert.assertEquals(PhiHistory.readModes(files).size(), 1);
    PhiHistory.appendMode(files, ModeLine.grp(3, "switch: grp"));
    Assert.assertEquals(PhiHistory.readModes(files).getLast(), ModeLine.grp(3, "switch: grp"));
    Files.writeString(
        files.modeHistory(), "{\"mode\":\"LATE\",\"effectiveFrom\":4,\"reason\":\"x\"}\n", StandardCharsets.UTF_8,
        StandardOpenOption.APPEND);
    IOException broken = Assert.expectThrows(IOException.class, () -> PhiHistory.readModes(files));
    Assert.assertTrue(broken.getMessage().startsWith("Broken line 3 of"), broken.getMessage());
    Assert.assertThrows(IOException.class, () -> PhiHistory.readLocked(files));
  }

  @Test
  public void aRunWithoutTheGrpDirectoryReadsEmpty() throws IOException {
    PhiHistory.Lines lines = PhiHistory.readLocked(files);
    Assert.assertEquals(lines.entries(), List.of());
    Assert.assertEquals(lines.modes(), List.of());
    Assert.assertFalse(Files.exists(files.directory()), "reading takes no lock and creates nothing");
  }
}
