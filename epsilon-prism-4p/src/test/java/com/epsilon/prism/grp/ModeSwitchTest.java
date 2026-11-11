package com.epsilon.prism.grp;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.TrainLock;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Optional;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * Φ のモードの切り替え（{@link PhiControl#switchMode} と、grp-switch の別名 late・grp）のテスト。足した行は、始まった最後の反復の次から
 * 効く（学習の JVM が動いていれば、さらに 1 つ先から）。既に始まった反復から効かせることはできない。LATE の行の打ち切りは --cutoff の値で、
 * 無ければ設定の値。その反復で既に同じモードが効いていれば何も足さない。
 *
 * <p>LATE の行を足すときは、その反復に効く GRP の項目が q を持てば、先にその項目を開いてみる。開けなければ何も書かずに止める。q を持たなければ
 * 警告を出して行を書き、その項目が効く反復は GRP のモードのままになる。固定の GRP の run でもモードは切り替えられ、そのときは設定にある
 * 初めの LATE の行が先に書かれる。項目の切り替えと再学習ができるのは、自前の GRP の run だけ。モードを持たない run での切り替えと、late
 * 以外への --cutoff は使い方の誤りになる。
 */
public class ModeSwitchTest {

  private Path run;
  private GrpFiles files;
  private ListAppender<ILoggingEvent> logs;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-mode-switch");
    files = new GrpFiles(run);
    logs = new ListAppender<>();
    logs.start();
    root().addAppender(logs);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    root().detachAppender(logs);
    Snapshot.deleteTree(run);
  }

  @Test
  public void theAliasesNameTheModesOnly() {
    Assert.assertEquals(PhiMode.byAlias("late"), Optional.of(PhiMode.LATE));
    Assert.assertEquals(PhiMode.byAlias("grp"), Optional.of(PhiMode.GRP));
    for (String other : new String[] {"LATE", "Grp", "late ", "", "g000", PhiEntry.FIXED_ID})
      Assert.assertEquals(PhiMode.byAlias(other), Optional.empty(), other);
    // 項目の id には別名を使えないので、grp-switch の切り替え先を取り違えることはない
    for (PhiMode mode : PhiMode.values())
      Assert.assertThrows(IllegalArgumentException.class, () -> GrpFixtures.scoreEntry(mode.alias(), 1));
  }

  @Test
  public void theAliasesAppendLinesFromTheFirstOpenIteration() throws Exception {
    String[] assignments = GrpFixtures.concat(GrpFixtures.scoreStartAssignments(), "phi.late.cutoff=1e-4");
    GrpFixtures.writeSettings(run, assignments);
    GrpFixtures.writeState(run, 5);
    PhiSetup setup = GrpFixtures.setup(assignments);
    // g000 は学習の JVM が書く。書かれる前は切り替えられない
    Assert.assertEquals(switchTo(run, "late"), Exit.FAILED.code());
    Assert.assertThrows(
        IllegalStateException.class, () -> PhiControl.switchMode(run, setup, PhiMode.LATE, null, null, "test"));
    Assert.assertFalse(Files.exists(files.modeHistory()));
    PhiHistory.initialize(files, setup);

    // 行が無ければ GRP のモードが効いているので、grp では何も足さない
    Assert.assertEquals(switchTo(run, "grp"), Exit.OK.code());
    Assert.assertFalse(Files.exists(files.modeHistory()));
    // 打ち切りは設定の値になる
    Assert.assertEquals(switchTo(run, "late"), Exit.OK.code());
    ModeLine late = ModeLine.late(1e-4, 5, "switch: grp-switch");
    Assert.assertEquals(PhiHistory.readModes(files), List.of(late));
    Assert.assertEquals(switchTo(run, "late"), Exit.OK.code());
    Assert.assertTrue(PhiControl.switchMode(run, setup, PhiMode.LATE, 1e-4, 5, "test").isEmpty());
    Assert.assertEquals(PhiHistory.readModes(files), List.of(late));
    // 打ち切りの値が違えば、別の行として足す
    Assert.assertEquals(switchTo(run, "late", "--cutoff", "1e-6", "--reason", "finer"), Exit.OK.code());
    ModeLine finer = ModeLine.late(1e-6, 5, "switch: finer");
    Assert.assertEquals(PhiHistory.readModes(files), List.of(late, finer));

    // 学習の JVM が動いていれば 1 反復先から効く。始まった反復からは効かせない
    ModeLine manual = ModeLine.grp(6, "switch: manual");
    try (TrainLock training = TrainLock.hold(new RunDirectory(run).trainLock())) {
      Assert.assertEquals(PhiControl.switchMode(run, setup, PhiMode.GRP, null, null, "manual"), Optional.of(manual));
      Assert.assertThrows(
          IllegalArgumentException.class, () -> PhiControl.switchMode(run, setup, PhiMode.LATE, null, 5, "test"));
      Assert.assertEquals(switchTo(run, "late", "--from", "5"), Exit.FAILED.code());
    }
    // コホート 6 を確定した後は、反復 6 も始まっているとみなす
    GrpFixtures.writeState(run, 6);
    GrpFixtures.seal(files, 6);
    Assert.assertEquals(switchTo(run, "late", "--from", "6"), Exit.FAILED.code());
    Assert.assertEquals(switchTo(run, "late"), Exit.OK.code());
    Assert.assertEquals(switchTo(run, "grp", "--from", "9", "--reason", "x1n3"), Exit.OK.code());
    List<ModeLine> written =
        List.of(late, finer, manual, ModeLine.late(1e-4, 7, "switch: grp-switch"), ModeLine.grp(9, "switch: x1n3"));
    Assert.assertEquals(PhiHistory.readModes(files), written);
    Assert.assertEquals(PhiHistory.read(files).size(), 1, "mode lines leave the Φ entries alone");

    // g000 は q を持たないので、LATE を求めた反復でも GRP のモードで値を求める
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      int[] iterations = {4, 5, 6, 7, 9};
      PhiMode[] requested = {PhiMode.GRP, PhiMode.LATE, PhiMode.GRP, PhiMode.LATE, PhiMode.GRP};
      for (int i = 0; i < iterations.length; i++) {
        PhiValue value = resolver.resolve(iterations[i], setup.phi()).value();
        Assert.assertEquals(value.requested(), requested[i], "iteration " + iterations[i]);
        Assert.assertEquals(value.mode(), PhiMode.GRP, "iteration " + iterations[i]);
      }
    }
    Assert.assertEquals(PhiHistory.readModes(files), written);
  }

  @Test
  public void theCutoffGoesWithLateOnly() throws Exception {
    GrpFixtures.writeSettings(run, GrpFixtures.scoreStartAssignments());
    GrpFixtures.writeState(run, 3);
    PhiHistory.initialize(files, GrpFixtures.scoreStart());
    Assert.assertEquals(switchTo(run, "grp", "--cutoff", "1e-5"), Exit.USAGE.code());
    Assert.assertEquals(switchTo(run, PhiEntry.INITIAL_ID, "--cutoff", "1e-5"), Exit.USAGE.code());
    Assert.assertEquals(switchTo(run, "late", "--cutoff", "0.02"), Exit.USAGE.code());
    Assert.assertEquals(switchTo(run, "late", "--cutoff", "-1e-6"), Exit.USAGE.code());
    Assert.assertFalse(Files.exists(files.modeHistory()));
    Assert.assertEquals(PhiHistory.read(files).size(), 1);
    // 打ち切り 0 は、全てを数え上げる
    Assert.assertEquals(switchTo(run, "late", "--cutoff", "0"), Exit.OK.code());
    Assert.assertEquals(PhiHistory.readModes(files), List.of(ModeLine.late(0, 3, "switch: grp-switch")));
  }

  @Test
  public void aLateLineOpensTheEntryWithQAndWarnsForOneWithout() throws Exception {
    Path leaf = GrpFixtures.network(run.resolve("leaf"), 11);
    Path q = GrpFixtures.outcomes(run.resolve("q"), GrpFixtures.VARIANT, 12);
    GrpFixtures.writeSettings(run, GrpFixtures.scoreStartAssignments());
    GrpFixtures.writeState(run, 3);
    PhiSetup setup = GrpFixtures.scoreStart();
    PhiHistory.initialize(files, setup);
    GrpFixtures.ready(files, "g002-a1", 2, leaf, q);
    PhiHistory.append(
        files, HandEndPhi.checkpointEntry("g002-a1", GrpFixtures.VARIANT, leaf, "", q, 6, PhiEntry.INITIAL_ID, "test"));

    // 反復 3 に効く g000 は q を持たないので、警告を出して書く
    Assert.assertEquals(switchTo(run, "late"), Exit.OK.code());
    List<String> warned = warnings();
    Assert.assertEquals(warned.size(), 1, warned.toString());
    Assert.assertTrue(
        warned.getFirst().contains("entry g000 in effect at iteration 3 has no outcome network"), warned.getFirst());
    Assert.assertEquals(switchTo(run, "grp", "--from", "5"), Exit.OK.code());
    // 反復 6 に効く候補は q を持つので、開いてみてから書き、警告は出さない
    Assert.assertEquals(switchTo(run, "late", "--from", "6"), Exit.OK.code());
    Assert.assertEquals(warnings(), List.of());
    Assert.assertEquals(
        PhiHistory.readModes(files),
        List.of(
            ModeLine.late(1e-5, 3, "switch: grp-switch"),
            ModeLine.grp(5, "switch: grp-switch"),
            ModeLine.late(1e-5, 6, "switch: grp-switch")));
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      PhiValue third = resolver.resolve(3, setup.phi()).value();
      Assert.assertEquals(third.requested(), PhiMode.LATE);
      Assert.assertEquals(third.mode(), PhiMode.GRP);
      ActivePhi sixth = resolver.resolve(6, setup.phi());
      Assert.assertEquals(sixth.entry().id(), "g002-a1");
      Assert.assertEquals(sixth.value().mode(), PhiMode.LATE);
      Assert.assertEquals((double) sixth.value().cutoff(), 1e-5);
    }

    // q が開けなければ、LATE の行は何も書かずに止める。GRP の行を足すときは q を開かない
    Files.writeString(q.resolve(GrpCheckpoint.WEIGHTS_FILE), "x", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
    Assert.assertEquals(switchTo(run, "grp", "--from", "7"), Exit.OK.code());
    Assert.assertEquals(switchTo(run, "late", "--from", "8"), Exit.FAILED.code());
    IllegalStateException refused =
        Assert.expectThrows(
            IllegalStateException.class, () -> PhiControl.switchMode(run, setup, PhiMode.LATE, 1e-6, 8, "test"));
    Assert.assertTrue(
        refused.getMessage().contains("entry g002-a1 in effect at iteration 8 cannot be opened in the LATE mode"),
        refused.getMessage());
    Assert.assertEquals(PhiHistory.readModes(files).size(), 4);
    Assert.assertEquals(PhiHistory.readModes(files).getLast(), ModeLine.grp(7, "switch: grp-switch"));
  }

  @Test
  public void aFixedGrpSwitchesItsModeOnly() throws Exception {
    Path leaf = GrpFixtures.network(run.resolve("leaf"), 11);
    Path q = GrpFixtures.outcomes(run.resolve("q"), GrpFixtures.VARIANT, 12);
    String[] assignments = {
      "phi.source=FIXED_GRP", "phi.grp=" + leaf, "phi.outcomes=" + q, "phi.late.from=3", "phi.late.cutoff=1e-4"
    };
    GrpFixtures.writeSettings(run, assignments);
    GrpFixtures.writeState(run, 2);
    // 学習の JVM が run を開く前でも、設定にある初めの LATE の行が先に書かれる
    Assert.assertEquals(switchTo(run, "grp", "--from", "6"), Exit.OK.code());
    ModeLine first = ModeLine.late(1e-4, 3, "phi.late.from = 3");
    ModeLine back = ModeLine.grp(6, "switch: grp-switch");
    Assert.assertEquals(PhiHistory.readModes(files), List.of(first, back));
    // 項目の切り替えと再学習ができるのは、自前の GRP の run だけ
    Assert.assertEquals(switchTo(run, PhiEntry.INITIAL_ID), Exit.USAGE.code());
    Assert.assertEquals(switchTo(run, PhiEntry.FIXED_ID), Exit.USAGE.code());
    Assert.assertEquals(PhiControlTest.prism("grp-refit", "--run", run.toString()), Exit.USAGE.code());
    Assert.assertEquals(switchTo(run, "late", "--from", "8"), Exit.OK.code());
    Assert.assertEquals(warnings(), List.of(), "the fixed GRP has q");
    Assert.assertEquals(switchTo(run, "late", "--from", "8"), Exit.OK.code());
    List<String> warned = warnings();
    Assert.assertEquals(warned.size(), 1, warned.toString());
    Assert.assertTrue(warned.getFirst().contains("LATE mode is already in effect at iteration 8"), warned.getFirst());
    ModeLine again = ModeLine.late(1e-4, 8, "switch: grp-switch");
    Assert.assertEquals(PhiHistory.readModes(files), List.of(first, back, again));
    Assert.assertFalse(Files.exists(files.phiHistory()), "a fixed GRP writes no Φ entries");

    // 学習の JVM は行を書き足さず、書かれた行のとおりのモードで値を求める
    PhiSetup setup = GrpFixtures.setup(assignments);
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      Assert.assertEquals(PhiHistory.readModes(files), List.of(first, back, again));
      int[] iterations = {2, 3, 6, 8};
      PhiMode[] modes = {PhiMode.GRP, PhiMode.LATE, PhiMode.GRP, PhiMode.LATE};
      for (int i = 0; i < iterations.length; i++) {
        ActivePhi active = resolver.resolve(iterations[i], setup.phi());
        Assert.assertEquals(active.entry().id(), PhiEntry.FIXED_ID);
        Assert.assertEquals(active.value().mode(), modes[i], "iteration " + iterations[i]);
      }
    }
    // q が開けなければ、LATE の行は書かない
    Files.writeString(q.resolve(GrpCheckpoint.WEIGHTS_FILE), "x", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
    Assert.assertEquals(switchTo(run, "late", "--from", "9", "--cutoff", "1e-6"), Exit.FAILED.code());
    Assert.assertEquals(PhiHistory.readModes(files), List.of(first, back, again));
  }

  @Test
  public void aFixedGrpWithoutQTakesTheLateLineWithAWarning() throws Exception {
    Path leaf = GrpFixtures.network(run.resolve("leaf"), 11);
    Path q = GrpFixtures.outcomes(run.resolve("q"), GrpFixtures.VARIANT, 12);

    // 使い始める反復が 0 なら、手で足した行だけを書く
    Path withQ = run.resolve("with-q");
    GrpFixtures.writeSettings(withQ, "phi.source=FIXED_GRP", "phi.grp=" + leaf, "phi.outcomes=" + q);
    Assert.assertEquals(switchTo(withQ, "late"), Exit.OK.code());
    Assert.assertEquals(
        PhiHistory.readModes(new GrpFiles(withQ)), List.of(ModeLine.late(1e-5, 1, "switch: grp-switch")));
    Assert.assertEquals(warnings(), List.of());

    Path plain = run.resolve("plain");
    String[] assignments = {"phi.source=FIXED_GRP", "phi.grp=" + leaf};
    GrpFixtures.writeSettings(plain, assignments);
    Assert.assertEquals(switchTo(plain, "late", "--cutoff", "1e-6"), Exit.OK.code());
    List<String> warned = warnings();
    Assert.assertEquals(warned.size(), 1, warned.toString());
    Assert.assertTrue(
        warned.getFirst().contains("entry fixed in effect at iteration 1 has no outcome network"), warned.getFirst());
    GrpFiles plainFiles = new GrpFiles(plain);
    Assert.assertEquals(PhiHistory.readModes(plainFiles), List.of(ModeLine.late(1e-6, 1, "switch: grp-switch")));
    PhiSetup setup = GrpFixtures.setup(assignments);
    try (PhiResolver resolver = PhiResolver.open(setup, plainFiles)) {
      PhiValue value = resolver.resolve(1, setup.phi()).value();
      Assert.assertEquals(value.requested(), PhiMode.LATE);
      Assert.assertEquals(value.mode(), PhiMode.GRP);
      Assert.assertNull(value.cutoff());
    }
  }

  @Test
  public void runsWithoutModesRefuseTheAliases() throws Exception {
    GrpFixtures.writeSettings(run, "phi.source=SCORE_RANK");
    Assert.assertEquals(switchTo(run, "late"), Exit.USAGE.code());
    Assert.assertEquals(switchTo(run, "grp"), Exit.USAGE.code());
    Assert.assertEquals(switchTo(run, "late", "--cutoff", "1e-6"), Exit.USAGE.code());
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> PhiControl.switchMode(run, GrpFixtures.setup("phi.source=SCORE_RANK"), PhiMode.LATE, null, null, "x"));
    Assert.assertFalse(Files.exists(files.directory()));
  }

  /** run に対して grp-switch を、切り替え先と追加の引数を付けて実行し、終了コードを返す。 */
  private static int switchTo(Path run, String to, String... more) {
    String[] command = {"grp-switch", "--run", run.toString(), "--to", to};
    return PhiControlTest.prism(GrpFixtures.concat(command, more));
  }

  /** 前の呼び出しの後に PhiControl が出した警告の文を、出した順に返す。 */
  private List<String> warnings() {
    List<String> warned =
        logs.list.stream()
            .filter(event -> event.getLevel() == Level.WARN && event.getLoggerName().equals(PhiControl.class.getName()))
            .map(ILoggingEvent::getFormattedMessage)
            .toList();
    logs.list.clear();
    return warned;
  }

  private static Logger root() {
    return (Logger) LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
  }
}
