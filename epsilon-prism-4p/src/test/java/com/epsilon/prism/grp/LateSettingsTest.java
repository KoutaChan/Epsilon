package com.epsilon.prism.grp;

import com.epsilon.agent.app.Exit;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.lang.reflect.RecordComponent;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 終盤の数え上げの設定（{@link LateSettings}）の読み込みと検査のテスト。この設定を持つのは自前の GRP と固定の GRP だけで、既定は
 * 「使わない、打ち切り 1e-5、スロット 1,048,576」。使い始める反復と打ち切りは run の途中で変えられない。スロットの数はマシンごとに
 * 変えられる。固定の GRP の q は、変えると次の学習から効く。エラーメッセージは、キーと値を添えた英語の 1 文にする。check-config は、q があれば
 * LATE のモードでも開いてみて、葉が GrpNetwork でないこと、ルールの違う q、q でないチェックポイントを問題として返す。
 */
public class LateSettingsTest {

  private Path directory;

  @BeforeMethod
  public void createDirectory() throws IOException {
    directory = Files.createTempDirectory("prism-late-settings");
  }

  @AfterMethod(alwaysRun = true)
  public void deleteDirectory() throws IOException {
    Snapshot.deleteTree(directory);
  }

  @Test
  public void bindsUnderTheGrpSourcesOnly() {
    LateSettings defaults = new LateSettings(0, 1e-5, 1_048_576);
    PhiSettings self = TestSettings.of("phi.source=SELF_GRP").phi();
    Assert.assertEquals(((PhiSettings.SelfGrp) self).late(), defaults);
    Assert.assertEquals(self.lateSettings().orElseThrow(), defaults);
    PhiSettings fixed = TestSettings.of("phi.source=FIXED_GRP", "phi.grp=/grp/a").phi();
    Assert.assertEquals(fixed.lateSettings().orElseThrow(), defaults);
    Assert.assertNull(((PhiSettings.FixedGrp) fixed).outcomes());
    Assert.assertTrue(TestSettings.of("phi.source=SCORE_RANK").phi().lateSettings().isEmpty());
    Assert.assertTrue(TestSettings.of("phi.source=SCORE_SMOOTH").phi().lateSettings().isEmpty());

    PhiSettings set =
        TestSettings.of("phi.source=SELF_GRP", "phi.late.from=4", "phi.late.cutoff=1e-6", "phi.late.memo=4096").phi();
    Assert.assertEquals(set.lateSettings().orElseThrow(), new LateSettings(4, 1e-6, 4096));
    var withOutcomes =
        (PhiSettings.FixedGrp)
            TestSettings.of("phi.source=FIXED_GRP", "phi.grp=/grp/a", "phi.outcomes=/grp/q", "phi.late.from=2").phi();
    Assert.assertEquals(withOutcomes.outcomes(), Path.of("/grp/q"));
    Assert.assertEquals(withOutcomes.late().from(), 2);
    var start =
        (PhiSettings.SelfGrp)
            TestSettings.of(
                    "phi.source=SELF_GRP", "phi.start.kind=GRP_CHECKPOINT", "phi.start.grp=/grp/a",
                    "phi.start.outcomes=/grp/q")
                .phi();
    Assert.assertEquals(((PhiStartSettings.GrpCheckpoint) start.start()).outcomes(), Path.of("/grp/q"));
  }

  @Test
  public void startAndCutoffAreFixedForTheRunAndTheMemoIsPerMachine() {
    Assert.assertEquals(change(LateSettings.class, "from"), Change.FIXED);
    Assert.assertEquals(change(LateSettings.class, "cutoff"), Change.FIXED);
    Assert.assertEquals(change(LateSettings.class, "memo"), Change.MACHINE);
    Assert.assertEquals(change(PhiSettings.FixedGrp.class, "outcomes"), Change.FIT);
  }

  @Test
  public void invalidValuesNameTheirKeyInEnglish() {
    assertProblem(
        List.of("phi.source=SELF_GRP", "phi.late.memo=3"),
        "phi.late.memo=3",
        "the late memo needs a power of two slots, got 3");
    assertProblem(List.of("phi.source=SELF_GRP", "phi.late.cutoff=0.02"), "phi.late.cutoff=0.02", "a number");
    assertProblem(List.of("phi.source=SELF_GRP", "phi.late.from=-1"), "phi.late.from=-1", "");
    assertProblem(
        List.of("phi.source=FIXED_GRP", "phi.grp=/grp/a", "phi.late.from=2"),
        "phi.late.from=2",
        "the late mode needs the outcome network: set outcomes");
    assertProblem(
        List.of("phi.source=FIXED_GRP", "phi.grp=/grp/a", "phi.table=/grp/t.json", "phi.outcomes=/grp/q"),
        "phi.outcomes",
        "a Φ table overlays a GruSinkhorn only");
    assertProblem(
        List.of(
            "phi.source=SELF_GRP", "phi.start.kind=GRP_CHECKPOINT", "phi.start.grp=/grp/a",
            "phi.start.table=/grp/t.json", "phi.start.outcomes=/grp/q"),
        "phi.start.outcomes",
        "a Φ table overlays a GruSinkhorn only");
    assertProblem(
        List.of("phi.source=SCORE_RANK", "phi.late.from=1"), "phi.late", "does not apply to phi.source=SCORE_RANK");
    InvalidSetting memo = Assert.expectThrows(InvalidSetting.class, () -> new LateSettings(0, 1e-5, 12));
    Assert.assertEquals(memo.component(), "memo");
  }

  /**
   * スロットの数は、1 から 2^26 までの 2 の冪だけを受け付ける。キーの既定値は 1,048,576（2 の冪）。2^26 スロットの Φ の配列は、四人打ち
   * でも長さが int に収まる。0、負の数、2^26 を超える数は範囲のエラーになり、2 の冪でない数は 2 の冪でないというエラーになる。どちらも、
   * キーと値を添えた英語の 1 文のエラーメッセージで知らせる。
   */
  @Test
  public void theMemoTakesPowersOfTwoFromOne() {
    Key memo = key(LateSettings.class, "memo");
    Assert.assertEquals(memo.value(), "1048576");
    Assert.assertEquals(Integer.bitCount(Integer.parseInt(memo.value())), 1);
    Assert.assertEquals(TestSettings.of("phi.source=SELF_GRP").phi().lateSettings().orElseThrow().memo(), 1 << 20);
    Assert.assertEquals(LateSettings.MAX_MEMO, 1 << 26);
    Assert.assertTrue((long) LateSettings.MAX_MEMO * Variant.TENHOU_4P.seats() <= Integer.MAX_VALUE);
    for (int slots : new int[] {1, 2, 1 << 12, 1 << 20, 1 << 24, 1 << 26})
      Assert.assertEquals(
          TestSettings.of("phi.source=SELF_GRP", "phi.late.memo=" + slots).phi().lateSettings().orElseThrow().memo(),
          slots);
    for (String slots : List.of("0", "-4", "-2147483648"))
      assertProblem(List.of("phi.source=SELF_GRP", "phi.late.memo=" + slots), "phi.late.memo=" + slots, "");
    for (String slots : List.of("134217728", "536870912", "1073741824"))
      assertProblem(
          List.of("phi.source=SELF_GRP", "phi.late.memo=" + slots),
          "phi.late.memo=" + slots,
          "an integer in [1, 67108864]");
    for (String slots : List.of("3", "6", "1048575", "1048577"))
      assertProblem(
          List.of("phi.source=SELF_GRP", "phi.late.memo=" + slots),
          "phi.late.memo=" + slots,
          "the late memo needs a power of two slots, got " + slots);
    for (int slots : new int[] {0, -4, 3, 1_048_575})
      Assert.assertEquals(
          Assert.expectThrows(InvalidSetting.class, () -> new LateSettings(0, 1e-5, slots)).component(), "memo");
  }

  @Test
  public void checkConfigOpensTheOutcomeNetworkInTheLateMode() throws Exception {
    Path leaf = GrpFixtures.network(directory.resolve("leaf"), 1);
    Path q = GrpFixtures.outcomes(directory.resolve("q"), GrpFixtures.VARIANT, 2);
    Assert.assertEquals(PhiControl.problems(GrpFixtures.grpStart(leaf, "phi.start.outcomes=" + q)), List.of());
    Assert.assertEquals(PhiControl.problems(fixed(leaf, q)), List.of());
    // q を持っていれば、使い始める反復が 0 でも LATE のモードで開いてみる
    Path gru = GrpFixtures.gru(directory.resolve("gru"), 3);
    assertOneProblem(PhiControl.problems(fixed(gru, q)), "The late mode enumerates with a GrpNetwork leaf");
    Path threePlayer = GrpFixtures.outcomes(directory.resolve("q3"), Variant.TENHOU_3P, 4);
    assertOneProblem(PhiControl.problems(fixed(leaf, threePlayer)), "is TENHOU_3P, the entry TENHOU_4P");
    assertOneProblem(PhiControl.problems(fixed(leaf, leaf)), "not an");
    assertOneProblem(
        PhiControl.problems(GrpFixtures.grpStart(leaf, "phi.start.outcomes=" + threePlayer)), "is TENHOU_3P");

    Assert.assertEquals(check("late.toml", fixedToml(leaf, q) + "\n[phi.late]\nfrom = 2\n"), Exit.OK.code());
    Assert.assertEquals(check("gru.toml", fixedToml(gru, q)), Exit.PROBLEMS.code());
    Assert.assertEquals(check("missing.toml", fixedToml(leaf, null) + "\n[phi.late]\nfrom = 2\n"), Exit.PROBLEMS.code());
  }

  private static PhiSetup fixed(Path grp, Path outcomes) {
    return GrpFixtures.setup("phi.source=FIXED_GRP", "phi.grp=" + grp, "phi.outcomes=" + outcomes);
  }

  private static String fixedToml(Path grp, Path outcomes) {
    return "[phi]\nsource = \"FIXED_GRP\"\ngrp = \"" + grp + "\"\n"
        + (outcomes == null ? "" : "outcomes = \"" + outcomes + "\"\n");
  }

  /** 本文を設定ファイルに書いて {@code settings check} を実行し、終了コードを返す。 */
  private int check(String name, String body) throws IOException {
    Path file = Files.writeString(directory.resolve(name), body + "\n[run]\nvariant = \"TENHOU_4P\"\nseed = 1\n");
    return PhiControlTest.prism("settings", "check", "--settings", file.toString());
  }

  private static Change change(Class<? extends Record> type, String name) {
    return key(type, name).change();
  }

  private static Key key(Class<? extends Record> type, String name) {
    for (RecordComponent component : type.getRecordComponents())
      if (component.getName().equals(name)) return component.getAnnotation(Key.class);
    throw new AssertionError("No component " + name + " in " + type);
  }

  /** 設定の読み込みがエラーを 1 つだけ返し、そのメッセージがキー（と値）と理由を英語で書いていること。 */
  private static void assertProblem(List<String> assignments, String key, String reason) {
    SettingsException failure =
        Assert.expectThrows(SettingsException.class, () -> TestSettings.of(assignments.toArray(String[]::new)));
    Assert.assertEquals(failure.problems().size(), 1, failure.getMessage());
    String problem = failure.problems().getFirst();
    Assert.assertTrue(problem.contains(key), problem);
    Assert.assertTrue(problem.contains(reason), problem);
    assertEnglish(problem);
  }

  private static void assertOneProblem(List<String> problems, String part) {
    Assert.assertEquals(problems.size(), 1, problems.toString());
    Assert.assertTrue(problems.getFirst().contains(part), problems.getFirst());
    assertEnglish(problems.getFirst());
  }

  /** 日本語の文字（かな・漢字）を含まないこと。 */
  private static void assertEnglish(String text) {
    Assert.assertTrue(text.codePoints().noneMatch(c -> c >= 0x3040 && c <= 0x9FFF), text);
  }
}
