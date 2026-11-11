package com.epsilon.prism.app;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.settings.Change;
import com.epsilon.prism.Main;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.eval.EvaluationTable;
import com.epsilon.prism.run.PrismCarry;
import com.epsilon.prism.run.PrismState;
import com.epsilon.prism.run.Regression;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.settings.PrismSettings;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 続いている run の設定を変えるテスト（settings change）。足した行は次に始まる反復から効き、変えたキーの前後の値と、いつから効くかの種類を残す。
 * 種類はセクションの中でもキーごとに決まる（推論の計算の型は収集から、学習器の計算の型は学習から、バッチの行数の上限は起動し直してから）。
 * 変えてはいけないキー（乱数の種、ネットワークの形、Φ の出どころ）を変えようとすると終了コード 2 で拒み、行を足さない。評価の卓を足すことと
 * 卓のキーを変えることは、次に始まる反復から効く SCHEDULE の変更になる。変更は、次に公開するバージョンのジョブが増える数・減る数・変わる数を
 * 出す。
 */
public class SettingsChangeTest {

  private Path run;

  @BeforeMethod
  public void createRun() throws Exception {
    run = Files.createTempDirectory("prism-settings-change");
    TestSettings.create(
        run,
        TestSettings.concat(
            new String[] {"phi.source=SCORE_RANK"}, TestSettings.table("early", EvaluationTable.HEURISTIC, 4, 1, 1)));
    // バージョン 1〜3 を公開済みで、反復 3 をまだ始めていない run
    DurableFiles.write(
        new RunDirectory(run).state(), new PrismState(3, 0, PrismCarry.NONE, 3, Regression.NONE));
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws Exception {
    Snapshot.deleteTree(run);
  }

  @Test
  public void rowsTakeEffectFromTheNextIterationAndRecordEachKeysChange() throws Exception {
    Assert.assertEquals(change("fit.lrPeak=0.002"), Exit.OK.code());
    List<SettingsRow> rows = rows();
    Assert.assertEquals(rows.size(), 2);
    SettingsRow row = rows.get(1);
    Assert.assertEquals(row.from(), 3);
    Assert.assertEquals(row.reason(), "test");
    Assert.assertEquals(changes(row), Map.of("fit.lrPeak", Change.FIT));
    Assert.assertEquals(PrismApp.latest(run).fit().lrPeak(), 0.002);

    // 同じセクションの中でも、キーごとに効き始めが違う
    Assert.assertEquals(
        change(
            "inference.precision=BFLOAT16",
            "inference.maxBatch=64",
            "learner.devices=[\"gpu0\"]",
            "learner.precision=BFLOAT16",
            "labels.maxCandidates=4",
            "schedule.cohorts=[{from = 1, games = 16}]",
            "noise.tempo.gain=0.8",
            "noise.rates.table.CALL=0.2",
            "table.workers=4"),
        Exit.OK.code());
    rows = rows();
    Assert.assertEquals(rows.size(), 3);
    Map<String, Change> expected = new LinkedHashMap<>();
    expected.put("inference.maxBatch", Change.MACHINE);
    expected.put("inference.precision", Change.COLLECT);
    expected.put("labels.maxCandidates", Change.COLLECT);
    expected.put("learner.devices", Change.MACHINE);
    expected.put("learner.precision", Change.FIT);
    expected.put("noise.rates.table.CALL", Change.COLLECT);
    expected.put("noise.tempo.gain", Change.FIT);
    expected.put("schedule.cohorts", Change.COLLECT);
    expected.put("table.workers", Change.MACHINE);
    Assert.assertEquals(changes(rows.get(2)), expected);
    Assert.assertEquals(rows.get(2).from(), 3);
    PrismSettings latest = PrismApp.latest(run);
    Assert.assertEquals(latest.inference().maxBatch(), 64);
    Assert.assertEquals(latest.fit().lrPeak(), 0.002, "the earlier row stays under the later one");

    // 同じ値なら行を足さない
    Assert.assertEquals(change("fit.lrPeak=0.002"), Exit.OK.code());
    Assert.assertEquals(rows().size(), 3);
  }

  @Test
  public void fixedKeysAreRefusedWithoutARow() throws Exception {
    String before = Files.readString(new RunDirectory(run).settings(), StandardCharsets.UTF_8);
    for (String fixed : new String[] {"run.seed=5", "network.depth=3", "phi.source=SELF_GRP"}) {
      Assert.assertEquals(change(fixed), Exit.USAGE.code(), fixed);
      Assert.assertEquals(change("fit.lrPeak=0.003", fixed), Exit.USAGE.code(), fixed + " with a FIT key");
    }
    Assert.assertEquals(Files.readString(new RunDirectory(run).settings(), StandardCharsets.UTF_8), before);
    // 学習の JVM は --set を受け付けない
    Assert.assertEquals(
        Main.run(new String[] {"train", "--run", run.toString(), "--set", "fit.lrPeak=0.002"}), Exit.USAGE.code());
  }

  @Test
  public void aTableAddedLaterIsAScheduleChangeFromTheNextIteration() throws Exception {
    Assert.assertEquals(change(TestSettings.table("late", EvaluationTable.HEURISTIC, 4, 1, 1)), Exit.OK.code());
    SettingsRow row = rows().getLast();
    Assert.assertEquals(row.from(), 3);
    Assert.assertTrue(changes(row).values().stream().allMatch(kind -> kind == Change.SCHEDULE), changes(row).toString());
    Assert.assertEquals(PrismApp.latest(run).evaluation().tables().keySet(), Set.of("early", "late"));
    Assert.assertEquals(change("evaluation.tables.early.walls=8"), Exit.OK.code());
    Assert.assertEquals(changes(rows().getLast()), Map.of("evaluation.tables.early.walls", Change.SCHEDULE));
    Assert.assertEquals(PrismApp.latest(run).evaluation().tables().get("early").walls(), 8);
  }

  /**
   * 変更が効く最初の反復が公開するバージョン（ここでは 4）のジョブを、変更の前と後で比べた数を出す。評価の卓のキーの変更だけが数を動かし、
   * 学習と収集のキーの変更では全て 0 になる。
   */
  @Test
  public void aChangeCountsTheJobsItAddsRemovesAndChangesAtTheNextPublishedVersion() throws Exception {
    Assert.assertEquals(
        jobsLine("fit.lrPeak=0.002"),
        "Version 4 is the first published with row 2: 1 job(s) instead of 1 (0 added [], 0 removed [], 0 changed [])");
    Assert.assertEquals(
        jobsLine("labels.maxCandidates=4"),
        "Version 4 is the first published with row 3: 1 job(s) instead of 1 (0 added [], 0 removed [], 0 changed [])");
    Assert.assertEquals(
        jobsLine(TestSettings.table("late", EvaluationTable.HEURISTIC, 4, 1, 1)),
        "Version 4 is the first published with row 4: 2 job(s) instead of 1 (1 added [late], 0 removed [], 0 changed"
            + " [])");
    // 卓 late はバージョン 4 を測らなくなり、卓 early は牌山の数が変わる
    Assert.assertEquals(
        jobsLine("evaluation.tables.early.walls=8", "evaluation.tables.late.every=3"),
        "Version 4 is the first published with row 5: 1 job(s) instead of 2 (0 added [], 1 removed [late], 1 changed"
            + " [early])");
  }

  /** 自前の GRP の運用と学習のキーは、起動し直してから効くキーとして行に残る。GRP のバックグラウンドの処理は、起動したときの設定の行で始まるため。 */
  @Test
  public void selfGrpOperationsTakeEffectAfterARestart() throws Exception {
    Snapshot.deleteTree(run);
    TestSettings.create(run, "phi.source=SELF_GRP", "phi.start.kind=SCORE_RANK");
    Assert.assertEquals(change("phi.grp.training.warmEpochs=3", "phi.grp.halfLife=3.0"), Exit.OK.code());
    Assert.assertEquals(
        changes(rows().getLast()),
        Map.of("phi.grp.halfLife", Change.MACHINE, "phi.grp.training.warmEpochs", Change.MACHINE));
  }

  /** {@code settings change --run <run> --change <上書き>... --reason test} を実行し、終了コードを返す。 */
  private int change(String... assignments) {
    List<String> arguments = new ArrayList<>(List.of("settings", "change", "--run", run.toString()));
    for (String assignment : assignments) {
      arguments.add("--change");
      arguments.add(assignment);
    }
    arguments.addAll(List.of("--reason", "test"));
    return Main.run(arguments.toArray(String[]::new));
  }

  /** {@link #change} を実行し、ジョブの数を比べたログの行を返す。 */
  private String jobsLine(String... assignments) {
    Logger root = (Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
    ListAppender<ILoggingEvent> logs = new ListAppender<>();
    logs.start();
    root.addAppender(logs);
    try {
      Assert.assertEquals(change(assignments), Exit.OK.code());
    } finally {
      root.detachAppender(logs);
    }
    List<String> lines =
        logs.list.stream().map(ILoggingEvent::getFormattedMessage).filter(line -> line.startsWith("Version ")).toList();
    Assert.assertEquals(lines.size(), 1, lines.toString());
    return lines.getFirst();
  }

  private List<SettingsRow> rows() throws Exception {
    return new SettingsRows(new RunDirectory(run)).read();
  }

  /** 行で変えたキーと、それぞれがいつから効くかの種類を、キーの名前の順に返す。 */
  private static Map<String, Change> changes(SettingsRow row) {
    Map<String, Change> changes = new LinkedHashMap<>();
    row.changes().stream()
        .sorted(Comparator.comparing(SettingsRow.ChangedKey::key))
        .forEach(changed -> changes.put(changed.key(), changed.change()));
    return changes;
  }
}
