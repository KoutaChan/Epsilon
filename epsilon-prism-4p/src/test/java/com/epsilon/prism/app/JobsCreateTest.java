package com.epsilon.prism.app;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
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
import java.util.List;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 公開したバージョンのジョブを作るテスト（jobs create）。最新の設定の行の卓で PENDING のジョブを作り、作った数と半荘の合計（計画の牌山を
 * 全て打たない止め方の卓では上限）を出す。何度作っても同じで、済んだジョブには触れない。公開していないバージョン、知らない卓、相手の
 * バージョンが公開していないバージョンになる卓、相手が王者で止め方が判定の卓、基準の卓の相手のバージョン自身は、英語の文で拒んで
 * ジョブを作らない。
 */
public class JobsCreateTest {

  private Path root;
  private RunDirectory run;
  private PrismSettings settings;
  private ListAppender<ILoggingEvent> logs;

  /** 卓は duel・early・ladder・probe・reference（名前の順）で、バージョン 1〜5 を公開した run。 */
  @BeforeMethod
  public void createRun() throws Exception {
    root = Files.createTempDirectory("prism-jobs-create");
    run = new RunDirectory(root);
    List<String> assignments = new ArrayList<>(List.of("phi.source=SCORE_RANK"));
    assignments.addAll(List.of(TestSettings.table("early", EvaluationTable.HEURISTIC, 4, 1, 1)));
    assignments.addAll(List.of(TestSettings.table("ladder", "version:-2", 8, 2, 3)));
    assignments.addAll(testTable("probe", EvaluationTable.HEURISTIC, false, 0, 1));
    assignments.addAll(testTable("duel", "champion", true, 1, 2));
    assignments.addAll(List.of(TestSettings.table("reference", "version:2", 4, 0, 3)));
    assignments.add("evaluation.tables.reference.onPromotion=false");
    settings = TestSettings.create(root, assignments.toArray(String[]::new));
    DurableFiles.write(run.state(), new PrismState(5, 0, PrismCarry.NONE, 5, Regression.NONE));
    logs = new ListAppender<>();
    logs.start();
    rootLogger().addAppender(logs);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws Exception {
    rootLogger().detachAppender(logs);
    Snapshot.deleteTree(root);
  }

  @Test
  public void createsPendingJobsWithTheLatestRowAndLogsTheirCountAndGames() throws Exception {
    Assert.assertEquals(create("early", "2..4"), Exit.OK.code());
    Node.Table early = RecordFile.tree(settings.evaluation().tables().get("early"));
    List<JobFile> created = jobs().list().jobs();
    Assert.assertEquals(
        created.stream().map(JobFile::name).toList(), List.of("v00002-early", "v00003-early", "v00004-early"));
    for (JobFile job : created) {
      Assert.assertEquals(job.state(), JobFile.State.PENDING, job.name());
      Assert.assertEquals(job.row(), 1, job.name());
      Assert.assertEquals(job.order(), 1, job.name());
      Assert.assertEquals(job.definition(), early, job.name());
      Assert.assertEquals(job.attempts(), 0, job.name());
    }
    // 1 牌山 1 半荘で 4 牌山のジョブが 3 つ
    Assert.assertEquals(
        lastCreated(),
        "Created 3 PENDING job(s) on table early for versions 2..4 with settings row 1 (0 existing job(s) left as they"
            + " are): 12 games");

    // 判定で止める卓の半荘は上限（1v3 と 3v1 の 8 つの割り当てを 1,024 牌山）
    Assert.assertEquals(create("probe", "1..2"), Exit.OK.code());
    Assert.assertEquals(
        lastCreated(),
        "Created 2 PENDING job(s) on table probe for versions 1..2 with settings row 1 (0 existing job(s) left as they"
            + " are): at most 16384 games");

    // 卓の値は、ジョブを作るときの最新の行から読む
    Assert.assertEquals(
        Main.run(
            new String[] {
              "settings", "change", "--run", root.toString(), "--change", "evaluation.tables.early.walls=8", "--reason",
              "test"
            }),
        Exit.OK.code());
    Assert.assertEquals(create("early", "5..5"), Exit.OK.code());
    JobFile latest = jobs().read("v00005-early");
    Assert.assertEquals(latest.row(), 2);
    Assert.assertEquals(RecordFile.read(TableSpec.class, latest.definition()).value().walls(), 8);
    Assert.assertEquals(
        lastCreated(),
        "Created 1 PENDING job(s) on table early for versions 5..5 with settings row 2 (0 existing job(s) left as they"
            + " are): 8 games");
  }

  @Test
  public void creatingTheSameJobsAgainChangesNothing() throws Exception {
    Assert.assertEquals(create("early", "1..3"), Exit.OK.code());
    List<String> before = jobFiles();
    Assert.assertEquals(create("early", "1..3"), Exit.OK.code());
    Assert.assertEquals(jobFiles(), before);
    Assert.assertEquals(
        lastCreated(),
        "Created 0 PENDING job(s) on table early for versions 1..3 with settings row 1 (3 existing job(s) left as they"
            + " are): 0 games");
  }

  @Test
  public void aDoneJobOfTheSameNameStaysAsItIs() throws Exception {
    JobFile pending = JobFile.pending(3, "early", 1, 1, RecordFile.tree(settings.evaluation().tables().get("early")));
    JobFile done =
        new JobFile(
            pending.name(),
            pending.version(),
            pending.table(),
            pending.order(),
            pending.row(),
            pending.definition(),
            null,
            JobFile.State.DONE,
            1,
            0,
            4,
            120,
            0,
            pending.created(),
            null);
    Path file = run.job(done.name()).resolve(Jobs.JOB);
    DurableFiles.write(file, done);
    String written = Files.readString(file, StandardCharsets.UTF_8);
    Assert.assertEquals(create("early", "2..4"), Exit.OK.code());
    Assert.assertEquals(Files.readString(file, StandardCharsets.UTF_8), written);
    Assert.assertEquals(
        jobs().list().jobs().stream().map(job -> job.name() + "=" + job.state()).toList(),
        List.of("v00002-early=PENDING", "v00003-early=DONE", "v00004-early=PENDING"));
    Assert.assertEquals(
        lastCreated(),
        "Created 2 PENDING job(s) on table early for versions 2..4 with settings row 1 (1 existing job(s) left as they"
            + " are): 8 games");
  }

  @Test
  public void unpublishedVersionsAndUnknownTablesAreRefusedInEnglishWithoutJobs() throws Exception {
    Assert.assertEquals(create("early", "4..6"), Exit.USAGE.code());
    Assert.assertTrue(
        logged("Version 6 is not published: the run has published versions 1..5"), messages().toString());
    Assert.assertEquals(create("nope", "1..2"), Exit.USAGE.code());
    Assert.assertTrue(
        logged("Unknown table nope in settings row 1; the tables are [duel, early, ladder, probe, reference]"),
        messages().toString());
    // 2 つ前のバージョンと打つ卓は、バージョン 2 の相手（バージョン 0）が無い
    Assert.assertEquals(create("ladder", "2..4"), Exit.USAGE.code());
    Assert.assertTrue(
        logged(
            "Table ladder plays version:-2, so versions 2..4 would play versions 0..2, but the run has published"
                + " versions 1..5"),
        messages().toString());
    Assert.assertEquals(create("duel", "2..3"), Exit.USAGE.code());
    Assert.assertTrue(
        logged(
            "Table duel duels the champion with a TEST stop: its results promote and roll back the lineage in the"
                + " order the versions are published, so it takes no jobs for past versions"),
        messages().toString());
    Assert.assertEquals(create("early", "3..2"), Exit.USAGE.code());
    Assert.assertFalse(Files.exists(run.jobs()), "a refused call creates no job");

    Files.delete(run.state());
    Assert.assertEquals(create("early", "1..1"), Exit.USAGE.code());
    Assert.assertTrue(
        logged("The run has published no version yet: " + run.state() + " is missing"), messages().toString());
    Assert.assertFalse(Files.exists(run.jobs()), "a refused call creates no job");

    // 相手のバージョンが全て公開していれば作る
    DurableFiles.write(run.state(), new PrismState(5, 0, PrismCarry.NONE, 5, Regression.NONE));
    Assert.assertEquals(create("ladder", "3..5"), Exit.OK.code());
    Assert.assertEquals(
        lastCreated(),
        "Created 3 PENDING job(s) on table ladder for versions 3..5 with settings row 1 (0 existing job(s) left as they"
            + " are): 24 games");
  }

  @Test
  public void aBaselineTableRefusesAJobForItsOwnOpponentVersion() throws Exception {
    // 基準の卓の相手はバージョン 2。範囲にバージョン 2 が入ると、そのジョブはバージョン 2 どうしの決闘になる
    for (String versions : List.of("2..2", "1..2", "2..3", "1..5")) {
      Assert.assertEquals(create("reference", versions), Exit.USAGE.code(), versions);
      Assert.assertTrue(
          logged("Table reference plays version:2, so version 2 would play itself"),
          versions + ": " + messages());
      Assert.assertFalse(Files.exists(run.jobs()), "a refused call creates no job: " + versions);
      logs.list.clear();
    }

    // 相手より前のバージョンも後のバージョンも、相手と別のバージョンなら作る
    Assert.assertEquals(create("reference", "1..1"), Exit.OK.code());
    Assert.assertEquals(create("reference", "3..5"), Exit.OK.code());
    Assert.assertEquals(
        jobs().list().jobs().stream().map(JobFile::name).toList(),
        List.of("v00001-reference", "v00003-reference", "v00004-reference", "v00005-reference"));
  }

  /**
   * 全ての割り当て（1v3 と 3v1 の 8 つ）で、新しい牌山を 1,024 まで打ち、512 牌山ごとに判定して止める卓。
   *
   * @param promote 良いなら王者を候補に替えるか
   * @param every 測るバージョンの間隔
   * @param from 測る最初のバージョン
   */
  private static List<String> testTable(String name, String opponent, boolean promote, int every, int from) {
    String prefix = "evaluation.tables." + name + ".";
    return List.of(
        prefix + "opponent=" + opponent,
        prefix + "seating={kind = \"EVERY_PLACEMENT\", splits = [\"1v3\", \"3v1\"]}",
        prefix + "draws=SEAT_STREAMS",
        prefix + "wallSet=FRESH",
        prefix + "walls=1024",
        prefix + "stop={kind = \"TEST\", evenMargin = 0.05, alpha = 0.05, spending = 1.0, checkEvery = 512, promote = "
            + promote + "}",
        prefix + "score=RANK",
        prefix + "every=" + every,
        prefix + "from=" + from);
  }

  /** {@code jobs create --run <run> --table <卓> --versions <範囲>} を実行し、終了コードを返す。 */
  private int create(String table, String versions) {
    return Main.run(
        new String[] {"jobs", "create", "--run", root.toString(), "--table", table, "--versions", versions});
  }

  private Jobs jobs() {
    return new Jobs(run.jobs(), run.report(RunDirectory.EVALUATIONS));
  }

  /** ジョブの記録の中身（ジョブの順）。 */
  private List<String> jobFiles() throws Exception {
    List<String> texts = new ArrayList<>();
    for (JobFile job : jobs().list().jobs()) {
      texts.add(Files.readString(run.job(job.name()).resolve(Jobs.JOB), StandardCharsets.UTF_8));
    }
    return texts;
  }

  private List<String> messages() {
    return logs.list.stream().map(ILoggingEvent::getFormattedMessage).toList();
  }

  private boolean logged(String line) {
    return messages().contains(line);
  }

  /** 最後に作ったジョブの数と半荘の合計の行。 */
  private String lastCreated() {
    List<String> created = messages().stream().filter(line -> line.startsWith("Created ")).toList();
    Assert.assertFalse(created.isEmpty(), messages().toString());
    return created.getLast();
  }

  private static Logger rootLogger() {
    return (Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
  }
}
