package com.epsilon.prism.run;

import com.epsilon.agent.checkpoint.CheckpointFiles;
import com.epsilon.agent.eval.Alarm;
import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.run.ControlRequests;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.Ending;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.Request;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.RunStatus;
import com.epsilon.agent.run.Runner;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.eval.EvaluationTable;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.settings.PrismSettings;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.BooleanSupplier;
import java.util.stream.Stream;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 系譜を止めて続きから回すテスト（CPU）。同じ設定の run を 2 つ作る。片方は止めずに最後まで回す。もう片方は、段の前と後・反復を閉じた後・
 * ジョブの前と後・起動時のセーフポイントで 1 回ずつ止めては起動し直す。止めた所には、落ちたときに残るファイルを置く。置くのは、段の
 * 書きかけ（コホート・学習・バージョンの .part）、記録より長い games.tsv と looks.tsv、反復の報告の書きかけの行。どこで止めても、
 * 反復の報告・ノイズの表・ジョブの計画と半荘と結果・state・王者の記録・バージョンの重み・コホートの行は、止めない run と同じになる。
 * 時刻・秒数・リソース、段を回したか使い回したか、コホートの行の順は比べない。
 *
 * <p>収集で打つバージョンは全て初期化したままのネットワーク（s ≡ 0。学習のステップ数は 0）なので、収集と評価の半荘は推論のバッチの
 * 組み方に依らない。途中で、COLLECT の済んだ反復を別の jar で続けようとすると拒まれる（status.json に FAILED と理由が入る）。
 */
public class LineageResumeTest {

  private static final String JAR = "test0001";

  private static final int MAX_LAUNCHES = 80;

  /** 段の後で止めたとき、次の起動で一時停止の要求を出して、その前で止める段。 */
  private static final Map<String, String> NEXT_STEP =
      Map.of("AFTER_STEP PHI", "COLLECT", "AFTER_STEP COLLECT", "LEARN", "AFTER_STEP LEARN", "PUBLISH");

  @Test(groups = "slow", timeOut = 3_600_000)
  public void stoppingAtEverySafePointAndResumingGivesTheSameRun() throws Exception {
    Path reference = Files.createTempDirectory("prism-resume-reference");
    Path stopped = Files.createTempDirectory("prism-resume-stopped");
    try {
      TestSettings.create(reference, settings());
      TestSettings.create(stopped, settings());
      Assert.assertEquals(runner(reference, JAR, () -> false).train(), Ending.FINISHED);
      Outcome expected = outcome(reference);
      Assert.assertEquals(expected.iterations().size(), 2);
      Assert.assertEquals(expected.evaluations().size(), 7, expected.evaluations().toString());

      List<String> points = stopEverywhere(stopped);
      for (String step : List.of("PHI", "COLLECT", "LEARN", "PUBLISH")) {
        Assert.assertEquals(Collections.frequency(points, "BEFORE_STEP " + step), 2, step + " in " + points);
        Assert.assertEquals(Collections.frequency(points, "AFTER_STEP " + step), 2, step + " in " + points);
      }
      Assert.assertEquals(Collections.frequency(points, "ITERATION"), 2, points.toString());
      Assert.assertEquals(Collections.frequency(points, "LAUNCH"), 1, points.toString());
      for (String job : expected.jobs().keySet()) {
        Assert.assertEquals(Collections.frequency(points, "AFTER_JOB " + job), 1, job + " in " + points);
      }
      Assert.assertEquals(RunStatus.read(new RunDirectory(stopped).status()).state(), RunStatus.State.FINISHED);
      Assert.assertEquals(outcome(stopped), expected);
      assertNoLeftovers(new RunDirectory(stopped));
    } finally {
      Snapshot.deleteTree(reference);
      Snapshot.deleteTree(stopped);
    }
  }

  /**
   * 収集の済んだ反復を止めている間に、王者との決闘が王者へ戻すと決めた run を続ける。その反復は、集めたバージョンから学習して公開する。
   * 戻しは次の反復から効き、その反復は王者のバージョンで集めて、そこから学習する。
   */
  @Test(groups = "slow", timeOut = 3_600_000)
  public void aRollbackDecidedAfterTheCollectionStartsAtTheNextIteration() throws Exception {
    Path root = Files.createTempDirectory("prism-resume-rollback");
    try {
      TestSettings.create(
          root,
          TestSettings.cpuLineage(
              "phi.source=SCORE_RANK",
              "schedule.cohorts=[{from = 1, games = 1}]",
              "schedule.iterations=3",
              "fit.maxSteps=0"));
      RunDirectory run = new RunDirectory(root);
      Runner.Options once = new Runner.Options(JAR, 1, null, () -> false);
      Assert.assertEquals(new Runner<>(run, PrismRun.LINEAGE, once).train(), Ending.LIMIT);
      ControlRequests requests = new ControlRequests(run);
      requests.submit(new Request.Pause("LEARN"));
      Assert.assertEquals(runner(root, JAR, () -> paused(run)).train(), Ending.STOPPED);
      requests.submit(new Request.Resume());
      Assert.assertTrue(Files.isDirectory(run.rows(2)), "iteration 2 stopped after its collection");
      // 起動のときに決着した王者との決闘が、王者（バージョン 1）へ戻すと決めたときの state
      PrismState stopped = RecordFile.read(PrismState.class, run.state()).value();
      Assert.assertEquals(stopped.next(), 2);
      Assert.assertEquals(stopped.start(), 2);
      DurableFiles.write(run.state(), stopped.regressed(stopped.regression(), 1));

      Assert.assertEquals(runner(root, JAR, () -> false).train(), Ending.FINISHED);
      List<String> iterations = DurableFiles.lines(run.report(RunDirectory.ITERATIONS));
      Assert.assertEquals(iterations.size(), 3);
      Assert.assertEquals(starts(iterations.get(1)), List.of(2, 2), "iteration 2 learns from the version it collected");
      Assert.assertEquals(starts(iterations.get(2)), List.of(1, 1), "iteration 3 starts from the champion");
      PrismState closed = RecordFile.read(PrismState.class, run.state()).value();
      Assert.assertEquals(closed.next(), 4);
      Assert.assertEquals(closed.start(), 4);
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /** 反復の報告の行から、収集の段と学習の段の始めのバージョンを、この順に返す。 */
  private static List<Integer> starts(String line) {
    JsonObject steps = JsonParser.parseString(line).getAsJsonObject().getAsJsonObject("steps");
    return List.of(
        steps.getAsJsonObject(PrismStep.COLLECT.name()).get("start").getAsInt(),
        steps.getAsJsonObject(PrismStep.LEARN.name()).get("start").getAsInt());
  }

  /**
   * CPU の小さな系譜（{@link TestSettings#cpuLineage}）の設定。Φ は点数の順位、1 コホート 1 半荘、2 反復、学習のステップ数 0 で、
   * バージョン 2 から方策の EMA を使う。卓はルールベースの打ち手（バージョン 1 から）、1 つ前のバージョン、王者（どちらもバージョン 2
   * から）の 3 つで、どれも 4 つの牌山を全て打つ。
   */
  private static String[] settings() {
    String[] lineage =
        TestSettings.cpuLineage(
            TestSettings.concat(
                TestSettings.EMA_FROM_2,
                "phi.source=SCORE_RANK",
                "schedule.cohorts=[{from = 1, games = 1}]",
                "schedule.iterations=2",
                "fit.maxSteps=0"));
    String[] tables =
        TestSettings.concat(
            TestSettings.concat(
                TestSettings.table("x1n3", EvaluationTable.HEURISTIC, 4, 1, 1),
                TestSettings.table("ladder", "version:-1", 4, 1, 2)),
            TestSettings.table("champ", "champion", 4, 1, 2));
    return TestSettings.concat(lineage, tables);
  }

  private static Runner<PrismSettings, PrismState, PrismIteration, PrismStep> runner(
      Path run, String jar, BooleanSupplier signalled) {
    return new Runner<>(new RunDirectory(run), PrismRun.LINEAGE, new Runner.Options(jar, null, null, signalled));
  }

  /**
   * 止めては続けて、run を最後まで回す。止めるのは、前に止めた所からファイルが進んだ後の最初のセーフポイント。段の後で止めた次の起動では、
   * 一時停止の要求で次の段の前に待たせて止める。
   *
   * @return 止まったセーフポイント（起きた順）
   */
  private static List<String> stopEverywhere(Path root) throws Exception {
    RunDirectory run = new RunDirectory(root);
    ControlRequests requests = new ControlRequests(run);
    Set<String> visited = new HashSet<>();
    List<String> points = new ArrayList<>();
    String before = null;
    for (int launch = 1; launch <= MAX_LAUNCHES; launch++) {
      Ending ending;
      if (before != null) {
        requests.submit(new Request.Pause(before));
        ending = runner(root, JAR, () -> paused(run)).train();
        requests.submit(new Request.Resume());
      } else {
        ending = runner(root, JAR, () -> visited.add(progress(run))).train();
      }
      RunStatus status = RunStatus.read(run.status());
      if (ending == Ending.FINISHED) {
        return points;
      }
      Assert.assertEquals(ending, Ending.STOPPED, points.toString());
      String reason = status.reason();
      Assert.assertTrue(reason.startsWith("Stopped at ") && reason.endsWith(" by a signal"), reason);
      String point = reason.substring("Stopped at ".length(), reason.length() - " by a signal".length());
      if (before != null) {
        Assert.assertEquals(point, "BEFORE_STEP " + before, "the pause request waits before its step");
      }
      points.add(point);
      leaveTraces(run, point, status.iteration() == null ? 0 : status.iteration());
      before = NEXT_STEP.get(point);
    }
    throw new AssertionError("The run did not finish in " + MAX_LAUNCHES + " launches: " + points);
  }

  /** 止めた所に、落ちたときに残るファイルを置く。 */
  private static void leaveTraces(RunDirectory run, String point, int t) throws Exception {
    switch (point) {
      case "BEFORE_STEP COLLECT" -> junk(PrismStep.part(run.rows(t)));
      case "BEFORE_STEP LEARN" -> {
        junk(PrismStep.part(run.work(t).resolve(LearnStep.LEARN)));
        if (t == 1) {
          // COLLECT の済んだ反復は、その反復を始めた jar でだけ続けられる
          Assert.assertEquals(runner(run.root(), "other002", () -> false).train(), Ending.REFUSED);
          RunStatus refused = RunStatus.read(run.status());
          Assert.assertEquals(refused.state(), RunStatus.State.FAILED);
          Assert.assertEquals(refused.reason(), "Iteration 1 started with jar " + JAR + "; restore that jar to finish it");
        }
      }
      case "BEFORE_STEP PUBLISH" -> junk(PrismStep.part(run.snapshot(t + 1)));
      case "AFTER_STEP PUBLISH" ->
          Files.writeString(
              run.report(RunDirectory.ITERATIONS),
              "{\"iteration\": 99, \"trunc",
              StandardCharsets.UTF_8,
              StandardOpenOption.CREATE,
              StandardOpenOption.APPEND);
      case "ITERATION" -> {
        JobFile next = new Jobs(run.jobs(), run.report(RunDirectory.EVALUATIONS)).list().jobs().stream()
            .filter(JobFile::open)
            .findFirst()
            .orElseThrow();
        Files.writeString(run.job(next.name()).resolve(JobOutputs.GAMES), "0\t0\t0\t1,2,3,4\t0\t0\n");
        Files.writeString(run.job(next.name()).resolve(JobOutputs.LOOKS), "junk\n");
      }
      default -> {}
    }
  }

  /** 書きかけのディレクトリを作り、中に要らないファイルを 1 つ置く。 */
  private static void junk(Path part) throws IOException {
    Files.createDirectories(part);
    Files.writeString(part.resolve("junk"), "junk");
  }

  /** status.json が一時停止を示すか。 */
  private static boolean paused(RunDirectory run) {
    try {
      return RunStatus.read(run.status()).state() == RunStatus.State.PAUSED;
    } catch (IOException unreadable) {
      throw new UncheckedIOException(unreadable);
    }
  }

  /**
   * run の進み具合を表す文字列。state、バージョン・コホート・作業ディレクトリのファイル、報告の行の数、ジョブの状態と打ち終えた牌山を並べる。
   * 書きかけの .part とジョブの出力は数えない。
   */
  private static String progress(RunDirectory run) {
    try {
      StringBuilder out = new StringBuilder();
      out.append(Files.exists(run.state()) ? Files.readString(run.state(), StandardCharsets.UTF_8) : "-").append('|');
      out.append(names(run.snapshots())).append('|').append(names(run.rows(1).getParent())).append('|');
      for (String t : names(run.work())) {
        out.append(t).append(names(run.work().resolve(t))).append('|');
      }
      out.append(DurableFiles.lines(run.report(RunDirectory.ITERATIONS)).size()).append('|');
      for (JobFile job : new Jobs(run.jobs(), run.report(RunDirectory.EVALUATIONS)).list().jobs()) {
        out.append(job.name()).append(' ').append(job.state()).append(' ').append(job.walls()).append(',');
      }
      return out.toString();
    } catch (IOException unreadable) {
      throw new UncheckedIOException(unreadable);
    }
  }

  /** ディレクトリの中の名前を、.part を除いて名前の順に返す。ディレクトリが無ければ空。 */
  private static List<String> names(Path directory) throws IOException {
    if (!Files.isDirectory(directory)) {
      return List.of();
    }
    try (Stream<Path> listed = Files.list(directory)) {
      return listed
          .map(path -> path.getFileName().toString())
          .filter(name -> !name.endsWith(PrismStep.PART))
          .sorted()
          .toList();
    }
  }

  /** 閉じた反復の作業ディレクトリと、段の書きかけが残っていないことを確かめる。 */
  private static void assertNoLeftovers(RunDirectory run) throws IOException {
    Assert.assertEquals(names(run.work()), List.of());
    for (Path directory : List.of(run.snapshots(), run.rows(1).getParent())) {
      try (Stream<Path> listed = Files.list(directory)) {
        Assert.assertEquals(
            listed.filter(path -> path.getFileName().toString().endsWith(PrismStep.PART)).toList(), List.of());
      }
    }
  }

  /**
   * 比べる run の結果。時刻・秒数・リソース、収集の Φ の呼び出しの数、段を回したか使い回したか、収集と同じ JVM で帳簿を当てはめたか、
   * ジョブの試行の数は除く。run のディレクトリの場所は {@code <run>} に置き換える。
   */
  private record Outcome(
      String state,
      List<String> iterations,
      List<String> noise,
      List<String> evaluations,
      List<String> alarms,
      Map<String, String> jobs,
      Map<String, String> snapshots,
      Map<String, String> cohorts,
      List<String> champion) {}

  private static Outcome outcome(Path root) throws Exception {
    RunDirectory run = new RunDirectory(root);
    String place = run.root().toString();
    List<String> iterations = new ArrayList<>();
    for (String line : DurableFiles.lines(run.report(RunDirectory.ITERATIONS))) {
      JsonObject report = JsonParser.parseString(line).getAsJsonObject();
      report.remove("started");
      report.remove("seconds");
      report.remove("resources");
      for (Map.Entry<String, JsonElement> step : report.getAsJsonObject("steps").entrySet()) {
        JsonObject metrics = step.getValue().getAsJsonObject();
        metrics.remove("seconds");
        metrics.remove("decisionsPerSecond");
        metrics.remove("waitSeconds");
        // Φ の呼び出しの数は、時間やメモのヒットがスレッドの回り方で変わるので比べない
        metrics.remove("phi");
        metrics.addProperty("status", "DONE");
        for (String part : List.of("ledger", "fit")) {
          if (metrics.has(part)) {
            metrics.getAsJsonObject(part).remove("seconds");
            metrics.getAsJsonObject(part).remove("streamed");
          }
        }
      }
      iterations.add(report.toString());
    }
    List<String> evaluations = new ArrayList<>();
    for (String line : DurableFiles.lines(run.report(RunDirectory.EVALUATIONS))) {
      evaluations.add(withoutSeconds(line));
    }
    Map<String, String> jobs = new TreeMap<>();
    for (JobFile job : new Jobs(run.jobs(), run.report(RunDirectory.EVALUATIONS)).list().jobs()) {
      Path home = run.job(job.name());
      JsonObject file = JsonParser.parseString(Files.readString(home.resolve(Jobs.JOB))).getAsJsonObject();
      file.remove("attempts");
      file.remove("cleanStops");
      file.remove("created");
      StringBuilder text = new StringBuilder(file.toString());
      for (String output : List.of(JobOutputs.GAMES, JobOutputs.LOOKS)) {
        Path path = home.resolve(output);
        text.append('\n').append(Files.exists(path) ? Files.readString(path) : "");
      }
      text.append('\n').append(withoutSeconds(Files.readString(home.resolve(Jobs.RESULT)).strip()));
      jobs.put(job.name(), text.toString().replace(place, "<run>"));
    }
    Map<String, String> snapshots = new TreeMap<>();
    for (String version : names(run.snapshots())) {
      snapshots.put(version, CheckpointFiles.sha256(run.snapshots().resolve(version)));
    }
    Map<String, String> cohorts = new TreeMap<>();
    for (String cohort : names(run.rows(1).getParent())) {
      cohorts.put(cohort, digest(run.rows(1).getParent().resolve(cohort)));
    }
    List<String> champion = new ArrayList<>();
    for (ChampionStore.Entry entry : ChampionStore.open(run.champion()).history()) {
      champion.add(entry.event().toString().replace(place, "<run>"));
    }
    return new Outcome(
        Files.readString(run.state(), StandardCharsets.UTF_8),
        iterations,
        DurableFiles.lines(run.report(NoiseLine.FILE)),
        evaluations,
        DurableFiles.lines(run.report(Alarm.FILE)),
        jobs,
        snapshots,
        cohorts,
        champion);
  }

  private static String withoutSeconds(String line) {
    JsonObject object = JsonParser.parseString(line).getAsJsonObject();
    object.remove("seconds");
    return object.toString();
  }

  /**
   * コホートのマニフェスト、クラスごとの数、行（インデックスの項目と本体の組）の多重集合から求めた SHA-256。ライターは行を届いた順に
   * 書き、その順は収集のスレッドの速さで変わるので、行の順は比べない。
   */
  private static String digest(Path cohort) throws Exception {
    CohortWriter.Manifest manifest = CohortWriter.readManifest(cohort);
    List<String> rows = new ArrayList<>();
    for (CohortWriter.Shard shard : manifest.shards()) {
      Path file = cohort.resolve(shard.file());
      byte[] body = Files.readAllBytes(file);
      ByteBuffer index = ByteBuffer.wrap(Files.readAllBytes(CohortWriter.index(file))).order(ByteOrder.LITTLE_ENDIAN);
      int offset = 0;
      while (index.hasRemaining()) {
        byte[] entry = new byte[CohortWriter.INDEX_ENTRY];
        index.get(entry);
        int length = ByteBuffer.wrap(entry).order(ByteOrder.LITTLE_ENDIAN).getInt();
        MessageDigest frame = MessageDigest.getInstance("SHA-256");
        frame.update(body, offset, length);
        rows.add(HexFormat.of().formatHex(entry) + HexFormat.of().formatHex(frame.digest()));
        offset += length;
      }
      Assert.assertEquals(offset, body.length, file.toString());
    }
    Collections.sort(rows);
    MessageDigest sha = MessageDigest.getInstance("SHA-256");
    sha.update(Files.readAllBytes(cohort.resolve(CohortWriter.MANIFEST)));
    sha.update(Files.readAllBytes(cohort.resolve(CollectReport.CLASSES)));
    for (String row : rows) {
      sha.update(row.getBytes(StandardCharsets.UTF_8));
    }
    return HexFormat.of().formatHex(sha.digest());
  }
}
