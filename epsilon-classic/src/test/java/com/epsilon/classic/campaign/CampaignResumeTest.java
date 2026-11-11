package com.epsilon.classic.campaign;

import com.epsilon.agent.checkpoint.CheckpointFiles;
import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.run.ControlRequests;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.Ending;
import com.epsilon.agent.run.Request;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.RunStatus;
import com.epsilon.agent.run.Runner;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Settings;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.RandomCheckpoints;
import com.epsilon.classic.app.ClassicSettings;
import com.epsilon.mahjong.rules.Variant;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.BooleanSupplier;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * campaign の止め方と続き（CPU）。同じ設定の run を 2 つ作り、片方は止めずに最後まで回し、もう片方は段の前と後・反復を閉じた後・起動の
 * セーフポイントで 1 回ずつ止めては起動し直す。最初の起動は停止の要求で止め、ほかは SIGTERM と同じ停止のフラグで止める。
 *
 * <p>止めた所には、落ちた後の残りを置く: 段の前なら書きかけの出力、学習の段の前なら書き終えて確定する前の収集の出力（確定を戻す）、学習の段の
 * 後なら公開する前の重み（公開を戻す）と書きかけの反復の報告の行、決闘の段の前なら書き終えて確定する前の学習の出力。どこで止めても、反復の
 * 報告・state・公開したバージョンの重み・残したコホートと学習器の途中の状態の中身・王者の保存先・自己対局が受け取った計画は、止めない run と
 * 同じ（時刻・秒・リソースと、段を回したか使い回したかを除く。学習器の途中の状態は、AdamW の状態のファイルに入る書いた時刻を除く）。
 *
 * <p>止めない run は macro 2 の学習を、同じ JVM が macro 1 で学んだ学習器のまま続ける。止める run は、学習器をいつも learner の途中の状態から
 * 開き直すので、2 つが同じことは学習器の途中の状態の往復も確かめる。
 */
public class CampaignResumeTest {

  private static final String JAR = "test0001";

  private static final Variant VARIANT = Variant.TENHOU_4P;

  private static final int MACROS = 2;

  private static final int MAX_LAUNCHES = 40;

  /** 学習器の途中の状態のうち、AdamW の状態のファイルの名前の末尾。 */
  private static final String OPTIMIZER_STATE = ".optimizer";

  /** 段の後で止めた次の起動で、一時停止の要求で止める次の段。 */
  private static final Map<String, String> NEXT_STEP =
      Map.of("AFTER_STEP COLLECT", "LEARN", "AFTER_STEP LEARN", "DUEL");

  @Test(groups = "native", timeOut = 1_800_000)
  public void stoppingAndCrashingAtEveryDonePointGivesTheSameCampaign() throws Exception {
    Path root = Files.createTempDirectory("classic-campaign-resume");
    try {
      Path start = root.resolve("start");
      RandomCheckpoints.write(start, Architecture.NANO, VARIANT, 7);
      Path reference = root.resolve("reference");
      Path stopped = root.resolve("stopped");
      create(reference, start);
      create(stopped, start);

      List<Collector.Macro> referencePlans = new ArrayList<>();
      Assert.assertEquals(runner(reference, referencePlans, () -> false).train(), Ending.FINISHED);
      Outcome expected = outcome(reference);
      Assert.assertEquals(expected.iterations().size(), MACROS);
      Assert.assertEquals(expected.snapshots().keySet(), Set.of("00001", "00002", "00003"));
      Assert.assertEquals(expected.learners().keySet(), Set.of("00003"));
      Assert.assertEquals(expected.cohorts().keySet(), Set.of("00002"));
      Assert.assertEquals(expected.champion().size(), 1);
      for (String line : expected.iterations()) {
        JsonObject steps = JsonParser.parseString(line).getAsJsonObject().getAsJsonObject("steps");
        Assert.assertEquals(steps.getAsJsonObject("DUEL").get("status").getAsString(), "SKIPPED", line);
        Assert.assertEquals(steps.getAsJsonObject("DECIDE").get("status").getAsString(), "SKIPPED", line);
        JsonObject learn = steps.getAsJsonObject("LEARN");
        Assert.assertTrue(learn.get("actorSteps").getAsInt() > 0, line);
        Assert.assertTrue(learn.has("valueLoss"), line);
      }
      List<String> plans = plans(reference, referencePlans);
      Assert.assertEquals(plans.size(), MACROS);

      List<Collector.Macro> stoppedPlans = new ArrayList<>();
      List<String> points = stopEverywhere(stopped, stoppedPlans);
      for (String point :
          List.of(
              "BEFORE_STEP COLLECT", "AFTER_STEP COLLECT", "BEFORE_STEP LEARN", "AFTER_STEP LEARN", "BEFORE_STEP DUEL",
              "ITERATION")) {
        Assert.assertEquals(Collections.frequency(points, point), MACROS, point + " in " + points);
      }
      Assert.assertEquals(Collections.frequency(points, "LAUNCH"), 2, points.toString());
      Assert.assertEquals(RunStatus.read(new RunDirectory(stopped).status()).state(), RunStatus.State.FINISHED);
      Assert.assertEquals(outcome(stopped), expected);
      // 確定を戻した収集はもう一度打つので、macro ごとに同じ計画が 2 回届く
      Assert.assertEquals(
          plans(stopped, stoppedPlans), List.of(plans.get(0), plans.get(0), plans.get(1), plans.get(1)));
      assertNoLeftovers(new RunDirectory(stopped));
    } finally {
      delete(root);
    }
  }

  /** 小さな campaign の設定で、設定の 1 行目を書いた run を作る。 */
  private static void create(Path root, Path start) throws IOException {
    List<String> sets =
        List.of(
            "run.seed=11",
            "run.start=" + start.toAbsolutePath(),
            "schedule.macros=" + MACROS,
            "schedule.games=3",
            "ppo.steps=2",
            "batches.rows=64",
            "table.workers=1",
            "table.activeGames=8",
            "inference.encoders=1",
            "inference.slots=1",
            "inference.readyBatches=1");
    Settings<ClassicSettings> settings =
        Settings.bind(ClassicSettings.class, List.of(Layer.assignments("--set", sets)));
    try (SettingsRows.Locked locked = new SettingsRows(new RunDirectory(root)).lock()) {
      locked.append(1, JAR, SettingsRow.CREATED, List.of(), settings.tree());
    }
  }

  private static Runner<ClassicSettings, CampaignState, CampaignIteration, CampaignStep> runner(
      Path run, List<Collector.Macro> plans, BooleanSupplier signalled) {
    return new Runner<>(
        new RunDirectory(run),
        new CampaignKind(RandomCollector.factory(plans)),
        new Runner.Options(JAR, null, null, signalled));
  }

  /**
   * 止めては続けて、run を最後まで回す。最初の起動は停止の要求で起動のセーフポイントに止める。ほかの起動は、ファイルが前に止めた所から進んだ
   * 最初のセーフポイントで止め、段の後で止めた次の起動では、一時停止の要求で待たせたその次の段の前で止める。
   *
   * @return 止まったセーフポイント（起きた順）
   */
  private static List<String> stopEverywhere(Path root, List<Collector.Macro> plans) throws Exception {
    RunDirectory run = new RunDirectory(root);
    ControlRequests requests = new ControlRequests(run);
    requests.submit(new Request.Stop(Request.At.STEP));
    Assert.assertEquals(runner(root, plans, () -> false).train(), Ending.STOPPED);
    Assert.assertEquals(RunStatus.read(run.status()).reason(), "Stopped at LAUNCH by request 1");
    Assert.assertEquals(requests.pending(), List.of(), "the stop request is still pending");
    List<String> points = new ArrayList<>(List.of("LAUNCH"));
    Set<String> visited = new HashSet<>();
    String before = null;
    for (int launch = 2; launch <= MAX_LAUNCHES; launch++) {
      Ending ending;
      if (before != null) {
        requests.submit(new Request.Pause(before));
        ending = runner(root, plans, () -> paused(run)).train();
        requests.submit(new Request.Resume());
      } else {
        ending = runner(root, plans, () -> visited.add(progress(run))).train();
      }
      RunStatus status = RunStatus.read(run.status());
      if (ending == Ending.FINISHED) {
        return points;
      }
      Assert.assertEquals(ending, Ending.STOPPED, points + " " + status.reason());
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
    throw new AssertionError("The campaign did not finish in " + MAX_LAUNCHES + " launches: " + points);
  }

  /** 止めた所に、落ちた後の残りを置く。 */
  private static void leaveTraces(RunDirectory run, String point, int t) throws IOException {
    switch (point) {
      case "BEFORE_STEP COLLECT" -> junk(CampaignStep.part(run.rows(t)));
      case "BEFORE_STEP LEARN" -> {
        // 収集の出力を書き終えて、確定する前に落ちた
        Files.move(run.rows(t), CampaignStep.part(run.rows(t)));
        junk(CampaignStep.part(LearnStep.learner(run, t + 1)));
      }
      case "AFTER_STEP LEARN" -> {
        // 学習の完了マーカーの後、重みを公開する前に落ちた。前に反復の報告の行を書きかけで落ちた
        unpublish(run, t + 1);
        Files.createDirectories(run.reports());
        Files.writeString(
            run.report(RunDirectory.ITERATIONS),
            "{\"iteration\": 99, \"trunc",
            StandardCharsets.UTF_8,
            StandardOpenOption.CREATE,
            StandardOpenOption.APPEND);
      }
      case "BEFORE_STEP DUEL" -> {
        // 学習の出力を書き終えて、確定する前に落ちた
        unpublish(run, t + 1);
        Path learner = LearnStep.learner(run, t + 1);
        Files.move(learner, CampaignStep.part(learner));
      }
      default -> {}
    }
  }

  /** 公開した重みを、学習の段の出力に戻す。 */
  private static void unpublish(RunDirectory run, int version) throws IOException {
    Files.move(run.snapshot(version), LearnStep.learner(run, version).resolve(LearnStep.NETWORK));
  }

  /** 書きかけのディレクトリの残り。 */
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
   * run の進み: state、バージョン・コホート・学習器の途中の状態・反復の作業のディレクトリの名前、反復の報告と王者の記録の行の数（書きかけの
   * .part は数えない）。
   */
  private static String progress(RunDirectory run) {
    try {
      StringBuilder out = new StringBuilder();
      out.append(Files.exists(run.state()) ? Files.readString(run.state(), StandardCharsets.UTF_8) : "-").append('|');
      out.append(names(run.snapshots())).append('|').append(names(run.rows(1).getParent())).append('|');
      out.append(names(LearnStep.learner(run, 1).getParent())).append('|');
      for (String t : names(run.work())) {
        out.append(t).append(names(run.work().resolve(t))).append('|');
      }
      out.append(DurableFiles.lines(run.report(RunDirectory.ITERATIONS)).size()).append('|');
      out.append(DurableFiles.lines(run.champion().resolve(ChampionStore.HISTORY)).size());
      return out.toString();
    } catch (IOException unreadable) {
      throw new UncheckedIOException(unreadable);
    }
  }

  /** ディレクトリの中の名前（.part を除く、名前の順）。ディレクトリが無ければ空。 */
  private static List<String> names(Path directory) throws IOException {
    if (!Files.isDirectory(directory)) {
      return List.of();
    }
    try (Stream<Path> listed = Files.list(directory)) {
      return listed
          .map(path -> path.getFileName().toString())
          .filter(name -> !name.endsWith(CampaignStep.PART))
          .sorted()
          .toList();
    }
  }

  /** 閉じた反復の作業のディレクトリと、段の書きかけが残っていない。 */
  private static void assertNoLeftovers(RunDirectory run) throws IOException {
    Assert.assertEquals(names(run.work()), List.of());
    for (Path directory : List.of(run.snapshots(), run.rows(1).getParent(), LearnStep.learner(run, 1).getParent())) {
      try (Stream<Path> listed = Files.list(directory)) {
        Assert.assertEquals(
            listed.filter(path -> path.getFileName().toString().endsWith(CampaignStep.PART)).toList(), List.of());
      }
    }
  }

  /**
   * 比べる run の結果。時刻・秒・リソースと、段を回したか使い回したかを除き、run のディレクトリの場所は {@code <run>} に置き換える。
   *
   * @param snapshots 公開したバージョンの SHA-256
   * @param learners 残した学習器の途中の状態の中身の SHA-256（{@link #learnerState}）
   * @param cohorts 残したコホートの軌跡と迷いの分布の SHA-256
   */
  private record Outcome(
      String state,
      List<String> iterations,
      Map<String, String> snapshots,
      Map<String, String> learners,
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
        if (!metrics.get("status").getAsString().equals("SKIPPED")) {
          metrics.addProperty("status", "DONE");
        }
      }
      iterations.add(report.toString());
    }
    Map<String, String> snapshots = new TreeMap<>();
    for (String version : names(run.snapshots())) {
      snapshots.put(version, CheckpointFiles.sha256(run.snapshots().resolve(version)));
    }
    Map<String, String> learners = new TreeMap<>();
    for (String version : names(LearnStep.learner(run, 1).getParent())) {
      Path state = LearnStep.learner(run, Integer.parseInt(version)).resolve(LearnStep.OPTIMIZER);
      learners.put(version, learnerState(state));
    }
    Map<String, String> cohorts = new TreeMap<>();
    for (String cohort : names(run.rows(1).getParent())) {
      Path directory = run.rows(Integer.parseInt(cohort));
      cohorts.put(
          cohort, digest(directory.resolve(Cohort.TRAJECTORIES)) + " " + digest(directory.resolve(Cohort.CALIBRATION)));
    }
    List<String> champion = new ArrayList<>();
    for (ChampionStore.Entry entry : ChampionStore.open(run.champion()).history()) {
      champion.add(entry.event().toString().replace(place, "<run>"));
    }
    return new Outcome(
        Files.readString(run.state(), StandardCharsets.UTF_8), iterations, snapshots, learners, cohorts, champion);
  }

  /** 自己対局が受け取った計画を、run の場所に依らない文にする（macro・種・学習器のバージョン・王者・過去の王者の数・半荘の数）。 */
  private static List<String> plans(Path root, List<Collector.Macro> received) {
    String place = new RunDirectory(root).root().toString();
    List<String> plans = new ArrayList<>();
    for (Collector.Macro macro : received) {
      plans.add(
          macro.number() + " " + macro.seed() + " " + macro.learner().toString().replace(place, "<run>") + " "
              + macro.champion().label() + " " + macro.pool().size() + " " + macro.settings().schedule().games());
    }
    return plans;
  }

  private static String digest(Path file) throws Exception {
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file)));
  }

  /** 学習器の途中の状態の中身の SHA-256。ファイルを名前の順に数え、AdamW の状態のファイルは {@link #optimizerContent} で数える。 */
  private static String learnerState(Path directory) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (Stream<Path> listed = Files.list(directory)) {
      for (Path file : listed.sorted().toList()) {
        String name = file.getFileName().toString();
        digest.update(name.getBytes(StandardCharsets.UTF_8));
        digest.update(name.endsWith(OPTIMIZER_STATE) ? optimizerContent(file) : Files.readAllBytes(file));
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  /**
   * AdamW の状態のファイル（DJL の {@code Optimizer#saveState}）の中身。配列は zip（NPZ）の項目に入り、項目には書いた時刻が入るので、
   * 時刻を除いて、ヘッダーの値（形式・クラス・更新の数）と、項目の名前と中身を名前の順に並べる。
   */
  private static byte[] optimizerContent(Path file) throws IOException {
    ByteArrayOutputStream content = new ByteArrayOutputStream();
    try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file)));
        DataOutputStream out = new DataOutputStream(content)) {
      out.writeUTF(in.readUTF());
      out.writeInt(in.readInt());
      out.writeUTF(in.readUTF());
      out.writeInt(in.readInt());
      out.writeInt(in.readInt());
      int counts = in.readInt();
      out.writeInt(counts);
      Map<String, Integer> updates = new TreeMap<>();
      for (int index = 0; index < counts; index++) {
        updates.put(in.readUTF(), in.readInt());
      }
      for (Map.Entry<String, Integer> update : updates.entrySet()) {
        out.writeUTF(update.getKey());
        out.writeInt(update.getValue());
      }
      byte[] npz = new byte[in.readInt()];
      in.readFully(npz);
      Assert.assertEquals(in.read(), -1, "trailing bytes after the arrays of " + file);
      Map<String, byte[]> arrays = new TreeMap<>();
      try (ZipInputStream zip = new ZipInputStream(new ByteArrayInputStream(npz))) {
        for (ZipEntry entry = zip.getNextEntry(); entry != null; entry = zip.getNextEntry()) {
          arrays.put(entry.getName(), zip.readAllBytes());
        }
      }
      Assert.assertFalse(arrays.isEmpty(), "no arrays in " + file);
      for (Map.Entry<String, byte[]> array : arrays.entrySet()) {
        out.writeUTF(array.getKey());
        out.writeInt(array.getValue().length);
        out.write(array.getValue());
      }
    }
    return content.toByteArray();
  }

  private static void delete(Path root) throws IOException {
    try (Stream<Path> paths = Files.walk(root)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
        Files.deleteIfExists(path);
      }
    }
  }
}
