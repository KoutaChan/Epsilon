package com.epsilon.classic.app;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.epsilon.agent.app.App;
import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.StoreEvent;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.KeyInfo;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.learn.GrpPretrain;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpShape;
import com.epsilon.grp.GrpTensor;
import com.epsilon.grp.GruShape;
import com.epsilon.grp.GruSinkhorn;
import com.epsilon.grp.GruTensor;
import com.epsilon.mahjong.rules.Variant;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * campaign の設定と、campaign・champion のコマンド。
 *
 * <ul>
 *   <li>結び付けと既定: campaign のセクション（run・schedule・learner・selfPlay・evaluation）は何も書かなければ既定になり、キーごとの変えて
 *       よい時を持つ。学習器のデバイスと計算の型の食い違い、相対パスの始めのネットワークと GRP、範囲の外の確率、王者を替えない予定の評価の卓は
 *       拒む。
 *   <li>引数の表: 引数をコマンドのセクションに書け、run の記録を読む campaign は --set を受けずに 2 で止まる。
 *   <li>champion: 王者がまだいなければ「まだいない」と出し、初めの王者を 1 回だけ置ける。ルールの違うチェックポイントは拒む。王者を出すだけの
 *       ときは、王者の保存先に何も書かない（campaign が参照を書き換えている途中でも、その一時ファイルを消さない）。
 *   <li>settings check: 始めのネットワークがチェックポイントでないか、GRP が読めないか、ルールが違うか、自己対局が使えない種類（GRU 型）
 *       なら 4 で止まる。
 * </ul>
 */
public class CampaignCommandsTest {

  private static final Path R = Path.of("/r");

  /** 1 回の呼び出しの終了コード、結果の出力、ログの行。 */
  record Ran(int exit, String out, List<String> logs) {

    boolean logged(String text) {
      return logs.stream().anyMatch(line -> line.contains(text));
    }

    @Override
    public String toString() {
      return "exit " + exit + " " + out + " " + logs;
    }
  }

  @Test
  public void campaignSectionsBindTheirDefaultsAndChanges() {
    ClassicSettings defaults = Settings.bind(ClassicSettings.class, List.of()).value();
    Assert.assertEquals(defaults.run(), new ClassicSettings.Run(Variant.TENHOU_4P, 1, null, null));
    Assert.assertEquals(defaults.schedule(), new ClassicSettings.Schedule(1_350, 32_768));
    Assert.assertEquals(defaults.learner(), new ClassicSettings.Learner(List.of("cpu"), Precision.FLOAT32));
    Assert.assertEquals(defaults.selfPlay(), new ClassicSettings.SelfPlay(0.7, 32));
    Assert.assertEquals(defaults.evaluation().tables(), Map.of());
    Assert.assertEquals(defaults.inference().precision(), Precision.FLOAT32);
    Assert.assertEquals(defaults.table().workers(), 28);

    Map<String, Change> changes = new LinkedHashMap<>();
    for (KeyInfo key : Settings.describe(ClassicSettings.class)) {
      changes.put(key.path(), key.change());
    }
    Map<String, Change> expected = new LinkedHashMap<>();
    expected.put("run.variant", Change.FIXED);
    expected.put("run.seed", Change.FIXED);
    expected.put("run.start", Change.FIXED);
    expected.put("run.grp", Change.FIXED);
    expected.put("schedule.macros", Change.FIT);
    expected.put("schedule.games", Change.COLLECT);
    expected.put("inference.devices", Change.MACHINE);
    expected.put("inference.precision", Change.COLLECT);
    expected.put("table.workers", Change.MACHINE);
    expected.put("learner.devices", Change.MACHINE);
    expected.put("learner.precision", Change.FIT);
    expected.put("selfPlay.championProbability", Change.COLLECT);
    expected.put("selfPlay.pool", Change.COLLECT);
    expected.put("evaluation.alarmZ", Change.SCHEDULE);
    for (Map.Entry<String, Change> entry : expected.entrySet()) {
      Assert.assertEquals(changes.get(entry.getKey()), entry.getValue(), entry.getKey());
    }

    ClassicSettings written =
        bind(
            "[run]\nseed = 5\nstart = \"/nets/start\"\ngrp = \"/nets/grp\"\n[schedule]\nmacros = 20\ngames = 1024\n"
                + "[learner]\ndevices = [\"gpu1\", \"gpu0\"]\nprecision = \"BFLOAT16\"\n"
                + "[selfPlay]\nchampionProbability = 1.0\n" + CHAMPION_TABLE + CONFIRM_TABLE);
    Assert.assertEquals(
        written.run(), new ClassicSettings.Run(Variant.TENHOU_4P, 5, Path.of("/nets/start"), Path.of("/nets/grp")));
    Assert.assertEquals(written.schedule(), new ClassicSettings.Schedule(20, 1024));
    Assert.assertEquals(written.learner().networkDevices().size(), 2);
    Assert.assertEquals(written.selfPlay(), new ClassicSettings.SelfPlay(1.0, 32));
    Assert.assertEquals(written.evaluation().tables().keySet(), Set.of("champion", "outside"));

    Map<String, String> refused = new LinkedHashMap<>();
    refused.put("[learner]\ndevices = [\"gpu0\"]\n", "the learner computes in BFLOAT16");
    refused.put("[learner]\ndevices = [\"cpu\", \"gpu0\"]\n", "Devices must be one cpu or distinct GPUs");
    refused.put("[run]\nstart = \"nets/start\"\n", "run.start");
    refused.put("[run]\ngrp = \"nets/grp\"\n", "run.grp");
    refused.put("[selfPlay]\nchampionProbability = 1.5\n", "selfPlay.championProbability");
    refused.put("[selfPlay]\npool = 0\n", "selfPlay.pool");
    refused.put("[schedule]\nmacros = 0\n", "schedule.macros");
    refused.put(SCHEDULED_TABLE, "the classic campaign schedules only the table that promotes the champion");
    refused.put(
        "[evaluation]\nonWorse = {kind = \"ROLLBACK\", streak = 1, maxRollbacks = 1, span = 10}\n",
        "the classic campaign reports or pauses when the champion duels stay worse");
    for (Map.Entry<String, String> entry : refused.entrySet()) {
      SettingsException thrown = Assert.expectThrows(SettingsException.class, () -> bind(entry.getKey()));
      Assert.assertTrue(
          thrown.problems().stream().anyMatch(problem -> problem.contains(entry.getValue())),
          entry.getValue() + " in " + thrown.problems());
    }
  }

  @Test
  public void campaignAndChampionArguments() throws Exception {
    Assert.assertEquals(section(Campaign.class, "campaign", "run = \"/r\""), new Campaign(R, null));
    Assert.assertEquals(section(Campaign.class, "campaign", "run = \"/r\"\nstopAfter = 3"), new Campaign(R, 3));
    Assert.assertEquals(section(Champion.class, "champion", "run = \"/r\""), new Champion(R, null));
    Assert.assertEquals(
        section(Champion.class, "champion", "run = \"/r\"\ninitial = \"/c\""), new Champion(R, Path.of("/c")));
    Path root = Files.createTempDirectory("classic-campaign-arguments");
    try {
      String run = root.resolve("run").toString();
      String[][] invalid = {
        {"campaign"},
        {"campaign", "--run", run},
        {"campaign", "--run", run, "--stop-after", "0"},
        {"campaign", "--run", run, "--set", "ppo.steps=4"},
        {"campaign", "--run", run, "--macros", "3"},
        {"champion"},
        {"champion", "--run", run},
        {"champion", "--run", run, "--set", "selfPlay.pool=4"},
      };
      for (String[] arguments : invalid) {
        Ran ran = run(arguments);
        Assert.assertEquals(ran.exit(), 2, Arrays.toString(arguments) + " " + ran);
      }
      Assert.assertTrue(
          run("campaign", "--run", run, "--set", "ppo.steps=4").logged("classic settings change --run DIR"));
      Assert.assertTrue(run("campaign", "--run", run).logged("create it with settings create"));
      Assert.assertTrue(run("champion", "--run", run).logged("No settings rows in"));
    } finally {
      delete(root);
    }
  }

  @Test
  public void championPlacesTheFirstChampionOnce() throws Exception {
    Path root = Files.createTempDirectory("classic-champion");
    try {
      Path run = root.resolve("run");
      Ran created = run("settings", "create", "--run", run.toString(), "--set", "run.seed=3");
      Assert.assertEquals(created.exit(), 0, created.toString());
      Ran none = run("champion", "--run", run.toString());
      Assert.assertEquals(none.exit(), 0, none.toString());
      Assert.assertTrue(none.out().contains("No champion yet"), none.toString());

      Path foreign = checkpoint(root.resolve("sanma"), Variant.TENHOU_3P);
      Ran refused = run("champion", "--run", run.toString(), "--initial", foreign.toString());
      Assert.assertEquals(refused.exit(), 2, refused.toString());
      Assert.assertTrue(refused.logged("is not a classic checkpoint for TENHOU_4P"), refused.toString());

      Path nano = checkpoint(root.resolve("nano-stage8"), Variant.TENHOU_4P);
      Ran placed = run("champion", "--run", run.toString(), "--initial", nano.toString());
      Assert.assertEquals(placed.exit(), 0, placed.toString());
      Assert.assertTrue(placed.out().startsWith("champion nano-stage8 version 0 since "), placed.out());
      Assert.assertTrue(placed.out().contains("(initial) " + nano.toAbsolutePath().normalize()), placed.out());
      Assert.assertFalse(placed.out().contains("sha256"), placed.out());
      List<ChampionStore.Entry> history = ChampionStore.open(new RunDirectory(run).champion()).history();
      Assert.assertEquals(history.size(), 1);
      StoreEvent.Initial initial = (StoreEvent.Initial) history.getFirst().event();
      Assert.assertEquals(initial.champion(), ChampionStore.Champion.placed("nano-stage8", nano, 0));

      Ran again = run("champion", "--run", run.toString(), "--initial", nano.toString());
      Assert.assertEquals(again.exit(), 2, again.toString());
      Assert.assertTrue(again.logged("already has the champion nano-stage8"), again.toString());
      Ran shown = run("champion", "--run", run.toString());
      Assert.assertEquals(shown.exit(), 0, shown.toString());
      Assert.assertEquals(shown.out(), placed.out());
    } finally {
      delete(root);
    }
  }

  @Test
  public void showingTheChampionWritesNothingToTheStore() throws Exception {
    Path root = Files.createTempDirectory("classic-champion-show");
    try {
      Path run = root.resolve("run");
      Ran created = run("settings", "create", "--run", run.toString());
      Assert.assertEquals(created.exit(), 0, created.toString());
      Path nano = checkpoint(root.resolve("nano-stage8"), Variant.TENHOU_4P);
      Ran placed = run("champion", "--run", run.toString(), "--initial", nano.toString());
      Assert.assertEquals(placed.exit(), 0, placed.toString());

      // campaign が参照を書き換えている途中の形: 記録に行があり、一時ファイルを書き終えて、rename する前
      Path store = new RunDirectory(run).champion();
      Path reference = store.resolve(ChampionStore.CURRENT);
      Path temporary = store.resolve("." + ChampionStore.CURRENT + ".7.tmp");
      Files.move(reference, temporary);
      Ran shown = run("champion", "--run", run.toString());
      Assert.assertEquals(shown.exit(), 0, shown.toString());
      Assert.assertEquals(shown.out(), placed.out());
      Assert.assertTrue(Files.exists(temporary), "showing the champion removed the temporary file of the campaign");
      Assert.assertFalse(Files.exists(reference), "showing the champion wrote the reference");
    } finally {
      delete(root);
    }
  }

  @Test
  public void settingsCheckFindsAMissingStartNetworkAndGrp() throws Exception {
    Path root = Files.createTempDirectory("classic-check");
    try {
      Path run = root.resolve("run");
      Path missing = root.resolve("missing").toAbsolutePath();
      Path grp = root.resolve("grp").toAbsolutePath();
      Ran created =
          run("settings", "create", "--run", run.toString(), "--set", "run.start=" + missing, "--set", "run.grp=" + grp);
      Assert.assertEquals(created.exit(), 0, created.toString());
      Ran checked = run("settings", "check", "--run", run.toString());
      Assert.assertEquals(checked.exit(), 4, checked.toString());
      Assert.assertTrue(checked.logged("The start network run.start " + missing), checked.toString());
      Assert.assertTrue(checked.logged("The GRP run.grp " + grp + " is not a usable GRP checkpoint"), checked.toString());

      checkpoint(missing, Variant.TENHOU_4P);
      grp(grp, Variant.TENHOU_3P);
      Ran foreign = run("settings", "check", "--run", run.toString());
      Assert.assertEquals(foreign.exit(), 4, foreign.toString());
      Assert.assertTrue(foreign.logged("The GRP run.grp " + grp + " is for TENHOU_3P, not TENHOU_4P"), foreign.toString());
      Assert.assertFalse(foreign.logged("The start network"), foreign.toString());

      delete(grp);
      grp(grp, Variant.TENHOU_4P);
      Ran usable = run("settings", "check", "--run", run.toString());
      Assert.assertEquals(usable.exit(), 0, usable.toString());
    } finally {
      delete(root);
    }
  }

  @Test
  public void settingsCheckRefusesAGrpThatIsNotANetwork() throws Exception {
    Path root = Files.createTempDirectory("classic-check-gru");
    try {
      Path run = root.resolve("run");
      Path grp = root.resolve("gru").toAbsolutePath();
      GruShape shape = new GruShape(Variant.TENHOU_4P, 4, 1);
      GrpCheckpoint.write(GruSinkhorn.of(shape, new float[GruTensor.count(shape)]), grp);
      Ran created = run("settings", "create", "--run", run.toString(), "--set", "run.grp=" + grp);
      Assert.assertEquals(created.exit(), 0, created.toString());
      Ran checked = run("settings", "check", "--run", run.toString());
      Assert.assertEquals(checked.exit(), 4, checked.toString());
      Assert.assertTrue(
          checked.logged("The GRP run.grp " + grp + " must be a network GRP, not GruSinkhorn"), checked.toString());
    } finally {
      delete(root);
    }
  }

  /** 王者を替える決闘の卓（10 macro ごと）。 */
  private static final String CHAMPION_TABLE =
      """
      [evaluation.tables.champion]
      opponent = "champion"
      seating = {kind = "EVERY_PLACEMENT", splits = ["1v3", "3v1"]}
      draws = "SEAT_STREAMS"
      wallSet = "FRESH"
      walls = 12288
      stop = {kind = "TEST", evenMargin = 0.004, alpha = 0.01, spending = 3.0, checkEvery = 1536, promote = true}
      score = "UTILITY"
      every = 10
      from = 10
      """;

  /** 予定では打たない卓。 */
  private static final String CONFIRM_TABLE =
      """
      [evaluation.tables.outside]
      opponent = "heuristic"
      seating = {kind = "EVERY_PLACEMENT", splits = ["1v3"]}
      draws = "SEAT_STREAMS"
      wallSet = "FIXED"
      walls = 8192
      stop = "FIXED"
      score = "UTILITY"
      every = 0
      from = 0
      """;

  /** 王者を替えずに予定で打つ卓。 */
  private static final String SCHEDULED_TABLE =
      """
      [evaluation.tables.rx3]
      opponent = "heuristic"
      seating = {kind = "EVERY_PLACEMENT", splits = ["1v3"]}
      draws = "SEAT_STREAMS"
      wallSet = "FIXED"
      walls = 4096
      stop = "FIXED"
      score = "RANK"
      every = 1
      from = 1
      """;

  /** ヘッダーだけを照合できる classic のチェックポイント（重みは中身の無いファイル）。 */
  private static Path checkpoint(Path directory, Variant variant) throws Exception {
    Files.createDirectories(directory);
    Files.writeString(
        directory.resolve("model.json"),
        "{\"architecture\": \"NANO\", \"variant\": \"" + variant + "\", \"utility\": \"TENHOU\", \"inputFingerprint\": \""
            + ClassicSchema.of(Architecture.NANO).fingerprint() + "\", \"step\": 0}",
        StandardCharsets.UTF_8);
    Files.write(directory.resolve("weights.bin"), new byte[] {1, 2, 3});
    return directory.toAbsolutePath();
  }

  /** 重みが全て 0 の小さな GRP のチェックポイント（pretrain-grp と同じフィールド）。 */
  private static void grp(Path directory, Variant variant) throws Exception {
    GrpShape shape = new GrpShape(variant, GrpPretrain.FIELDS, 4, false);
    float[] weights = new float[GrpTensor.starts(shape)[GrpTensor.values().length]];
    GrpCheckpoint.write(GrpNetwork.of(shape, weights), directory);
  }

  private static ClassicSettings bind(String toml) {
    return Settings.bind(ClassicSettings.class, List.of(Layer.parse("campaign.toml", Layer.Source.FILE, toml))).value();
  }

  private static <T extends Record> T section(Class<T> type, String path, String body) {
    String text = "[" + path + "]\n" + body + "\n";
    return Settings.section(type, path, List.of(Layer.parse("classic.toml", Layer.Source.FILE, text))).value();
  }

  private static Ran run(String... args) {
    Logger root = (Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
    ListAppender<ILoggingEvent> appender = new ListAppender<>();
    appender.start();
    root.addAppender(appender);
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    int exit;
    try (PrintStream out = new PrintStream(bytes, true, StandardCharsets.UTF_8)) {
      exit = App.run(Main.application(), args, out);
    } finally {
      root.detachAppender(appender);
    }
    List<String> logs = appender.list.stream().map(ILoggingEvent::getFormattedMessage).toList();
    return new Ran(exit, bytes.toString(StandardCharsets.UTF_8), logs);
  }

  private static void delete(Path root) throws Exception {
    try (Stream<Path> paths = Files.walk(root)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
        Files.deleteIfExists(path);
      }
    }
  }
}
