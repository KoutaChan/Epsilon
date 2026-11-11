package com.epsilon.classic.app;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.epsilon.agent.app.App;
import com.epsilon.agent.learn.GrpTraining;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.KeyInfo;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.ClassicHeader;
import com.epsilon.classic.data.ClassicRows;
import com.epsilon.classic.data.LogRows;
import com.epsilon.classic.data.RowChunks;
import com.epsilon.classic.data.RuleGames;
import com.epsilon.classic.learn.Exploration;
import com.epsilon.classic.learn.KlLearningRate;
import com.epsilon.classic.learn.SelfPlayLearner;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.train.GrpSlice;
import com.epsilon.grp.train.LogGames;
import com.epsilon.grp.train.TrainingEnding;
import com.epsilon.mahjong.rules.Variant;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
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
 * classic のコマンドを入口（{@link App#run}）から回す。
 *
 * <ul>
 *   <li>設定の結び付けと既定値: 必須のキーだけを書くと、4 つのコマンドと GRP の学び方が既定値になる。
 *   <li>通しの流れ（{@code native}）: ルールベースの打ち手の牌譜から GRP を学ぶと、早期終了用の検証の行の NLL が順列の頻度の NLL 以下に
 *       なる。行を作って Nano を 1 エポック学ぶと、検証の値が出て、書いた checkpoint を Nano の系列で開ける。
 *   <li>使い方の誤り（終了コード 2）で止まるもの: 始めの重みとゼロから学ぶ形を両方書く・どちらも書かない、途中の状態を書かずに
 *       時間の上限（分）か区切りのファイルを書く、デバイスの書き方の誤り、範囲の外の検証の割合、既にある書き込み先、ディレクトリでない
 *       牌譜、学習を止める回数が学習率を半分にする回数より少ない学び方。
 *   <li>区切り: {@code pretrain} は区切りのファイルがあると終了コード 3 で止まり、同じ引数で呼ぶと続きから学ぶ。{@code pretrain-grp} は
 *       停止の要求を受けると何も書かない。
 * </ul>
 */
public class ClassicCommandsTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final Path R = Path.of("/r");
  private static final Path O = Path.of("/o");

  /** 1 回の呼び出しの終了コード、結果の出力、ログの行。 */
  record Ran(int exit, String out, List<String> logs) {

    boolean logged(String text) {
      return logs.stream().anyMatch(line -> line.contains(text));
    }

    /** 終わりを知らせるログの行から読んだ、要約の {@code 名前=値}。 */
    Map<String, String> summary() {
      String finished =
          logs.stream()
              .filter(line -> line.startsWith("Command finished: "))
              .reduce((first, second) -> second)
              .orElseThrow(() -> new AssertionError("no finishing line in " + logs));
      Map<String, String> values = new LinkedHashMap<>();
      for (String token : finished.substring(finished.indexOf(" s: ") + 4).split(" ")) {
        int equals = token.indexOf('=');
        if (equals > 0) {
          values.putIfAbsent(token.substring(0, equals), token.substring(equals + 1));
        }
      }
      return values;
    }

    double number(String name) {
      return Double.parseDouble(summary().get(name));
    }

    @Override
    public String toString() {
      return "exit " + exit + " " + logs;
    }
  }

  @Test
  public void requiredKeysAloneBindTheDesignDefaults() {
    Assert.assertEquals(
        section(Init.class, "init", "architecture = \"MAJOR\"\nout = \"/o\""),
        new Init(Architecture.MAJOR, O, VARIANT, 1));
    Assert.assertEquals(
        section(Rows.class, "rows", "records = \"/r\"\nout = \"/o\""), new Rows(R, O, VARIANT, 0.1, 2048));
    Assert.assertEquals(
        section(Pretrain.class, "pretrain", "rows = \"/r\"\nout = \"/o\"\narchitecture = \"NANO\""),
        new Pretrain(
            R, O, null, Architecture.NANO, null, 4, 1024, 2e-4, 0.01, 0.5, 2.0, 10.0, "cpu", Precision.FLOAT32, null, 1,
            null, null, null));
    Assert.assertEquals(
        section(PretrainGrp.class, "pretrain-grp", "records = \"/r\"\nout = \"/o\""),
        new PretrainGrp(R, O, VARIANT, 1, 1));
    Assert.assertEquals(
        Settings.bind(ClassicSettings.class, List.of()).value().grp().training(),
        new GrpTraining(4096, 10.0, 2, 4, 1e-3, 16, 3e-4, 1));
    GrpTraining written =
        Settings.bind(
                ClassicSettings.class,
                List.of(
                    Layer.parse("file", Layer.Source.FILE, "[grp.training]\nwarmEpochs = 3\n"),
                    Layer.assignments("--set", List.of("grp.training.scratchEpochs=2"))))
            .value()
            .grp()
            .training();
    Assert.assertEquals(written, new GrpTraining(4096, 10.0, 2, 4, 1e-3, 2, 3e-4, 3));
    // 学び方のキーを変えてよい時期は、属するセクションと同じ（classic の grp なら、次の反復の学習から効く）
    Assert.assertEquals(
        Settings.describe(ClassicSettings.class).stream()
            .filter(key -> key.path().startsWith("grp.training."))
            .map(KeyInfo::change)
            .distinct()
            .toList(),
        List.of(Change.FIT));
    Assert.assertThrows(
        SettingsException.class,
        () -> Settings.bind(
            ClassicSettings.class, List.of(Layer.assignments("--set", List.of("grp.training.stopPatience=1")))));
  }

  /** 自己対局のセクションの既定値は Nano の自己対局の値。キーを変えてよい時期は、属するセクションと同じ。 */
  @Test
  public void selfPlaySectionsBindTheNanoValues() {
    ClassicSettings defaults = Settings.bind(ClassicSettings.class, List.of()).value();
    Assert.assertEquals(defaults.explore(), new Exploration.Settings(0.004, 0.010, 1e-4, 0.1, 5.0, 4.5, 0.5));
    Assert.assertEquals(
        defaults.learning(),
        new SelfPlayLearner.Settings(
            new SelfPlayLearner.Ppo(32, 0.2, 0.10, 0.001, 0.01, 1e-4, 0.5),
            new KlLearningRate.Settings(1.25e-6, 2.5e-7, 5e-6, 3, 3, 1.25, 1.25, 1.02, 43_200, 0.5),
            new SelfPlayLearner.ValueRate(1.25e-6, 1e-4, 0.5),
            new SelfPlayLearner.VTrace(0.95),
            new SelfPlayLearner.Batches(1_024, 16_384)));
    Set<String> learning = Set.of("grp", "explore", "ppo", "actorRate", "valueRate", "vtrace", "batches");
    Map<String, Change> changes = new LinkedHashMap<>();
    for (KeyInfo key : Settings.describe(ClassicSettings.class)) {
      String section = key.path().split("\\.")[0];
      if (!learning.contains(section)) {
        continue;
      }
      Change previous = changes.putIfAbsent(section, key.change());
      Assert.assertTrue(previous == null || previous == key.change(), key.path());
    }
    Assert.assertEquals(
        changes,
        Map.of(
            "grp", Change.FIT, "explore", Change.COLLECT, "ppo", Change.FIT, "actorRate", Change.FIXED,
            "valueRate", Change.FIT, "vtrace", Change.FIT, "batches", Change.MACHINE));
    Assert.assertThrows(
        SettingsException.class,
        () -> Settings.bind(ClassicSettings.class, List.of(Layer.assignments("--set", List.of("ppo.clip=1.0")))));
  }

  @Test
  public void helpListsTheCommandsAndTheirArguments() {
    Ran help = run("help");
    Assert.assertEquals(help.exit(), 0, help.toString());
    List<String> commands = new ArrayList<>();
    boolean listing = false;
    for (String line : help.out().split("\n")) {
      if (line.equals("Commands:")) {
        listing = true;
      } else if (listing && line.isBlank()) {
        break;
      } else if (listing) {
        commands.add(line.trim().split(" ")[0]);
      }
    }
    // 組み込みのコマンドが先に並び、classic のコマンドは宣言の順で最後に並ぶ
    Assert.assertEquals(
        commands.subList(commands.size() - 6, commands.size()),
        List.of("init", "rows", "pretrain", "pretrain-grp", "campaign", "champion"),
        help.out());
    Ran rows = run("help", "rows");
    Assert.assertEquals(rows.exit(), 0, rows.toString());
    for (String expected :
        List.of("--records", "--out", "--variant", "--validation", "--chunk-games", "default 2048", "[rows]")) {
      Assert.assertTrue(rows.out().contains(expected), expected + " in " + rows.out());
    }
    Ran pretrain = run("help", "pretrain");
    for (String expected : List.of("--from", "--architecture", "--state", "--minutes", "--pause-file", "--precision")) {
      Assert.assertTrue(pretrain.out().contains(expected), expected + " in " + pretrain.out());
    }
    Ran campaign = run("help", "campaign");
    for (String expected : List.of("--run", "--stop-after", "classic settings change --run DIR")) {
      Assert.assertTrue(campaign.out().contains(expected), expected + " in " + campaign.out());
    }
    Ran champion = run("help", "champion");
    for (String expected : List.of("--run", "--initial", "[champion]")) {
      Assert.assertTrue(champion.out().contains(expected), expected + " in " + champion.out());
    }
    Ran unknown = run("no-such");
    Assert.assertEquals(unknown.exit(), 2, unknown.toString());
  }

  @Test
  public void invalidArgumentsExitWithUsageBeforeAnyWork() throws Exception {
    Path root = Files.createTempDirectory("classic-invalid");
    try {
      Path existing = Files.createDirectories(root.resolve("existing"));
      Path file = Files.writeString(root.resolve("file"), "");
      String out = root.resolve("out").toString();
      String[][] invalid = {
        {"pretrain", "--rows", existing.toString(), "--out", out},
        {
          "pretrain", "--rows", existing.toString(), "--out", out, "--architecture", "NANO", "--from",
          existing.toString()
        },
        {"pretrain", "--rows", existing.toString(), "--out", out, "--architecture", "NANO", "--minutes", "1"},
        {"pretrain", "--rows", existing.toString(), "--out", out, "--architecture", "NANO", "--pause-file", "p"},
        {"pretrain", "--rows", existing.toString(), "--out", out, "--architecture", "NANO", "--device", "gpu-1"},
        {"pretrain", "--rows", existing.toString(), "--out", existing.toString(), "--architecture", "NANO"},
        {"rows", "--records", existing.toString(), "--out", out, "--validation", "1.0"},
        {"rows", "--records", existing.toString(), "--out", out, "--validation", "0"},
        {"rows", "--records", file.toString(), "--out", out},
        {"rows", "--records", existing.toString(), "--out", existing.toString()},
        {"init", "--architecture", "NANO", "--out", existing.toString()},
        {"init", "--architecture", "TINY", "--out", out},
        {"pretrain-grp", "--records", existing.toString(), "--out", existing.toString()},
        {"pretrain-grp", "--records", file.toString(), "--out", out},
        {"pretrain-grp", "--records", existing.toString(), "--out", out, "--set", "grp.training.stopPatience=1"},
        {"pretrain-grp", "--records", existing.toString(), "--out", out, "--set", "grp.training.batchRow=8"},
      };
      for (String[] arguments : invalid) {
        Ran ran = run(arguments);
        Assert.assertEquals(ran.exit(), 2, Arrays.toString(arguments) + " " + ran);
      }
      Assert.assertFalse(Files.exists(root.resolve("out")));
      Assert.assertTrue(
          run("pretrain", "--rows", existing.toString(), "--out", out, "--architecture", "NANO", "--device", "gpu-1")
              .logged("expected cpu or gpuN"));
      Assert.assertTrue(
          run("rows", "--records", existing.toString(), "--out", existing.toString())
              .logged("The rows directory already exists: " + existing));
      Assert.assertTrue(
          run("rows", "--records", file.toString(), "--out", out)
              .logged("Argument --records names no directory: " + file));
      Assert.assertTrue(
          run("pretrain-grp", "--records", existing.toString(), "--out", out, "--set", "grp.training.stopPatience=1")
              .logged("stopPatience 1 is below halvingPatience 2"));
      Assert.assertTrue(
          run("pretrain-grp", "--records", existing.toString(), "--out", out, "--set", "grp.training.batchRow=8")
              .logged("did you mean grp.training.batchRows?"));
      Path settings = Files.writeString(root.resolve("classic.toml"), "[pretrain]\ndevice = \"gpu-1\"\n");
      Ran section =
          run("pretrain", "--rows", existing.toString(), "--out", out, "--architecture", "NANO", "--settings",
              settings.toString());
      Assert.assertEquals(section.exit(), 2, section.toString());
      Assert.assertTrue(section.logged("expected cpu or gpuN"), section.toString());
    } finally {
      delete(root);
    }
  }

  @Test(groups = "native", timeOut = 1_200_000)
  public void recordsTrainTheGrpAndANano() throws Exception {
    Path root = Files.createTempDirectory("classic-commands");
    try {
      Path all = root.resolve("all");
      Path few = root.resolve("few");
      int stopGames = 0;
      int trainGames = 0;
      boolean validation = false;
      for (int index = 0; stopGames < 2 || !validation || index < 32; index++) {
        String id = "rule-" + index;
        var game = RuleGames.play(VARIANT, 1_000 + index, id);
        RuleGames.write(all, game);
        stopGames += GrpSlice.of(LogGames.gameId(id)) == GrpSlice.STOP ? 1 : 0;
        boolean held = LogRows.validation(id, 0.1);
        if ((held && !validation) || (!held && trainGames < 4)) {
          RuleGames.write(few, game);
          validation |= held;
          trainGames += held ? 0 : 1;
        }
      }
      Assert.assertTrue(validation, "no validation game among the generated records");

      Path grp = root.resolve("grp");
      Ran grpRun =
          run("pretrain-grp", "--records", all.toString(), "--out", grp.toString(), "--set",
              "grp.training.scratchEpochs=8");
      Assert.assertEquals(grpRun.exit(), 0, grpRun.toString());
      Assert.assertTrue(grpRun.number("stop") > 0, grpRun.toString());
      Assert.assertTrue(grpRun.number("half-epochs") <= 16, grpRun.toString());
      Assert.assertTrue(grpRun.number("trained") <= grpRun.number("frequency") + 1e-9, grpRun.toString());
      Assert.assertTrue(GrpCheckpoint.read(grp) instanceof GrpNetwork);
      Assert.assertFalse(grpRun.summary().containsKey("sha256"), grpRun.toString());
      Ran again = run("pretrain-grp", "--records", all.toString(), "--out", grp.toString());
      Assert.assertEquals(again.exit(), 2, again.toString());

      Path rows = root.resolve("rows");
      Ran rowsRun = run("rows", "--records", few.toString(), "--out", rows.toString());
      Assert.assertEquals(rowsRun.exit(), 0, rowsRun.toString());
      Assert.assertEquals(rowsRun.summary().get("skipped"), "0");
      Assert.assertEquals(rowsRun.summary().get("games"), String.valueOf(trainGames + 1));
      Assert.assertEquals(rowsRun.summary().get("validationGames"), "1");

      Path nano = root.resolve("nano");
      Ran trained =
          run(
              "pretrain",
              "--rows", rows.toString(),
              "--out", nano.toString(),
              "--architecture", "NANO",
              "--grp", grp.toString(),
              "--epochs", "1",
              "--batch", "64");
      Assert.assertEquals(trained.exit(), 0, trained.toString());
      Map<String, String> summary = trained.summary();
      Assert.assertEquals(summary.get("finished"), "true");
      Assert.assertEquals(summary.get("epochs"), "1");
      Assert.assertTrue(
          Double.isFinite(trained.number("nll")) && Double.isFinite(trained.number("value")), summary.toString());
      Assert.assertTrue(trained.number("agreement") >= 0 && trained.number("agreement") <= 1, summary.toString());
      // ルールベースの打ち手の牌譜なので、ルールベースの打ち手をまねた手は全て一致する
      Assert.assertEquals(trained.number("rulebot"), 1.0);
      Assert.assertTrue(trained.number("uniform") < 0.5, summary.toString());
      ClassicHeader header = ClassicCheckpoint.header(nano);
      Assert.assertEquals(header.architecture(), Architecture.NANO);
      Assert.assertEquals(String.valueOf(header.step()), summary.get("steps"));
      Assert.assertTrue(Files.isRegularFile(root.resolve("nano" + Pretrain.REPORT_SUFFIX)));
      try (Policy policy =
          ModelFamily.of(Series.NANO).open(nano, VARIANT, PolicyOptions.defaults())) {
        Assert.assertEquals(policy.label(), "nano:nano");
      }
    } finally {
      delete(root);
    }
  }

  /**
   * rows は牌譜を名前の順に --chunk-games 個ずつ区切ってファイルに書く（5 半荘を 2 ずつなら 3 ファイル）。区切っても、区切らずに書いた
   * ときと同じ行を同じ順に持つ。設定ファイルの [rows] セクションにも同じ区切りを書ける。--out が既にあれば、何も書かずに終了コード 2 で
   * 止まる。
   */
  @Test
  public void rowsSplitTheRecordsIntoChunkFiles() throws Exception {
    Path root = Files.createTempDirectory("classic-rows-chunks");
    try {
      Path records = root.resolve("records");
      for (int index = 0; index < 5; index++) {
        RuleGames.write(records, RuleGames.play(VARIANT, 4_000 + index, "chunk-" + index));
      }
      Path split = root.resolve("split");
      Ran result = run("rows", "--records", records.toString(), "--out", split.toString(), "--chunk-games", "2");
      Assert.assertEquals(result.exit(), 0, result.toString());
      Assert.assertEquals(result.summary().get("chunks"), "3");
      Assert.assertEquals(result.summary().get("games"), "5");
      Assert.assertEquals(result.summary().get("skipped"), "0");
      Assert.assertEquals(
          RowChunks.files(split), List.of(RowChunks.file(split, 0), RowChunks.file(split, 1), RowChunks.file(split, 2)));
      Assert.assertEquals(
          RowChunks.files(split).stream().map(file -> read(file).games().size()).toList(), List.of(2, 2, 1));

      Path settings = Files.writeString(root.resolve("classic.toml"), "[rows]\nchunkGames = 2\n");
      Path fromFile = root.resolve("from-file");
      Ran section =
          run("rows", "--records", records.toString(), "--out", fromFile.toString(), "--settings", settings.toString());
      Assert.assertEquals(section.exit(), 0, section.toString());
      Assert.assertEquals(RowChunks.files(fromFile).size(), 3);

      Path whole = root.resolve("whole");
      Ran wholeResult = run("rows", "--records", records.toString(), "--out", whole.toString());
      Assert.assertEquals(wholeResult.summary().get("chunks"), "1");
      RowChunks splitRows = RowChunks.read(split);
      RowChunks wholeRows = RowChunks.read(whole);
      Assert.assertEquals(String.valueOf(splitRows.size()), result.summary().get("rows"));
      Assert.assertEquals(wholeRows.size(), splitRows.size());
      Assert.assertEquals(wholeRows.games(), splitRows.games());
      for (int row = 0; row < splitRows.size(); row++) {
        Assert.assertEquals(splitRows.gameIndex(row), wholeRows.gameIndex(row), "row " + row);
        Assert.assertEquals(splitRows.round(row), wholeRows.round(row), "row " + row);
        Assert.assertEquals(splitRows.chosen(row), wholeRows.chosen(row), "row " + row);
      }

      Ran refused = run("rows", "--records", records.toString(), "--out", split.toString());
      Assert.assertEquals(refused.exit(), 2, refused.toString());
      Assert.assertEquals(RowChunks.files(split).size(), 3);
    } finally {
      delete(root);
    }
  }

  /**
   * pretrain の区切り: --pause-file があれば最初のステップの前に止まって終了コード 3 で終わり、checkpoint も報告も書かない。ファイルを
   * 消して同じ引数で呼ぶと最後まで学ぶ。そのとき checkpoint のステップ数は全体のステップ数で、基準の値は止めた回と同じ。
   */
  @Test(groups = "native", timeOut = 1_200_000)
  public void pretrainingPausesWithoutACheckpointAndResumes() throws Exception {
    Path root = Files.createTempDirectory("classic-pretrain-pause");
    try {
      Path records = root.resolve("records");
      boolean validation = false;
      int trainGames = 0;
      for (int index = 0; !validation || trainGames < 1; index++) {
        String id = "pause-" + index;
        boolean held = LogRows.validation(id, 0.1);
        if ((held && !validation) || (!held && trainGames < 1)) {
          RuleGames.write(records, RuleGames.play(VARIANT, 5_000 + index, id));
          validation |= held;
          trainGames += held ? 0 : 1;
        }
      }
      Path rows = root.resolve("rows");
      Assert.assertEquals(run("rows", "--records", records.toString(), "--out", rows.toString()).exit(), 0);
      Path nano = root.resolve("nano");
      Path state = root.resolve("state");
      Path pauseFile = root.resolve("pause");
      Files.writeString(pauseFile, "");
      String[] arguments = {
        "pretrain", "--rows", rows.toString(), "--out", nano.toString(), "--architecture", "NANO", "--epochs", "1",
        "--batch", "64", "--state", state.toString(), "--pause-file", pauseFile.toString()
      };
      Ran paused = run(arguments);
      Assert.assertEquals(paused.exit(), 3, paused.toString());
      Assert.assertEquals(paused.summary().get("finished"), "false");
      Assert.assertEquals(paused.summary().get("steps"), "0");
      Assert.assertFalse(Files.exists(nano));
      Assert.assertFalse(Files.exists(root.resolve("nano" + Pretrain.REPORT_SUFFIX)));
      Assert.assertTrue(Files.isRegularFile(state.resolve("progress.json")));

      Files.delete(pauseFile);
      Ran finished = run(arguments);
      Assert.assertEquals(finished.exit(), 0, finished.toString());
      Assert.assertTrue(finished.number("steps") > 0, finished.toString());
      Assert.assertEquals(finished.summary().get("uniform"), paused.summary().get("uniform"));
      Assert.assertEquals(finished.summary().get("rulebot"), paused.summary().get("rulebot"));
      Assert.assertEquals(String.valueOf(ClassicCheckpoint.header(nano).step()), finished.summary().get("steps"));
      Assert.assertTrue(Files.isRegularFile(root.resolve("nano" + Pretrain.REPORT_SUFFIX)));
    } finally {
      delete(root);
    }
  }

  /** pretrain-grp は停止の要求を受けると、checkpoint を書かずに止まる。 */
  @Test(groups = "native", timeOut = 600_000)
  public void grpPretrainingStopsWithoutACheckpoint() throws Exception {
    Path root = Files.createTempDirectory("classic-grp-stop");
    try {
      Path records = root.resolve("records");
      int games = 0;
      boolean stopGame = false;
      boolean trainGame = false;
      for (int index = 0; !stopGame || !trainGame; index++) {
        String id = "stop-" + index;
        RuleGames.write(records, RuleGames.play(VARIANT, 6_000 + index, id));
        boolean stop = GrpSlice.of(LogGames.gameId(id)) == GrpSlice.STOP;
        stopGame |= stop;
        trainGame |= !stop;
        games++;
      }
      Path out = root.resolve("grp");
      PretrainGrp.Result stopped =
          new PretrainGrp(records, out, VARIANT, 1, 1)
              .train(Settings.bind(ClassicSettings.class, List.of()).value().grp().training(), () -> true);
      Assert.assertEquals(stopped.training().trained().ending(), TrainingEnding.STOPPED);
      Assert.assertEquals(stopped.games(), games);
      Assert.assertFalse(Files.exists(out));
    } finally {
      delete(root);
    }
  }

  @Test(groups = "native")
  public void initWritesAFreshCheckpointOnce() throws Exception {
    Path root = Files.createTempDirectory("classic-init");
    try {
      Path out = root.resolve("major");
      Ran ran = run("init", "--architecture", "MAJOR", "--out", out.toString(), "--seed", "3");
      Assert.assertEquals(ran.exit(), 0, ran.toString());
      ClassicHeader header = ClassicCheckpoint.header(out);
      Assert.assertEquals(header.step(), 0);
      Assert.assertEquals(header.utility(), Networks.utility(Architecture.MAJOR));
      Assert.assertEquals(ran.summary().get("architecture"), "MAJOR");
      Ran again = run("init", "--architecture", "MAJOR", "--out", out.toString(), "--seed", "4");
      Assert.assertEquals(again.exit(), 2, again.toString());
      Assert.assertTrue(again.logged("The output checkpoint already exists: " + out), again.toString());
      Assert.assertEquals(ClassicCheckpoint.header(out), header);
    } finally {
      delete(root);
    }
  }

  private static <T extends Record> T section(Class<T> type, String path, String body) {
    String text = "[" + path + "]\n" + body + "\n";
    return Settings.section(type, path, List.of(Layer.parse("classic.toml", Layer.Source.FILE, text))).value();
  }

  private static Ran run(String... args) {
    Logger root = (Logger) LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
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

  private static ClassicRows read(Path file) {
    try {
      return ClassicRows.read(file);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  private static void delete(Path root) throws Exception {
    try (Stream<Path> paths = Files.walk(root)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
        Files.deleteIfExists(path);
      }
    }
  }
}
