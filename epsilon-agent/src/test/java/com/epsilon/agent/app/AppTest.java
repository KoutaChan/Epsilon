package com.epsilon.agent.app;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.epsilon.agent.app.DemoApp.Demo;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * アプリの入口のテスト。引数からキーへの対応、くり返しと知らない名前、必須の引数の欠け、範囲、2 語の名前、使い方の表示、終了コード、
 * run を動かすコマンドに渡す層、開始と終了のログの行、設定ファイルとプリセット、組み込みの settings のコマンド、コマンドの宣言のエラーを確かめる。
 */
public class AppTest {

  /** 1 回の呼び出しの結果。 */
  record Ran(int exit, String out, List<String> logs) {

    boolean logged(String line) {
      return logs.contains(line);
    }

    boolean loggedStart(String prefix) {
      return logs.stream().anyMatch(line -> line.startsWith(prefix));
    }
  }

  private static Ran run(Application<?> application, Signals signals, String... args) {
    Logger root = (Logger) LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
    ListAppender<ILoggingEvent> appender = new ListAppender<>();
    appender.start();
    root.addAppender(appender);
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    int exit;
    try (PrintStream out = new PrintStream(bytes, true, StandardCharsets.UTF_8)) {
      exit = App.run(application, args, out, signals);
    } finally {
      root.detachAppender(appender);
    }
    List<String> logs = appender.list.stream().map(ILoggingEvent::getFormattedMessage).toList();
    return new Ran(exit, bytes.toString(StandardCharsets.UTF_8), logs);
  }

  private static Ran run(String... args) {
    return run(DemoApp.application(), new Signals(), args);
  }

  @Test
  public void exitCodesFollowTheTable() {
    assertEquals(
        Arrays.stream(Exit.values()).map(exit -> exit.name() + "=" + exit.code()).toList(),
        List.of("OK=0", "FAILED=1", "USAGE=2", "STOPPED=3", "PROBLEMS=4", "RESTART=75", "YIELD=76", "INTERRUPTED=130"));
  }

  @Test
  public void argumentsMapToTheComponentsOfTheCommand() {
    Ran train = run("train", "--run", "/r", "--stop-after", "3");
    assertEquals(train.exit(), 0, train.logs().toString());
    assertEquals(train.out(), "train /r 3");
    Ran eval =
        run("eval", "run", "--table", "a", "--mode", "SLOW", "--table", "b", "--out", "/o", "--set", "workers=3");
    assertEquals(eval.exit(), 0, eval.logs().toString());
    assertEquals(eval.out(), "table=[a, b] mode=SLOW out=/o workers=3 rate=0.5 presets=[]");
    Ran preset = run("--preset", "fast", "eval", "run");
    assertEquals(preset.out(), "table=[] mode=FAST out=null workers=2 rate=0.5 presets=[fast]");
    assertEquals(Catalog.argument("stopAfter"), "stop-after");
  }

  @Test
  public void everyArgumentProblemIsReportedAndExitsWithUsage() {
    Ran unknown = run("train", "--run", "/r", "--stop-aftr", "2");
    assertEquals(unknown.exit(), 2);
    assertTrue(
        unknown.logged("Unknown argument --stop-aftr for train (did you mean --stop-after?)"),
        unknown.logs().toString());
    Ran twice = run("train", "--run", "/a", "--run", "/b");
    assertEquals(twice.exit(), 2);
    assertTrue(twice.logged("Argument --run is given twice"), twice.logs().toString());
    Ran missing = run("train");
    assertEquals(missing.exit(), 2);
    assertTrue(missing.loggedStart("Missing argument --run"), missing.logs().toString());
    Ran range = run("train", "--run", "/r", "--stop-after", "0");
    assertEquals(range.exit(), 2);
    assertTrue(range.loggedStart("Invalid argument --stop-after"), range.logs().toString());
    Ran choice = run("eval", "run", "--mode", "MEDIUM");
    assertEquals(choice.exit(), 2);
    assertTrue(choice.loggedStart("Invalid argument --mode"), choice.logs().toString());
    Ran noValue = run("train", "--run");
    assertEquals(noValue.exit(), 2);
    assertTrue(noValue.logged("Argument --run needs a value"), noValue.logs().toString());
    Ran extra = run("train", "extra", "--run", "/r");
    assertEquals(extra.exit(), 2);
    assertTrue(extra.logged("Unexpected argument extra for train"), extra.logs().toString());
    Ran all = run("train", "--run", "/r", "--bogus", "1", "--stop-after", "1", "--stop-after", "2");
    assertEquals(all.exit(), 2);
    assertTrue(all.logged("Unknown argument --bogus for train"), all.logs().toString());
    assertTrue(all.logged("Argument --stop-after is given twice"), all.logs().toString());
    assertTrue(all.loggedStart("Command finished: demo train USAGE in "), all.logs().toString());
  }

  @Test
  public void commandNamesTakeTheLongestMatchOfOneOrTwoWords() {
    Ran one = run("eval");
    assertEquals(one.exit(), 2);
    assertTrue(one.logged("Unknown command eval; run 'demo help' for the commands"), one.logs().toString());
    Ran typo = run("settings", "chnage", "--run", "/r");
    assertEquals(typo.exit(), 2);
    assertTrue(
        typo.logged(
            "Unknown command settings chnage (did you mean settings change?); run 'demo help' for the commands"),
        typo.logs().toString());
    Ran none = run();
    assertEquals(none.exit(), 2);
    assertTrue(none.logged("No command given"));
    assertTrue(none.out().startsWith("Usage: demo <command> [--name value ...]\n"), none.out());
  }

  @Test
  public void helpListsTheCommandsAndTheArgumentsOfOne() {
    Ran list = run("help");
    assertEquals(list.exit(), 0);
    assertTrue(list.out().contains("  eval run"), list.out());
    assertTrue(list.out().contains("卓を測る"), list.out());
    assertTrue(list.out().contains("  settings change"), list.out());
    Ran eval = run("help", "eval", "run");
    assertEquals(eval.exit(), 0, eval.logs().toString());
    assertTrue(eval.out().startsWith("demo eval run [--table <"), eval.out());
    assertTrue(eval.out().contains("repeatable"), eval.out());
    assertTrue(eval.out().contains("default FAST"), eval.out());
    assertTrue(eval.out().contains("[eval.run]"), eval.out());
    Ran train = run("help", "--command", "train");
    assertTrue(train.out().startsWith("demo train --run <"), train.out());
    assertTrue(train.out().contains("required"), train.out());
    assertTrue(train.out().contains("Reads the settings recorded in the run"), train.out());
    Ran typo = run("help", "trian");
    assertEquals(typo.exit(), 2);
    assertTrue(typo.logged("Unknown command trian (did you mean train?)"), typo.logs().toString());
  }

  @Test
  public void aCommandThatDrivesARunRefusesLayers() {
    for (List<String> layer :
        List.of(List.of("--set", "workers=3"), List.of("--preset", "fast"), List.of("--settings", "/x.toml"))) {
      String[] args = new String[] {"train", "--run", "/r", layer.get(0), layer.get(1)};
      Ran refused = run(args);
      assertEquals(refused.exit(), 2, layer.toString());
      assertTrue(
          refused.logged(
              "Use 'demo settings change --run DIR' to change a running lineage (train takes no --settings, --preset"
                  + " or --set)"),
          refused.logs().toString());
    }
  }

  @Test
  public void theStartAndFinishLinesNameTheCommandAndHideSecrets() {
    Ran train = run("train", "--run", "/r");
    assertTrue(train.logged("Command started: demo train --run /r"), train.logs().toString());
    assertTrue(
        train.logs().stream()
            .anyMatch(line -> line.matches("Command finished: demo train OK in \\d+\\.\\d s: trained")),
        train.logs().toString());
    Ran secret = run("eval", "run", "--table", "a", "--set", "token=abc");
    assertTrue(
        secret.logged("Command started: demo eval run --table a --set token=********"), secret.logs().toString());
    assertFalse(secret.logs().stream().anyMatch(line -> line.contains("abc")), secret.logs().toString());
    assertEquals(secret.exit(), 2);
  }

  /** 秘密のキーを持つコマンドの口。 */
  sealed interface Keyed extends Command<Demo> permits Login, RemoteLogin {}

  @Named(value = "login", doc = "秘密のキーを持つコマンド")
  record Login(@Key(doc = "トークン", presence = Presence.ABSENT, secret = true) String token) implements Keyed {
    @Override
    public Outcome run(Context<Demo> context) {
      return Outcome.ok("never");
    }
  }

  @Named(value = "remote login", doc = "秘密のキーを持つ 2 語のコマンド")
  record RemoteLogin(@Key(doc = "トークン", presence = Presence.ABSENT, secret = true) String token)
      implements Keyed {
    @Override
    public Outcome run(Context<Demo> context) {
      return Outcome.ok("never");
    }
  }

  /** コマンドの節の秘密のキーに {@code --set} で書いた値も、始めのログの行で伏せる。 */
  @Test
  public void theStartLineHidesASecretOfTheCommandSectionWrittenWithSet() {
    List<Class<? extends Command<Demo>>> commands = List.of(Keyed.class);
    Application<Demo> application = Application.of("demo", Demo.class, commands);
    Ran one = run(application, new Signals(), "login", "--set", "login.token=abc");
    assertTrue(one.logged("Command started: demo login --set login.token=********"), one.logs().toString());
    assertFalse(one.logs().stream().anyMatch(line -> line.contains("abc")), one.logs().toString());
    assertTrue(
        one.logged("Setting login.token at --set is secret and is read only from the settings file"),
        one.logs().toString());
    assertEquals(one.exit(), 2);
    Ran two = run(application, new Signals(), "remote", "login", "--set", "remote.login.token=abc");
    assertTrue(
        two.logged("Command started: demo remote login --set remote.login.token=********"), two.logs().toString());
    assertFalse(two.logs().stream().anyMatch(line -> line.contains("abc")), two.logs().toString());
    assertEquals(two.exit(), 2);
  }

  @Test
  public void failuresInterruptsAndStopsMapToTheirExitCodes() {
    Ran boom = run("boom");
    assertEquals(boom.exit(), 1);
    assertTrue(
        boom.logs().stream()
            .anyMatch(
                line ->
                    line.matches("Command finished: demo boom FAILED in .* s: java.lang.IllegalStateException: boom")),
        boom.logs().toString());
    assertEquals(run("wait").exit(), 130);
    // 入口はスレッドの割り込みの状態を残す（テストのスレッドでは、ここで消しておく）
    assertTrue(Thread.interrupted());
    Signals signals = new Signals();
    assertEquals(run(DemoApp.application(), signals, "halt").exit(), 0);
    signals.request("SIGTERM");
    Ran stopped = run(DemoApp.application(), signals, "halt");
    assertEquals(stopped.exit(), 3);
    assertTrue(stopped.loggedStart("Command finished: demo halt STOPPED in "), stopped.logs().toString());
    assertTrue(
        stopped.logs().stream().anyMatch(line -> line.endsWith("s: stopped by SIGTERM")), stopped.logs().toString());
  }

  @Test
  public void aSettingsFileHoldsPresetsSettingsAndCommandSections() throws Exception {
    Path file = Files.createTempDirectory("app-settings").resolve("demo.toml");
    Files.writeString(file, "presets = [\"fast\"]\nrate = 0.25\n\n[eval.run]\nmode = \"SLOW\"\n");
    Ran read = run("eval", "run", "--settings", file.toString());
    assertEquals(read.exit(), 0, read.logs().toString());
    assertEquals(read.out(), "table=[] mode=SLOW out=null workers=2 rate=0.25 presets=[fast]");
    Ran overridden = run("eval", "run", "--settings", file.toString(), "--mode", "FAST", "--set", "rate=0.5");
    assertEquals(overridden.out(), "table=[] mode=FAST out=null workers=2 rate=0.5 presets=[fast]");
    Ran missing = run("eval", "run", "--settings", file.resolveSibling("none.toml").toString());
    assertEquals(missing.exit(), 2);
    assertTrue(
        missing.logged("Settings file " + file.resolveSibling("none.toml") + " does not exist"),
        missing.logs().toString());
    Ran preset = run("eval", "run", "--preset", "nope");
    assertEquals(preset.exit(), 2);
    assertTrue(preset.logged("Unknown preset nope"), preset.logs().toString());
    Path typo = file.resolveSibling("typo.toml");
    Files.writeString(typo, "[settings.chnage]\nrun = \"/r\"\n");
    Ran section = run("help", "--settings", typo.toString());
    assertEquals(section.exit(), 2);
    assertTrue(section.loggedStart("Unknown setting settings.chnage at "), section.logs().toString());
    assertTrue(
        section.logs().stream().anyMatch(line -> line.endsWith("(did you mean [settings.change]?)")),
        section.logs().toString());
  }

  @Test
  public void theSettingsCommandsCreateShowChangeAndCheckARun() throws Exception {
    Path run = Files.createTempDirectory("app-run").resolve("run");
    String dir = run.toString();
    SettingsRows rows = new SettingsRows(new RunDirectory(run));

    assertEquals(run("settings", "create", "--run", dir, "--set", "workers=4").exit(), 0);
    SettingsRow first = rows.latest();
    assertEquals(first.row(), 1);
    assertEquals(first.from(), 1);
    assertEquals(first.reason(), SettingsRow.CREATED);
    assertEquals(first.bind(Demo.class).value().workers(), 4);
    Ran again = run("settings", "create", "--run", dir);
    assertEquals(again.exit(), 2);
    assertTrue(again.logged("The run " + dir + " already has settings rows; change them with settings change"));

    Ran change = run("settings", "change", "--run", dir, "--change", "workers=6", "--reason", "more");
    assertEquals(change.exit(), 0, change.logs().toString());
    SettingsRow second = rows.latest();
    assertEquals(second.row(), 2);
    assertEquals(second.from(), 1);
    assertEquals(second.reason(), "more");
    assertEquals(second.changes().size(), 1);
    assertEquals(second.changes().getFirst().key(), "workers");
    assertEquals(second.changes().getFirst().change(), Change.MACHINE);
    assertTrue(change.logged("Settings row 2 takes effect from iteration 1"), change.logs().toString());
    assertTrue(
        change.logged(
            "  MACHINE: takes effect after restart: run 'demo control restart --run " + dir + " --at ITERATION'"),
        change.logs().toString());

    // 反復 1 が始まった後に足した設定の行は、反復 2 から効く
    Files.writeString(run.resolve("state.json"), "{\"next\": 1}\n");
    Files.createDirectories(run.resolve("work").resolve("00001"));
    Files.writeString(run.resolve("work").resolve("00001").resolve("ITERATION.json"), "{}\n");
    assertEquals(run("settings", "change", "--run", dir, "--change", "rate=0.75", "--reason", "rate").exit(), 0);
    assertEquals(rows.latest().from(), 2);

    Ran fixed = run("settings", "change", "--run", dir, "--change", "seed=8", "--reason", "seed");
    assertEquals(fixed.exit(), 2);
    assertTrue(
        fixed.loggedStart("Setting seed is fixed for the run: the run started with 7, "), fixed.logs().toString());
    Ran same = run("settings", "change", "--run", dir, "--change", "workers=6", "--reason", "same");
    assertEquals(same.exit(), 0);
    assertTrue(same.logs().stream().anyMatch(line -> line.endsWith("s: no change to row 3")), same.logs().toString());
    assertEquals(rows.read().size(), 3);

    Ran show = run("settings", "show", "--run", dir);
    assertEquals(show.exit(), 0, show.logs().toString());
    assertTrue(show.out().startsWith("# settings row 3 of 3 (from iteration 2), written "), show.out());
    assertTrue(show.out().contains("rate = 0.75  # settings row 3\n"), show.out());
    // 既定値と違う値には最後に変えた行を添え、既定値と同じ値（seed = 7）には何も添えない
    assertTrue(show.out().contains("workers = 6  # settings row 2\n"), show.out());
    assertTrue(show.out().contains("seed = 7\n"), show.out());
    Ran layered = run("settings", "show", "--run", dir, "--set", "workers=1");
    assertEquals(layered.exit(), 2);
    assertTrue(layered.logged("This command reads the run's settings: drop --settings, --preset and --set"));

    Application<Demo> checked =
        DemoApp.application()
            .withCheck(
                settings -> settings.workers() > 5 ? List.of("Too many workers: " + settings.workers()) : List.of());
    Ran problems = run(checked, new Signals(), "settings", "check", "--run", dir);
    assertEquals(problems.exit(), 4);
    assertTrue(problems.logged("Too many workers: 6"), problems.logs().toString());
    Ran refused =
        run(checked, new Signals(), "settings", "change", "--run", dir, "--change", "workers=7", "--reason", "x");
    assertEquals(refused.exit(), 4);
    assertEquals(rows.read().size(), 3);
    assertEquals(run("settings", "check", "--run", dir).exit(), 0);

    Path absent = run.resolveSibling("absent");
    Ran nothing = run("settings", "change", "--run", absent.toString(), "--change", "workers=1", "--reason", "x");
    assertEquals(nothing.exit(), 2);
    assertTrue(nothing.logged("The run " + absent + " has no settings rows: create it with settings create"));
    assertFalse(Files.exists(absent));
    assertEquals(run("settings", "check", "--run", absent.toString()).exit(), 4);
    Ran empty = run("settings", "change", "--run", dir, "--reason", "x");
    assertEquals(empty.exit(), 2);
    assertTrue(empty.logged("settings change needs --change or --apply"));
  }

  @Test
  public void settingsChangeLayersThePresetsOfTheAppliedFile() throws Exception {
    Path run = Files.createTempDirectory("app-apply").resolve("run");
    String dir = run.toString();
    SettingsRows rows = new SettingsRows(new RunDirectory(run));
    assertEquals(run("settings", "create", "--run", dir, "--set", "workers=4").exit(), 0);

    // プリセットの名前だけを書いたファイルでも、プリセットの値で行が足される
    Path preset = run.resolveSibling("preset.toml");
    Files.writeString(preset, "presets = [\"fast\"]\n");
    Ran applied = run("settings", "change", "--run", dir, "--apply", preset.toString(), "--reason", "preset");
    assertEquals(applied.exit(), 0, applied.logs().toString());
    SettingsRow second = rows.latest();
    assertEquals(second.row(), 2);
    assertEquals(second.changes().size(), 1);
    assertEquals(second.changes().getFirst().key(), "workers");
    assertEquals(second.bind(Demo.class).value().workers(), 2);

    // ファイルに書いたキーは、プリセットの値の後に重なる
    Path both = run.resolveSibling("both.toml");
    Files.writeString(both, "presets = [\"fast\"]\nworkers = 3\n");
    Ran layered = run("settings", "change", "--run", dir, "--apply", both.toString(), "--reason", "file");
    assertEquals(layered.exit(), 0, layered.logs().toString());
    assertEquals(rows.latest().row(), 3);
    assertEquals(rows.latest().bind(Demo.class).value().workers(), 3);
  }

  @Test
  public void theTemplateWritesEveryKeyWithItsDefaultCommentedOut() {
    Ran template = run("settings", "template");
    assertEquals(template.exit(), 0, template.logs().toString());
    assertTrue(template.out().contains("# workers = 8"), template.out());
    Ran set = run("settings", "template", "--set", "workers=3");
    assertTrue(set.out().contains("\nworkers = 3"), set.out());
  }

  /** 宣言のエラーを確かめるためのコマンドのインターフェース。 */
  sealed interface Unnamed extends Command<Demo> permits NoName {}

  record NoName() implements Unnamed {
    @Override
    public Outcome run(Context<Demo> context) {
      return Outcome.ok("never");
    }
  }

  sealed interface ThreeWords extends Command<Demo> permits Long3 {}

  @Named(value = "a b c", doc = "3 語")
  record Long3() implements ThreeWords {
    @Override
    public Outcome run(Context<Demo> context) {
      return Outcome.ok("never");
    }
  }

  sealed interface Unkeyed extends Command<Demo> permits Plain {}

  @Named(value = "plain", doc = "鍵の無い引数")
  record Plain(int x) implements Unkeyed {
    @Override
    public Outcome run(Context<Demo> context) {
      return Outcome.ok("never");
    }
  }

  sealed interface Taken extends Command<Demo> permits SetArgument {}

  @Named(value = "taken", doc = "--set と同じ名前の引数")
  record SetArgument(@Key(doc = "値", value = "a") String set) implements Taken {
    @Override
    public Outcome run(Context<Demo> context) {
      return Outcome.ok("never");
    }
  }

  sealed interface Section extends Command<Demo> permits Workers {}

  @Named(value = "workers", doc = "設定の節と同じ名前")
  record Workers() implements Section {
    @Override
    public Outcome run(Context<Demo> context) {
      return Outcome.ok("never");
    }
  }

  sealed interface Twins extends Command<Demo> permits TwinA, TwinB {}

  @Named(value = "twin", doc = "1 つ目")
  record TwinA() implements Twins {
    @Override
    public Outcome run(Context<Demo> context) {
      return Outcome.ok("never");
    }
  }

  @Named(value = "twin", doc = "2 つ目")
  record TwinB() implements Twins {
    @Override
    public Outcome run(Context<Demo> context) {
      return Outcome.ok("never");
    }
  }

  sealed interface Prefix extends Command<Demo> permits Eval {}

  @Named(value = "eval", doc = "eval run の頭の語")
  record Eval() implements Prefix {
    @Override
    public Outcome run(Context<Demo> context) {
      return Outcome.ok("never");
    }
  }

  @SafeVarargs
  private static String declarationError(Class<? extends Command<Demo>>... commands) {
    Application<Demo> application = Application.of("demo", Demo.class, List.of(commands));
    return expectThrows(IllegalStateException.class, () -> Catalog.of(application)).getMessage();
  }

  @Test
  public void badCommandDeclarationsAreRefusedWhenTheCatalogIsBuilt() {
    assertEquals(declarationError(Unnamed.class), "The command " + NoName.class.getName() + " has no @Named");
    assertEquals(
        declarationError(ThreeWords.class), "Invalid command name 'a b c' on " + Long3.class.getName());
    assertEquals(declarationError(Unkeyed.class), "The argument Plain.x has no @Key");
    assertEquals(declarationError(Taken.class), "The argument --set of taken is taken");
    assertEquals(
        declarationError(Section.class),
        "The command workers shares its first word with the settings section workers");
    assertEquals(declarationError(Twins.class), "Two commands are named twin");
    assertEquals(
        declarationError(DemoApp.DemoCommand.class, Prefix.class),
        "The command eval is also the first word of other commands");
  }
}
