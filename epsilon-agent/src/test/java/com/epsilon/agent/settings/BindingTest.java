package com.epsilon.agent.settings;

import static com.epsilon.agent.settings.Fixtures.assertProblems;
import static com.epsilon.agent.settings.Fixtures.bind;
import static com.epsilon.agent.settings.Fixtures.file;
import static com.epsilon.agent.settings.Fixtures.minimal;
import static com.epsilon.agent.settings.Fixtures.problems;
import static com.epsilon.agent.settings.Fixtures.set;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.runtime.RuntimeSettings;
import com.epsilon.agent.settings.Fixtures.Cohort;
import com.epsilon.agent.settings.Fixtures.Evaluation;
import com.epsilon.agent.settings.Fixtures.Fit;
import com.epsilon.agent.settings.Fixtures.Flags;
import com.epsilon.agent.settings.Fixtures.Kind;
import com.epsilon.agent.settings.Fixtures.Memory;
import com.epsilon.agent.settings.Fixtures.Riichi;
import com.epsilon.agent.settings.Fixtures.Root;
import com.epsilon.agent.settings.Fixtures.Run;
import com.epsilon.agent.settings.Fixtures.Stop;
import com.epsilon.agent.settings.Fixtures.Thinning;
import com.epsilon.agent.settings.Fixtures.Variant;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.testng.annotations.Test;

/** 層を重ねた木を record に結び付ける。全ての型、キーの有無の 3 通り、範囲、選んだ形に無いキー、record の中の検査、似た名前の提案、エラーを全て集めることを確かめる。 */
public class BindingTest {

  /** run の必須のキーを 3 行に書き、4 行目から渡された続きを書いた設定ファイル。 */
  private static Layer withRun(String rest) {
    return file("[run]\nvariant = \"FOUR\"\nseed = 1\n" + rest);
  }

  @Test
  public void bindsEveryType() {
    Layer file =
        file(
            """
            [run]
            variant = "FOUR"
            seed = 20_261_110
            name = 1234
            directory = '/runs/4p-b'

            [fit]
            lrPeak = 0.002
            lrFloor = 0
            temperature = {CALL = 0.5}
            classes = ["KAN"]

            [memory]
            halfLife = 4

            [thinning]
            enabled = true
            tau = 3

            [[cohorts]]
            from = 1
            games = 64

            [flags]
            verbose = false
            scale = 0.25
            sizes = []
            at = "2026-11-09T00:00:00Z"
            limit = -9_000_000_000
            labels = {b = "y", "c d" = 'z'}
            nested = [[], [4]]

            [inference]
            devices = ["gpu0", "gpu1"]
            precision = "BFLOAT16"
            kernels = "NATIVE"

            [evaluation.tables.x1n3]
            opponent = 'nano:/home/x'
            walls = 131072
            stop = {kind = "TEST", evenMargin = 0.01}
            checkpoint = "/tmp/c"

            [evaluation.tables.fixed]
            opponent = "heuristic"
            walls = 8
            stop = "FIXED"

            [riichi]
            token = "abc"
            """);
    Settings<Root> settings = bind(file);
    Root expected =
        new Root(
            new Run(Variant.FOUR, 20261110L, "1234", Path.of("/runs/4p-b")),
            new Fit(0.002, 0.0, Map.of(Kind.DISCARD, 0.01, Kind.CALL, 0.5, Kind.KAN, 0.02), Set.of(Kind.KAN)),
            new Memory.Ema(4.0, 16),
            new Thinning.On(3.0),
            List.of(new Cohort(1, 64)),
            new Flags(
                false,
                0.25f,
                List.of(),
                Instant.parse("2026-11-09T00:00:00Z"),
                -9_000_000_000L,
                Map.of("a", "x", "b", "y", "c d", "z"),
                List.of(List.of(), List.of(4))),
            new InferenceSettings(List.of("gpu0", "gpu1"), Precision.BFLOAT16, Kernels.NATIVE, 8, 20, 2, 4),
            new TableSettings(8192, 28, 100000, 4),
            new RuntimeSettings(1),
            new Evaluation(
                Map.of(
                    "x1n3",
                    new Fixtures.Table("nano:/home/x", 131072, new Stop.Sequential(0.01, 0.01), Path.of("/tmp/c")),
                    "fixed",
                    new Fixtures.Table("heuristic", 8, new Stop.Fixed(), null))),
            new Riichi("abc"));
    assertEquals(settings.value(), expected);
    assertEquals(settings.change("run.seed"), Change.FIXED);
    assertEquals(settings.change("fit.temperature.CALL"), Change.FIT);
    assertEquals(settings.change("memory.scheme"), Change.FIT);
    assertEquals(settings.change("memory.halfLife"), Change.FIT);
    assertEquals(settings.change("thinning.enabled"), Change.COLLECT);
    assertEquals(settings.change("cohorts"), Change.COLLECT);
    assertNull(settings.change("cohorts[0].from"));
    assertEquals(settings.change("flags.labels.b"), Change.MACHINE);
    assertEquals(settings.change("flags.nested"), Change.MACHINE);
    assertEquals(settings.change("inference.precision"), Change.COLLECT);
    assertEquals(settings.change("inference.devices"), Change.MACHINE);
    assertEquals(settings.change("evaluation.tables.x1n3.stop.kind"), Change.SCHEDULE);
    assertEquals(settings.change("evaluation.tables.x1n3.stop.evenMargin"), Change.SCHEDULE);
    assertEquals(settings.change("riichi.token"), Change.MACHINE);
    assertNull(settings.change("fit"));
  }

  @Test
  public void defaultsFillEveryAbsentKey() {
    Root root = bind(minimal()).value();
    assertEquals(root.run(), new Run(Variant.FOUR, 20261110L, "lineage", Path.of("runs/x")));
    assertEquals(
        root.fit(),
        new Fit(0.0012, 0.00006, Map.of(Kind.DISCARD, 0.01, Kind.CALL, 0.02, Kind.KAN, 0.02), Set.of(Kind.DISCARD, Kind.CALL)));
    assertEquals(root.memory(), new Memory.Ema(8.0, 16));
    assertEquals(root.thinning(), new Thinning.Off());
    assertEquals(root.cohorts(), List.of(new Cohort(1, 8192), new Cohort(9, 16384)));
    assertEquals(
        root.flags(),
        new Flags(
            true,
            1.5f,
            List.of(1, 2),
            Instant.parse("2026-11-12T03:00:00Z"),
            9_000_000_000L,
            Map.of("a", "x"),
            List.of(List.of(1), List.of(2, 3))));
    assertEquals(root.inference(), new InferenceSettings(List.of("cpu"), Precision.FLOAT32, Kernels.EAGER, 8, 20, 2, 4));
    assertEquals(root.evaluation(), new Evaluation(Map.of()));
    assertEquals(root.riichi(), new Riichi(null));
  }

  @Test
  public void stringsTakeNumbersAndBooleansAsWritten() {
    Run run = bind(withRun("name = 1.50\ndirectory = true\n")).value().run();
    assertEquals(run.name(), "1.50");
    assertEquals(run.directory(), Path.of("true"));
    assertEquals(bind(minimal(), set("run.name=true", "run.directory=12")).value().run().directory(), Path.of("12"));
    assertEquals(bind(minimal(), set("run.name=true")).value().run().name(), "true");
  }

  @Test
  public void stringsKeepTheSignAndTheSeparatorsOfANumber() {
    Layer written = set("run.name=2026_11_11", "run.directory=+5", "flags.labels={a = 20261110_2}");
    Settings<Root> settings = bind(minimal(), written);
    assertEquals(settings.value().run().name(), "2026_11_11");
    assertEquals(settings.value().run().directory(), Path.of("+5"));
    assertEquals(settings.value().flags().labels(), Map.of("a", "20261110_2"));
    assertEquals(bind(withRun("name = +1_000.5\n")).value().run().name(), "+1_000.5");
    assertEquals(bind(minimal(), set("run.seed=+1_000")).value().run().seed(), 1000L);
    // 設定の行とひな形から読み直しても、書いたままの文字列になる
    assertEquals(bind(Layer.tree("settings row 1", Json.read(settings.json(), "row"))).value(), settings.value());
    String template = Settings.template(Root.class, List.of(minimal(), written), List.of());
    assertEquals(bind(Layer.parse("template.toml", Layer.Source.FILE, template)).value(), settings.value(), template);
  }

  @Test
  public void choicesTakeTheShortForm() {
    Root root = bind(file("memory = \"NONE\"\nthinning = true\n[run]\nvariant = \"FOUR\"\nseed = 1\n")).value();
    assertEquals(root.memory(), new Memory.None());
    assertEquals(root.thinning(), new Thinning.On(2.0));
    assertEquals(bind(minimal(), set("thinning=true")).value().thinning(), new Thinning.On(2.0));
    assertEquals(bind(minimal(), set("thinning.enabled=true", "thinning.tau=5")).value().thinning(), new Thinning.On(5.0));
    assertEquals(
        bind(withRun("[evaluation.tables.t]\nopponent = \"h\"\nwalls = 1\nstop = \"FIXED\"\n"))
            .value()
            .evaluation()
            .tables()
            .get("t")
            .stop(),
        new Stop.Fixed());
    assertProblems(
        List.of(withRun("[evaluation.tables.t]\nopponent = \"h\"\nwalls = 1\nstop = \"TEST\"\n")),
        "Missing setting evaluation.tables.t.stop.evenMargin");
  }

  @Test
  public void threePresences() {
    assertProblems(List.of(file("[run]\nvariant = \"FOUR\"\n")), "Missing setting run.seed");
    assertProblems(List.of(), "Missing setting run.variant", "Missing setting run.seed");
    assertProblems(
        List.of(withRun("[evaluation.tables.y]\nopponent = \"h\"\n")),
        "Missing setting evaluation.tables.y.walls",
        "Missing setting evaluation.tables.y.stop");
    Root root = bind(withRun("[evaluation.tables.y]\nopponent = \"h\"\nwalls = 2\nstop = \"FIXED\"\n")).value();
    assertNull(root.evaluation().tables().get("y").checkpoint());
    assertNull(root.riichi().token());
    assertEquals(root.run().name(), "lineage");
  }

  @Test
  public void rangesAndWidths() {
    assertProblems(
        List.of(
            withRun(
                """
                [memory]
                halfLife = 0
                [fit]
                temperature = {CALL = 1.5}
                [[cohorts]]
                from = 0
                games = 1
                [inference]
                slots = 5
                readyBatches = 5
                [table]
                batchWaitMicros = -1
                [flags]
                limit = 1e3
                """)),
        "Invalid setting fit.temperature.CALL=1.5 at run.toml:7: expected a number in [0.0, 1.0]",
        "Invalid setting memory.halfLife=0 at run.toml:5: expected a number > 0.0",
        "Invalid setting cohorts[0].from=0 at run.toml:9: expected an integer >= 1",
        "Invalid setting flags.limit=1e3 at run.toml:17: expected an integer",
        "Invalid setting inference.slots=5 at run.toml:12: expected an integer in [1, 4]",
        "Invalid setting table.batchWaitMicros=-1 at run.toml:15: expected an integer >= 0");
    assertProblems(
        List.of(minimal(), set("flags.sizes=[3000000000]", "flags.limit=9223372036854775808", "flags.scale=1e39")),
        "Invalid setting flags.scale=1e39 at --set: expected a finite number",
        "Invalid setting flags.sizes[0]=3000000000 at --set: expected an integer in [-2147483648, 2147483647]",
        "Invalid setting flags.limit=9223372036854775808 at --set: expected an integer in "
            + "[-9223372036854775808, 9223372036854775807]");
    Root edges =
        bind(minimal(), set("fit.temperature.CALL=1", "fit.temperature.KAN=0", "inference.slots=4", "table.batchWaitMicros=0"))
            .value();
    assertEquals(edges.fit().temperature().get(Kind.CALL), 1.0);
    assertEquals(edges.inference().slots(), 4);
    assertEquals(edges.table().batchWaitMicros(), 0L);
    assertProblems(
        List.of(withRun("[evaluation.tables.t]\nopponent = \"h\"\nwalls = 1\nstop = {kind = \"TEST\", evenMargin = 0, alpha = 1}\n")),
        "Invalid setting evaluation.tables.t.stop.alpha=1 at run.toml:7: expected a number in (0.0, 1.0)");
  }

  /** float の値でも、範囲の端と同じリテラルを書けば範囲に入る。 */
  record Weighted(
      @Key(doc = "重み", value = "0.1", change = Change.FIXED) @Range(min = 0.1, max = 0.3) float weight) {}

  @Test
  public void floatRangeTakesItsBounds() {
    assertEquals(Settings.bind(Weighted.class, List.of()).value().weight(), 0.1f);
    assertEquals(Settings.bind(Weighted.class, List.of(set("weight=0.3"))).value().weight(), 0.3f);
    SettingsException above =
        expectThrows(SettingsException.class, () -> Settings.bind(Weighted.class, List.of(set("weight=0.31"))));
    assertEquals(above.problems(), List.of("Invalid setting weight=0.31 at --set: expected a number in [0.1, 0.3]"));
  }

  @Test
  public void typeErrors() {
    assertProblems(
        List.of(
            withRun(
                """
                [flags]
                verbose = "yes"
                scale = "big"
                sizes = 3
                at = "later"
                nested = [1]
                [fit]
                classes = ["KAN", "KAN"]
                [table]
                workers = 1.5
                [inference]
                precision = "float32"
                [runtime]
                torchThreads = []
                """)),
        "Invalid setting fit.classes=[\"KAN\", \"KAN\"] at run.toml:11: KAN appears twice",
        "Invalid setting flags.verbose=yes at run.toml:5: expected true or false",
        "Invalid setting flags.scale=big at run.toml:6: expected a number",
        "Invalid setting flags.sizes=3 at run.toml:7: expected a list",
        "Invalid setting flags.at=later at run.toml:8: expected an instant such as 2026-11-12T03:00:00Z",
        "Invalid setting flags.nested[0]=1 at run.toml:9: expected a list",
        "Invalid setting inference.precision=float32 at run.toml:15: expected one of [FLOAT32, BFLOAT16]",
        "Invalid setting table.workers=1.5 at run.toml:13: expected an integer >= 1",
        "Invalid setting runtime.torchThreads=[] at run.toml:17: expected an integer >= 1");
    assertProblems(
        List.of(file("table = 3\nmemory = 3\nevaluation = {tables = 1}\n[run]\nvariant = \"FOUR\"\nseed = 1\n")),
        "Invalid setting memory=3 at run.toml:2: expected a table or one of [NONE, EMA]",
        "Invalid setting table=3 at run.toml:1: expected a table",
        "Invalid setting evaluation.tables=1 at run.toml:3: expected a table");
    List<String> nul = problems(List.of(file("[run]\nvariant = \"FOUR\"\nseed = 1\ndirectory = \"a\\u0000b\"\n")));
    assertEquals(nul.size(), 1);
    assertTrue(nul.getFirst().startsWith("Invalid setting run.directory=a"), nul.getFirst());
    assertTrue(nul.getFirst().endsWith(" at run.toml:4: expected a path"), nul.getFirst());
  }

  @Test
  public void choiceErrors() {
    assertProblems(
        List.of(withRun("[memory]\nscheme = \"BAD\"\n[thinning]\nenabled = 3\n")),
        "Invalid setting memory.scheme=BAD at run.toml:5: expected one of [NONE, EMA]",
        "Invalid setting thinning.enabled=3 at run.toml:7: expected one of [false, true]");
    assertProblems(
        List.of(withRun("[evaluation.tables.y]\nopponent = \"h\"\nwalls = 1\n[evaluation.tables.y.stop]\nevenMargin = 0.1\n")),
        "Missing setting evaluation.tables.y.stop.kind");
  }

  @Test
  public void keysOutsideTheChosenFormAreRejected() {
    assertProblems(
        List.of(withRun("[memory]\nscheme = \"NONE\"\nhalfLife = 3.0\n")),
        "Setting memory.halfLife at run.toml:6 does not apply to memory.scheme=NONE");
    assertProblems(
        List.of(withRun("[thinning]\ntau = 3\n")), "Setting thinning.tau at run.toml:5 does not apply to thinning.enabled=false");
    assertProblems(
        List.of(minimal(), set("memory=NONE", "thinning.qqqqqqqq=1")),
        "Unknown setting thinning.qqqqqqqq at --set");
  }

  @Test
  public void recordChecksNameTheKeyAndWhereItWasWritten() {
    assertProblems(
        List.of(withRun("[fit]\nlrFloor = 0.5\n")),
        "Invalid setting fit.lrFloor=0.5 at run.toml:5: lrFloor 0.5 exceeds lrPeak 0.0012");
    assertProblems(
        List.of(minimal(), set("fit.lrFloor=0.002")),
        "Invalid setting fit.lrFloor=0.002 at --set: lrFloor 0.002 exceeds lrPeak 0.0012");
    assertProblems(
        List.of(minimal(), set("fit.lrPeak=0.00001")),
        "Invalid setting fit.lrFloor=0.00006 at default: lrFloor 6.0E-5 exceeds lrPeak 1.0E-5");
    assertProblems(
        List.of(minimal(), set("inference.readyBatches=1")),
        "Invalid setting inference.readyBatches=1 at --set: readyBatches 1 is below slots 2");
    List<String> devices = problems(List.of(minimal(), set("inference.devices=[gpu0, gpu0]")));
    assertEquals(
        devices,
        List.of(
            "Invalid setting inference.devices=[\"gpu0\", \"gpu0\"] at --set: Devices must be one cpu or distinct GPUs: "
                + "gpu0,gpu0"));
  }

  record Guarded(@Key(doc = "値", value = "1") int value) {
    Guarded {
      if (value == 13) {
        throw new IllegalArgumentException("13 is unlucky");
      }
    }
  }

  record Outer(@Key(doc = "検査のある値", change = Change.FIXED) Guarded guarded) {}

  record Checked(
      @Key(doc = "下限", value = "1", change = Change.FIXED) int low,
      @Key(doc = "上限", value = "2", change = Change.FIXED) int high) {
    Checked {
      if (low > high) {
        throw new IllegalArgumentException("low " + low + " exceeds high " + high);
      }
    }
  }

  @Test
  public void otherChecksNameTheSection() {
    SettingsException section =
        expectThrows(
            SettingsException.class,
            () -> Settings.bind(Outer.class, List.of(Layer.parse("g.toml", Layer.Source.FILE, "[guarded]\nvalue = 13\n"))));
    assertEquals(section.problems(), List.of("Invalid setting guarded at g.toml:1: 13 is unlucky"));
    SettingsException root =
        expectThrows(SettingsException.class, () -> Settings.bind(Checked.class, List.of(set("low=3"))));
    assertEquals(root.problems(), List.of("Invalid settings at --set: low 3 exceeds high 2"));
  }

  @Test
  public void nearNames() {
    assertProblems(
        List.of(
            withRun(
                """
                [fit]
                lrPeek = 0.1
                [colect.table]
                workers = 3
                [zzz]
                q = 1
                [evaluation.tables.y]
                opponent = "h"
                wals = 2
                walls = 1
                stop = "FIXED"
                """)),
        "Unknown setting fit.lrPeek at run.toml:5 (did you mean fit.lrPeak?)",
        "Unknown setting evaluation.tables.y.wals at run.toml:12 (did you mean evaluation.tables.y.walls?)",
        "Unknown setting colect.table.workers at run.toml:7 (did you mean table.workers?)",
        "Unknown setting zzz.q at run.toml:9");
    assertProblems(
        List.of(withRun("[fit]\ntemperature = {PON = 0.1}\n")),
        "Unknown setting fit.temperature.PON at run.toml:5 (expected a key in [DISCARD, CALL, KAN])");
    assertProblems(List.of(minimal(), set("table.worker=2")), "Unknown setting table.worker at --set (did you mean table.workers?)");
    assertEquals(Keys.similar("a.table.workers", List.of("workers", "table.workers")), "table.workers");
    assertNull(Keys.similar("abcdefgh", List.of("zyxwvuts")));
    assertEquals(Keys.similar("fit.lrPek", List.of("fit.lrPeak", "fit.lrFloor")), "fit.lrPeak");
  }

  @Test
  public void allErrorsAreCollected() {
    Layer bad =
        file(
            """
            [run]
            variant = "FIVE"
            seed = 1
            [colect.table]
            workers = 3
            [fit]
            lrPeek = 0.1
            temperature = {CALL = 1.5, PON = 0.1}
            [memory]
            scheme = "NONE"
            halfLife = 3.0
            [inference]
            slots = 4
            readyBatches = 2
            [evaluation.tables.y]
            opponent = "heuristic"
            """);
    List<String> expected =
        List.of(
            "Setting riichi.token at --set is secret and is read only from the settings file",
            "Invalid setting run.variant=FIVE at run.toml:2: expected one of [FOUR, THREE]",
            "Invalid setting fit.temperature.CALL=1.5 at run.toml:8: expected a number in [0.0, 1.0]",
            "Unknown setting fit.temperature.PON at run.toml:8 (expected a key in [DISCARD, CALL, KAN])",
            "Unknown setting fit.lrPeek at run.toml:7 (did you mean fit.lrPeak?)",
            "Setting memory.halfLife at run.toml:11 does not apply to memory.scheme=NONE",
            "Invalid setting cohorts[0].from=0 at --set: expected an integer >= 1",
            "Missing setting cohorts[0].games",
            "Invalid setting inference.readyBatches=2 at run.toml:14: readyBatches 2 is below slots 4",
            "Invalid setting table.workers=0 at --set: expected an integer >= 1",
            "Missing setting evaluation.tables.y.walls",
            "Missing setting evaluation.tables.y.stop",
            "Unknown setting colect.table.workers at run.toml:5 (did you mean table.workers?)");
    SettingsException thrown =
        expectThrows(
            SettingsException.class,
            () -> bind(bad, set("riichi.token=x", "table.workers=0", "cohorts=[{from = 0}]")));
    assertEquals(thrown.problems(), expected);
    assertEquals(thrown.getMessage(), String.join("\n", expected));
  }

  @Test
  public void ofBindsABuiltRecord() {
    Root root = bind(minimal(), set("run.name=built")).value();
    assertEquals(Settings.of(root).value(), root);
    Root broken =
        new Root(
            root.run(),
            root.fit(),
            root.memory(),
            root.thinning(),
            root.cohorts(),
            root.flags(),
            root.inference(),
            new TableSettings(0, 1, 0, 1),
            root.runtime(),
            root.evaluation(),
            root.riichi());
    SettingsException thrown = expectThrows(SettingsException.class, () -> Settings.of(broken));
    assertEquals(thrown.problems(), List.of("Invalid setting table.activeGames=0 at record: expected an integer >= 1"));
  }

  record AbsentPrimitive(@Key(doc = "x", presence = Presence.ABSENT, change = Change.FIXED) int a) {}

  record SecretNumber(@Key(doc = "x", value = "1", secret = true, change = Change.FIXED) int a) {}

  record RequiredWithDefault(@Key(doc = "x", value = "1", presence = Presence.REQUIRED, change = Change.FIXED) Integer a) {}

  record NoDefault(@Key(doc = "x", change = Change.FIXED) int a) {}

  record RangeOnText(@Key(doc = "x", value = "a", change = Change.FIXED) @Range(min = 0) String a) {}

  record EmptyRange(@Key(doc = "x", value = "1", change = Change.FIXED) @Range(min = 2, max = 1) int a) {}

  record Recursive(@Key(doc = "x", change = Change.FIXED) Recursive child) {}

  @Choice(key = "kind")
  sealed interface Uncased permits UncasedCase {}

  record UncasedCase() implements Uncased {}

  record HoldsUncased(@Key(doc = "x", value = "A", change = Change.FIXED) Uncased a) {}

  @Choice(key = "kind")
  sealed interface Clash {
    @Case("A")
    record A(@Key(doc = "x", value = "1") int kind) implements Clash {}
  }

  record HoldsClash(@Key(doc = "x", value = "A", change = Change.FIXED) Clash a) {}

  record IntKeys(@Key(doc = "x", change = Change.FIXED) Map<Integer, String> a) {}

  record Unsupported(@Key(doc = "x", value = "1", change = Change.FIXED) Object a) {}

  record BadDefault(@Key(doc = "x", value = "abc", change = Change.FIXED) int a) {}

  record OutOfRangeDefault(@Key(doc = "x", value = "0", change = Change.FIXED) @Range(min = 1) int a) {}

  record UnreadableDefault(@Key(doc = "x", value = "[1, 2", change = Change.FIXED) List<Integer> a) {}

  record RawTree(@Key(doc = "x", presence = Presence.ABSENT, change = Change.FIXED) Node.Table a) {}

  record SecretItem(@Key(doc = "x", presence = Presence.ABSENT, secret = true) String token) {}

  record SecretInList(@Key(doc = "x", change = Change.FIXED) List<SecretItem> items) {}

  record InheritRoot(@Key(doc = "x", value = "1") int a) {}

  record Unannotated(int a) {}

  @Test
  public void unusableShapesFailWhenTheTypeIsRead() {
    Object[][] cases = {
      {AbsentPrimitive.class, "may be absent, so it needs a reference type"},
      {SecretNumber.class, "must be a string"},
      {RequiredWithDefault.class, "has a default but presence REQUIRED"},
      {NoDefault.class, "needs a default, or presence REQUIRED or ABSENT"},
      {RangeOnText.class, "@Range on a value that is not a number"},
      {EmptyRange.class, "Empty @Range"},
      {Recursive.class, "Recursive settings type"},
      {HoldsUncased.class, "must be a record with @Case"},
      {HoldsClash.class, "has a component named kind"},
      {IntKeys.class, "Table keys must be strings or enums"},
      {Unsupported.class, "Unsupported settings type"},
      {BadDefault.class, "Invalid default of BadDefault.a"},
      {OutOfRangeDefault.class, "Invalid default of OutOfRangeDefault.a"},
      {UnreadableDefault.class, "Unreadable default of UnreadableDefault.a"},
      {Unannotated.class, "has no @Key"},
    };
    for (Object[] shape : cases) {
      @SuppressWarnings("unchecked")
      Class<? extends Record> type = (Class<? extends Record>) shape[0];
      IllegalStateException thrown = expectThrows(IllegalStateException.class, () -> Settings.describe(type));
      assertTrue(thrown.getMessage().contains((String) shape[1]), type.getSimpleName() + ": " + thrown.getMessage());
    }
    IllegalStateException raw =
        expectThrows(IllegalStateException.class, () -> Settings.bind(RawTree.class, List.of(set("a={b = 1}"))));
    assertTrue(raw.getMessage().contains("holds a raw tree"), raw.getMessage());
    IllegalStateException secret =
        expectThrows(IllegalStateException.class, () -> Settings.bind(SecretInList.class, List.of()));
    assertTrue(secret.getMessage().contains("sits in a list or table"), secret.getMessage());
    IllegalStateException inherit =
        expectThrows(IllegalStateException.class, () -> Settings.bind(InheritRoot.class, List.of()));
    assertEquals(inherit.getMessage(), "Top-level setting a must say when it may change");
    expectThrows(IllegalStateException.class, () -> Settings.template(InheritRoot.class, List.of(), List.of()));
    expectThrows(IllegalStateException.class, () -> Settings.bind(Unannotated.class, List.of()));
  }
}
