package com.epsilon.agent.settings;

import static com.epsilon.agent.settings.Fixtures.BASE;
import static com.epsilon.agent.settings.Fixtures.PRESET;
import static com.epsilon.agent.settings.Fixtures.VARIANT;
import static com.epsilon.agent.settings.Fixtures.assertProblems;
import static com.epsilon.agent.settings.Fixtures.bind;
import static com.epsilon.agent.settings.Fixtures.file;
import static com.epsilon.agent.settings.Fixtures.minimal;
import static com.epsilon.agent.settings.Fixtures.number;
import static com.epsilon.agent.settings.Fixtures.set;
import static com.epsilon.agent.settings.Fixtures.text;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.settings.Fixtures.Cohort;
import com.epsilon.agent.settings.Fixtures.Kind;
import com.epsilon.agent.settings.Fixtures.Memory;
import com.epsilon.agent.settings.Fixtures.Riichi;
import com.epsilon.agent.settings.Fixtures.Root;
import com.epsilon.agent.settings.Fixtures.Stop;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.stream.Collectors;
import org.testng.annotations.Test;

/** 層の重ね方。表はキーごとに、列は丸ごと上書きする。ほかに、値を書いた場所、秘密のキー、2 回目の結び付けで入るルールの層、設定の行と変更、ひな形を確かめる。 */
public class LayeringTest {

  private static final String FILE =
      """
      [run]
      variant = "FOUR"
      seed = 7
      [table]
      writers = 2
      [evaluation.tables.smoke]
      walls = 96
      [memory]
      from = 2
      [[cohorts]]
      from = 5
      games = 10
      """;

  private static Settings<Root> layered() {
    return Settings.bind(
        Root.class, List.of(PRESET, file(FILE), set("fit.temperature.CALL=0.05", "table.activeGames=16")), BASE);
  }

  @Test
  public void tablesMergeByKeyAndListsReplaceWhole() {
    Root root = layered().value();
    assertEquals(root.fit().lrPeak(), 0.002);
    assertEquals(root.fit().temperature(), Map.of(Kind.DISCARD, 0.01, Kind.CALL, 0.05, Kind.KAN, 0.02));
    assertEquals(root.table(), new TableSettings(16, 3, 100000, 2));
    assertEquals(root.evaluation().tables().get("smoke"), new Fixtures.Table("heuristic", 96, new Stop.Fixed(), null));
    assertEquals(root.memory(), new Memory.Ema(8.0, 2));
    assertEquals(root.cohorts(), List.of(new Cohort(5, 10)));
  }

  @Test
  public void everyValueKeepsWhereItWasWritten() {
    Settings<Root> settings = layered();
    SortedMap<String, Node> effective = settings.effective();
    Map<String, String> origins =
        Map.ofEntries(
            Map.entry("table.workers", "preset smoke:2"),
            Map.entry("table.writers", "run.toml:5"),
            Map.entry("table.activeGames", "--set"),
            Map.entry("table.batchWaitMicros", "default"),
            Map.entry("fit.lrPeak", "variant FOUR:2"),
            Map.entry("fit.temperature.CALL", "--set"),
            Map.entry("fit.temperature.KAN", "default"),
            Map.entry("evaluation.tables.smoke.walls", "run.toml:7"),
            Map.entry("evaluation.tables.smoke.opponent", "preset smoke:4"),
            Map.entry("evaluation.tables.smoke.stop.kind", "preset smoke:6"),
            Map.entry("memory.scheme", "default"),
            Map.entry("memory.from", "run.toml:9"),
            Map.entry("cohorts", "run.toml:10"),
            Map.entry("run.seed", "run.toml:3"));
    origins.forEach((key, origin) -> assertEquals(effective.get(key).origin(), origin, key));
    String toml = settings.toml();
    assertTrue(toml.contains("workers = 3  # preset smoke\n"), toml);
    assertTrue(toml.contains("lrPeak = 0.002  # variant FOUR\n"), toml);
    assertTrue(toml.contains("activeGames = 16  # --set\n"), toml);
    assertTrue(toml.contains("seed = 7  # run.toml\n"), toml);
    assertFalse(toml.contains("# default"), toml);
    // キーとセクションの見出しの上には、ひな形と同じ説明を書く。型と、セクションと同じ変えてよい時期は書かない
    assertTrue(toml.contains("\n# 記憶（次の反復の当てはめから効く）\n[memory]\n# 記憶の方式（NONE・EMA）\nscheme = \"EMA\"\n"), toml);
    assertTrue(toml.contains("\n# 半減期（> 0.0）\nhalfLife = 8.0\n"), toml);
    assertTrue(toml.contains("\n# 判断のクラスごとの温度\n[fit.temperature]\n"), toml);
    assertTrue(toml.contains("\n# 卓（次に公開するバージョンの仕事から効く）\n[evaluation.tables.smoke]\n"), toml);
    assertTrue(toml.contains("\n# 止め方（FIXED・TEST）\nkind = \"FIXED\"  # preset smoke\n"), toml);
    // 空のテーブルのキーにも説明を書く
    String bare = bind(minimal()).toml();
    assertTrue(bare.contains("\n# 評価（次に公開するバージョンの仕事から効く）\n[evaluation]\n# 卓\ntables = {}\n"), bare);
    assertFalse(effective.containsKey("evaluation.tables.smoke.checkpoint"));
    assertFalse(effective.containsKey("riichi.token"));
  }

  @Test
  public void anUpperLayerThatChangesTheKindReplacesTheChoice() {
    Layer ema = Layer.parse("preset ema", Layer.Source.BUNDLED, "[memory]\nscheme = \"EMA\"\nhalfLife = 4.0\n");
    Layer halfLife = Layer.parse("preset half", Layer.Source.BUNDLED, "[memory]\nhalfLife = 4.0\n");
    assertEquals(bind(ema, minimal(), set("memory=NONE")).value().memory(), new Memory.None());
    assertEquals(
        bind(halfLife, file("[run]\nvariant = \"FOUR\"\nseed = 1\n[memory]\nscheme = \"NONE\"\n")).value().memory(),
        new Memory.None());
    assertEquals(bind(halfLife, minimal(), set("memory.scheme=EMA")).value().memory(), new Memory.Ema(4.0, 16));
    assertEquals(bind(ema, minimal(), set("memory.from=3")).value().memory(), new Memory.Ema(4.0, 3));
    assertEquals(bind(minimal(), set("run.seed=9")).value().run().seed(), 9L);
  }

  @Test
  public void secretsAreMaskedAndReadOnlyFromTheSettingsFile() {
    Settings<Root> settings = bind(file("[run]\nvariant = \"FOUR\"\nseed = 1\n[riichi]\ntoken = \"abc\"\n"));
    assertEquals(settings.value().riichi().token(), "abc");
    assertEquals(settings.effective().get("riichi.token"), new Node.Text(Binder.MASK, "x"));
    assertEquals(settings.effective().get("riichi.token").origin(), "run.toml:5");
    for (String written : List.of(settings.toml(), settings.json(), Toml.write(settings.tree()))) {
      assertFalse(written.contains("abc"), written);
      assertTrue(written.contains(Binder.MASK), written);
    }
    assertProblems(
        List.of(minimal(), set("riichi.token=x")),
        "Setting riichi.token at --set is secret and is read only from the settings file");
    assertProblems(
        List.of(Layer.parse("preset leak", Layer.Source.BUNDLED, "[riichi]\ntoken = \"x\"\n"), minimal()),
        "Setting riichi.token at preset leak:2 is secret and is read only from the settings file");
    assertProblems(
        List.of(minimal(), Layer.tree("settings row 1", Json.read(settings.json(), "row"))),
        "Setting riichi.token at settings row 1 is secret and is read only from the settings file");
    Settings<Root> built = Settings.of(settings.value());
    assertEquals(built.value().riichi(), new Riichi("abc"));
    assertFalse(built.json().contains("abc"), built.json());
    SettingsException section =
        expectThrows(
            SettingsException.class, () -> Settings.section(Riichi.class, "riichi", List.of(set("riichi.token=x"))));
    assertEquals(
        section.problems(), List.of("Setting riichi.token at --set is secret and is read only from the settings file"));
    String template =
        Settings.template(Root.class, List.of(file("[riichi]\ntoken = \"abc\"\n")), List.of());
    assertFalse(template.contains("abc"), template);
    assertTrue(template.contains("\ntoken = \"\"\n"), template);
  }

  @Test
  public void theVariantLayerComesInOnTheSecondBinding() {
    List<Layer> stacked = Settings.withBase(Root.class, List.of(minimal()), BASE);
    assertSame(stacked.getFirst(), VARIANT);
    assertEquals(stacked.size(), 2);
    assertEquals(Settings.bind(Root.class, List.of(minimal()), BASE).value().fit().lrPeak(), 0.002);
    assertEquals(
        Settings.bind(Root.class, List.of(minimal(), set("fit.lrPeak=0.003")), BASE).value().fit().lrPeak(), 0.003);
    assertEquals(
        Settings.bind(Root.class, List.of(file("[run]\nvariant = \"THREE\"\nseed = 1\n"), set("run.variant=FOUR")), BASE)
            .value()
            .fit()
            .lrPeak(),
        0.002);
    SettingsException missing =
        expectThrows(SettingsException.class, () -> Settings.bind(Root.class, List.of(set("run.seed=1")), BASE));
    assertEquals(missing.problems(), List.of("Missing setting run.variant"));
    SettingsException invalid =
        expectThrows(SettingsException.class, () -> Settings.bind(Root.class, List.of(set("run.variant=SIX")), BASE));
    assertEquals(invalid.problems(), List.of("Invalid setting run.variant=SIX at --set: expected one of [FOUR, THREE]"));
    SettingsException unknown =
        expectThrows(
            SettingsException.class,
            () -> Settings.bind(Root.class, List.of(set("run.variant=THREE", "run.seed=1")), BASE));
    assertEquals(
        unknown.problems(), List.of("Invalid setting run.variant=THREE at --set: no variant layer for THREE"));
    expectThrows(
        IllegalStateException.class,
        () -> Settings.bind(Root.class, List.of(minimal()), new BaseLayer("run", name -> VARIANT)));
  }

  @Test
  public void settingsRowsBindBackAndChangesAreClassified() {
    Settings<Root> start = bind(file("[run]\nvariant = \"FOUR\"\nseed = 20261110\n"));
    assertEquals(Toml.read(Toml.write(start.tree()), "again"), start.tree());
    Node.Table row = Json.read(start.json(), "row");
    assertEquals(row, start.tree());
    Settings<Root> again = bind(Layer.tree("settings row 1", row));
    assertEquals(again.value(), start.value());
    assertEquals(again.effective().get("table.workers").origin(), "settings row 1");
    assertEquals(again.since(start), List.of());

    Settings<Root> changed =
        bind(
            Layer.tree("settings row 1", row),
            Layer.assignments("--change", List.of("memory=NONE", "run.seed=20261111", "table.workers=4")));
    assertEquals(changed.value().memory(), new Memory.None());
    List<Difference> differences = changed.since(again);
    assertEquals(
        differences,
        List.of(
            new Difference("memory.from", number("16"), null, Change.FIT),
            new Difference("memory.halfLife", number("8.0"), null, Change.FIT),
            new Difference("memory.scheme", text("EMA"), text("NONE"), Change.FIT),
            new Difference("run.seed", number("20261110"), number("20261111"), Change.FIXED),
            new Difference("table.workers", number("28"), number("4"), Change.MACHINE)));
    SettingsException fixed = expectThrows(SettingsException.class, () -> Difference.rejectFixed(differences));
    assertEquals(
        fixed.problems(),
        List.of("Setting run.seed is fixed for the run: the run started with 20261110, the change asks for 20261111"));
    Difference.rejectFixed(differences.stream().filter(difference -> difference.change() != Change.FIXED).toList());
    SettingsException edges =
        expectThrows(
            SettingsException.class,
            () ->
                Difference.rejectFixed(
                    List.of(
                        new Difference("a", null, number("1"), Change.FIXED),
                        new Difference("b", text("x"), null, Change.FIXED))));
    assertEquals(
        edges.problems(),
        List.of(
            "Setting a is fixed for the run: the run started without it, the change asks for 1",
            "Setting b is fixed for the run: the run started with x, the change removes it"));
    expectThrows(IllegalArgumentException.class, () -> new Difference("c", null, null, Change.FIT));
  }

  @Test
  public void stringsCompareByTheirTextHoweverTheyWereWritten() {
    Settings<Root> start = bind(file("[run]\nvariant = \"FOUR\"\nseed = 1\nname = \"8\"\ndirectory = \"12\"\n"));
    Layer row = Layer.tree("settings row 1", Json.read(start.json(), "row"));
    Settings<Root> same = bind(row, Layer.assignments("--change", List.of("run.name=8", "run.directory=12")));
    assertEquals(same.value(), start.value());
    assertEquals(same.since(start), List.of());
    assertEquals(same.effective().get("run.name"), text("8"));
    assertEquals(same.effective().get("run.name").origin(), "--change");
    // 数としては等しい 8 と 8.0 も、文字列としては違う
    Settings<Root> longer = bind(row, Layer.assignments("--change", List.of("run.name=8.0")));
    assertEquals(
        longer.since(same), List.of(new Difference("run.name", text("8"), text("8.0"), Change.FIXED)));
    Settings<Root> yes = bind(minimal(), set("run.name=true"));
    assertEquals(bind(minimal(), set("run.name=\"true\"")).since(yes), List.of());
  }

  @Test
  public void aCommandSectionBindsOnItsOwn() {
    TableSettings table =
        Settings.section(TableSettings.class, "table", List.of(PRESET, minimal(), set("table.writers=1"))).value();
    assertEquals(table, new TableSettings(8192, 3, 100000, 1));
    Settings<TableSettings> settings = Settings.section(TableSettings.class, "table", List.of());
    assertEquals(settings.change("table.workers"), Change.MACHINE);
    SettingsException invalid =
        expectThrows(
            SettingsException.class,
            () -> Settings.section(TableSettings.class, "table", List.of(file("[table]\nworkers = 0\nworkerz = 1\n"))));
    assertEquals(
        invalid.problems(),
        List.of(
            "Invalid setting table.workers=0 at run.toml:2: expected an integer >= 1",
            "Unknown setting table.workerz at run.toml:3 (did you mean table.workers?)"));
    SettingsException scalar =
        expectThrows(
            SettingsException.class, () -> Settings.section(TableSettings.class, "table", List.of(file("table = 3\n"))));
    assertEquals(scalar.problems(), List.of("Invalid setting table=3 at run.toml:1: expected a table"));
  }

  @Test
  public void describeListsEveryKey() {
    Map<String, KeyInfo> keys =
        Settings.describe(Root.class).stream()
            .collect(Collectors.toMap(key -> key.path() + "|" + key.condition(), key -> key));
    assertEquals(keys.size(), Settings.describe(Root.class).size());
    KeyInfo seed = keys.get("run.seed|");
    assertEquals(seed.presence(), Presence.REQUIRED);
    assertEquals(seed.change(), Change.FIXED);
    assertEquals(seed.type(), "integer");
    KeyInfo scheme = keys.get("memory.scheme|");
    assertEquals(scheme.value(), "EMA");
    assertEquals(scheme.type(), "NONE|EMA");
    assertEquals(scheme.change(), Change.FIT);
    KeyInfo halfLife = keys.get("memory.halfLife|memory.scheme=EMA");
    assertEquals(halfLife.range(), "> 0.0");
    assertEquals(halfLife.value(), "8.0");
    assertEquals(halfLife.change(), Change.FIT);
    assertTrue(keys.containsKey("thinning.tau|thinning.enabled=true"));
    assertEquals(keys.get("fit.temperature|").type(), "number per Kind");
    assertEquals(keys.get("fit.classes|").type(), "distinct DISCARD|CALL|KAN list");
    assertEquals(keys.get("cohorts|").type(), "table list");
    assertEquals(keys.get("evaluation.tables|").type(), "table per name");
    KeyInfo opponent = keys.get("evaluation.tables.<name>.opponent|");
    assertEquals(opponent.presence(), Presence.REQUIRED);
    assertEquals(opponent.change(), Change.SCHEDULE);
    assertTrue(keys.containsKey("evaluation.tables.<name>.stop.alpha|evaluation.tables.<name>.stop.kind=TEST"));
    assertTrue(keys.get("riichi.token|").secret());
    assertEquals(keys.get("inference.slots|").range(), "[1, 4]");
    assertEquals(keys.get("inference.precision|").change(), Change.COLLECT);
    assertEquals(keys.get("table.batchWaitMicros|").range(), ">= 0");
  }

  @Test
  public void templateCommentsWhatTheUserDidNotWrite() {
    List<Layer> layers = Settings.withBase(Root.class, List.of(PRESET, set("run.variant=FOUR")), BASE);
    String template = Settings.template(Root.class, layers, List.of("smoke"));
    Set<String> lines = new HashSet<>(Arrays.asList(template.split("\n")));
    assertTrue(template.startsWith("presets = [\"smoke\"]\n"), template);
    assertTrue(lines.contains("variant = \"FOUR\""), template);
    assertTrue(lines.contains("# seed ="), template);
    assertTrue(lines.contains("# lrPeak = 0.002  # variant FOUR:2"), template);
    assertTrue(lines.contains("# workers = 3  # preset smoke:2"), template);
    assertTrue(lines.contains("# activeGames = 8192"), template);
    assertTrue(lines.contains("# [evaluation.tables.smoke]"), template);
    assertTrue(lines.contains("# [memory]"), template);
    assertTrue(lines.contains("[table]"), template);
    assertTrue(lines.contains("# 乱数の種（必須）"), template);
    assertTrue(lines.contains("# 記憶の方式（NONE・EMA）"), template);
    assertTrue(lines.contains("# 半減期（> 0.0）"), template);
    assertTrue(lines.contains("# 相手のチェックポイント（省略できる）"), template);
    Layer reread = Layer.parse("template.toml", Layer.Source.FILE, template);
    assertEquals(reread.presets(), List.of("smoke"));
    SettingsException missing =
        expectThrows(SettingsException.class, () -> Settings.bind(Root.class, List.of(PRESET, reread), BASE));
    assertEquals(missing.problems(), List.of("Missing setting run.seed"));
    String empty = Settings.template(Root.class, List.of(), List.of());
    assertTrue(empty.contains("# [evaluation.tables.<名前>]\n"), empty);
    assertFalse(empty.startsWith(Layer.PRESETS), empty);
  }

  @Test
  public void aTemplateBindsToWhatItWasMadeFrom() {
    Layer written =
        file(
            """
            [run]
            variant = "FOUR"
            seed = 7
            [fit]
            temperature = {CALL = 0.3}
            [memory]
            halfLife = 2
            [thinning]
            enabled = true
            [[cohorts]]
            from = 1
            games = 64
            [flags]
            labels = {q = "r"}
            [table]
            writers = 1
            [evaluation.tables.x1n3]
            opponent = "nano:/x"
            walls = 4
            stop = {kind = "TEST", evenMargin = 0.02}
            checkpoint = "/c"
            """);
    List<Layer> layers = List.of(PRESET, written, set("inference.slots=3", "memory.from=4"));
    Root expected = Settings.bind(Root.class, layers, BASE).value();
    String template = Settings.template(Root.class, Settings.withBase(Root.class, layers, BASE), List.of("smoke"));
    Layer reread = Layer.parse("template.toml", Layer.Source.FILE, template);
    assertEquals(Settings.bind(Root.class, List.of(PRESET, reread), BASE).value(), expected, template);
  }

  @Test
  public void layersFromFilesAndResources() throws Exception {
    Path path = Files.createTempFile("settings", ".toml");
    try {
      Files.writeString(path, "presets = [\"production\", \"smoke\"]\n[run]\nseed = 3\n", StandardCharsets.UTF_8);
      Layer layer = Layer.toml(path);
      assertEquals(layer.source(), Layer.Source.FILE);
      assertEquals(layer.presets(), List.of("production", "smoke"));
      assertNull(layer.tree().get(Layer.PRESETS));
      assertEquals(((Node.Table) layer.tree().get("run")).get("seed").origin(), path + ":3");
    } finally {
      Files.delete(path);
    }
    SettingsException presets =
        expectThrows(SettingsException.class, () -> Layer.parse("run.toml", Layer.Source.FILE, "presets = \"smoke\"\n"));
    assertEquals(presets.problems(), List.of("Setting presets at run.toml:1 must be a list of preset names"));
    Layer bundled = Layer.parse("preset x", Layer.Source.BUNDLED, "presets = [\"y\"]\n");
    assertEquals(bundled.presets(), List.of());
    assertEquals(bundled.tree().get(Layer.PRESETS), Fixtures.array(text("y")));
    expectThrows(IllegalStateException.class, () -> Layer.resource("no/such/settings.toml", "preset none"));
    assertEquals(minimal().without("run").tree(), Fixtures.table());
    assertEquals(set("a=1").origin(), "--set");
    assertEquals(set("a=1").source(), Layer.Source.COMMAND_LINE);
  }
}
