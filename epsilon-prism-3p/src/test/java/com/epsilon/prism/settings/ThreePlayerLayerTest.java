package com.epsilon.prism.settings;

import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.ThreePlayerSettings;
import com.epsilon.prism.core.ClassTables;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.grp.GrpOperations;
import com.epsilon.prism.grp.PhiSettings;
import com.epsilon.prism.variant.PrismVariant;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの層の設定のテスト。層の全てのキーが設定に反映され、どのプリセットの組を重ねても三人打ちの値が残る。同じ組なら、重ねる順に
 * 依らず同じ設定になる。北抜きのクラスの行が欠けた表、四人打ちの run、席の数の倍数でない評価の牌山の数は、
 * キーのパスを添えて拒む。
 */
public class ThreePlayerLayerTest {

  private static final Variant THREE = Variant.TENHOU_3P;

  /** 重ねるプリセットの組。production・smoke・三人打ちの系譜から選んだ部分集合を、全ての順に並べたもの。 */
  private static final List<List<String>> PRESETS =
      arrangements(List.of(), List.of("production", "smoke", ThreePlayerLineagePresetTest.PRESET));

  /** すでに並べたプリセットの後ろに、残りのプリセットの部分集合を並べる。全ての並べ方を返し、すでに並べた分だけの並びも含む。 */
  private static List<List<String>> arrangements(List<String> chosen, List<String> rest) {
    List<List<String>> all = new ArrayList<>();
    all.add(chosen);
    for (String next : rest) {
      List<String> longer = new ArrayList<>(chosen);
      longer.add(next);
      List<String> left = new ArrayList<>(rest);
      left.remove(next);
      all.addAll(arrangements(List.copyOf(longer), left));
    }
    return all;
  }

  @Test
  public void everyPresetCombinationKeepsTheThreePlayerValues() {
    Set<DecisionClass> classes = ClassTables.classes(THREE);
    Assert.assertEquals(classes.size(), 11);
    Assert.assertTrue(classes.containsAll(Set.of(DecisionClass.NUKI, DecisionClass.NUKI_RESPONSE)), classes.toString());
    Assert.assertEquals(PRESETS.size(), 1 + 3 + 6 + 6);
    // プリセットの集合ごとに最初に作った設定。重ねる順が違っても、同じ設定になる
    Map<Set<String>, PrismSettings> bySet = new LinkedHashMap<>();
    for (List<String> presets : PRESETS) {
      String label = presets.toString();
      PrismSettings settings = ThreePlayerSettings.bind(presets).value();
      PrismSettings first = bySet.putIfAbsent(new TreeSet<>(presets), settings);
      if (first != null) {
        Assert.assertEquals(settings, first, label);
      }
      Assert.assertEquals(settings.run().variant(), THREE, label);
      Assert.assertEquals(settings.module().variant(), THREE, label);
      boolean production = presets.contains("production");
      Assert.assertEquals(settings.inference().precision(), production ? Precision.BFLOAT16 : Precision.FLOAT32, label);
      Assert.assertEquals(settings.inference().kernels(), production ? Kernels.NATIVE : Kernels.EAGER, label);
      Assert.assertEquals(settings.learner().precision(), production ? Precision.BFLOAT16 : Precision.FLOAT32, label);
      boolean smoke = presets.contains("smoke");
      Assert.assertEquals(settings.schedule().iterations(), smoke ? 2 : 1000, label);
      // 三人打ちの系譜のプリセットは、評価の卓と、自前の GRP の運用の値を書く
      boolean lineage = presets.contains(ThreePlayerLineagePresetTest.PRESET);
      Set<String> tables = new TreeSet<>();
      if (smoke) {
        tables.add("smoke");
      }
      if (lineage) {
        tables.addAll(ThreePlayerLineagePresetTest.TABLES);
      }
      Assert.assertEquals(settings.evaluation().tables().keySet(), tables, label);
      Assert.assertEquals(settings.evaluation().anchor(), lineage ? "reference" : null, label);
      Assert.assertTrue(settings.phi() instanceof PhiSettings.SelfGrp, label + " " + settings.phi());
      GrpOperations grp = ((PhiSettings.SelfGrp) settings.phi()).grp();
      Assert.assertEquals(grp.halfLife(), lineage ? 1.0 : 2.0, label);
      Assert.assertEquals(grp.trustGames(), lineage ? 8192 : 16384, label);
      Assert.assertEquals(settings.network(), new NetworkSettings.V2(2), label);
      // クラスの表は三人打ちの 11 クラスを持つ
      Assert.assertEquals(settings.labels().rollin().keySet(), classes, label);
      Assert.assertEquals(settings.fit().temperature().keySet(), classes, label);
      Assert.assertEquals(settings.noise().rates().table().keySet(), classes, label);
      // smoke は北抜きの行を書かないので、北抜きの率は三人打ちの層の値のまま
      for (DecisionClass nuki : List.of(DecisionClass.NUKI, DecisionClass.NUKI_RESPONSE)) {
        Assert.assertEquals(settings.labels().rollin().get(nuki), 0.03, label);
        Assert.assertEquals(settings.fit().temperature().get(nuki), 0.02, label);
        Assert.assertEquals(settings.noise().rates().table().get(nuki), 0.104, label);
      }
      Assert.assertEquals(settings.labels().rollin().get(DecisionClass.CALL), smoke ? 0.015 : 0.03, label);
    }
    Assert.assertThrows(SettingsException.class, () -> ThreePlayerSettings.bind(List.of("no-such-preset")));
  }

  /** 三人打ちの層の値は全て効いていて、出どころは三人打ちの層になる。設定の行に書いた木を読み直すと、同じ設定に戻る。 */
  @Test
  public void everyKeyOfTheLayerBinds() {
    Map<String, Node> written = new TreeMap<>();
    flatten(PrismVariant.of(THREE).settings().tree(), "", written);
    Assert.assertEquals(written.size(), 3 * 11, written.keySet().toString());
    Settings<PrismSettings> bound = ThreePlayerSettings.bind(List.of());
    Map<String, Node> effective = bound.effective();
    for (Map.Entry<String, Node> entry : written.entrySet()) {
      Node node = effective.get(entry.getKey());
      Assert.assertNotNull(node, entry.getKey());
      Assert.assertEquals(node, entry.getValue(), entry.getKey());
      Assert.assertTrue(node.origin().startsWith("variant " + THREE), entry.getKey() + " from " + node.origin());
    }
    SettingsRow row = new SettingsRow(1, 1, Instant.EPOCH, "test", SettingsRow.CREATED, List.of(), bound.tree());
    Assert.assertEquals(row.bind(PrismSettings.class).value(), bound.value());
  }

  @Test
  public void variantMismatchesAreRefused() {
    // 四人打ちのモジュールが無い classpath では、四人打ちの run を作れない
    SettingsException missing =
        Assert.expectThrows(SettingsException.class, () -> ThreePlayerSettings.bind(List.of(), "run.variant=TENHOU_4P"));
    Assert.assertTrue(
        missing.problems().toString().contains("No PRISM module for TENHOU_4P on the classpath"),
        missing.problems().toString());
    // クラスの表から北抜きの行を抜くと、欠けたクラスを名前で拒む
    Layer layer = PrismVariant.of(THREE).settings();
    for (String table : List.of("labels.rollin", "fit.temperature", "noise.rates.table")) {
      for (DecisionClass nuki : List.of(DecisionClass.NUKI, DecisionClass.NUKI_RESPONSE)) {
        Layer dropped = new Layer(layer.origin(), layer.source(), without(layer.tree(), table, nuki.name()), List.of());
        List<Layer> layers = new ArrayList<>();
        layers.add(dropped);
        layers.add(Layer.assignments("test run", List.of("run.variant=TENHOU_3P", "run.seed=1")));
        SettingsException refused =
            Assert.expectThrows(SettingsException.class, () -> Settings.bind(PrismSettings.class, layers));
        String problems = refused.problems().toString();
        Assert.assertTrue(problems.contains(table) && problems.contains("missing [" + nuki + "]"), problems);
      }
    }
    // 全ての手を測るクラスには北抜きのクラスも選べる
    Assert.assertEquals(
        ThreePlayerSettings.of("noise.rates.allActions=[NUKI]").noise().rates().allActions(), Set.of(DecisionClass.NUKI));
    // 評価の牌山の数は 3 の倍数
    String prefix = "evaluation.tables.t.";
    String[] table = {
      prefix + "opponent=heuristic", prefix + "seating=BY_GAME", prefix + "draws=SEAT_STREAMS", prefix + "wallSet=FIXED",
      prefix + "stop=FIXED", prefix + "score=RANK", prefix + "every=1", prefix + "from=1"
    };
    rejects(prefix + "walls", "multiple of 3", concat(table, prefix + "walls=8"));
    Assert.assertEquals(ThreePlayerSettings.of(concat(table, prefix + "walls=9")).evaluation().tables().get("t").walls(), 9);
  }

  private static void rejects(String path, String because, String... assignments) {
    SettingsException failure = Assert.expectThrows(SettingsException.class, () -> ThreePlayerSettings.of(assignments));
    boolean named = failure.problems().stream().anyMatch(problem -> problem.contains(path) && problem.contains(because));
    Assert.assertTrue(named, String.join(" ", assignments) + " -> " + failure.problems());
  }

  private static String[] concat(String[] first, String... second) {
    List<String> all = new ArrayList<>(List.of(first));
    all.addAll(List.of(second));
    return all.toArray(String[]::new);
  }

  /** 設定の木の葉を「キーのパス → 値」の表に集める。層のキーは全て引用符の要らない名前なので、点でつなぐだけでよい。 */
  private static void flatten(Node node, String path, Map<String, Node> leaves) {
    if (node instanceof Node.Table table) {
      table.entries().forEach((key, child) -> flatten(child, path.isEmpty() ? key : path + "." + key, leaves));
    } else {
      leaves.put(path, node);
    }
  }

  /**
   * パスが指す表から 1 つのキーを除いた木を作る。元の木は変えない。
   *
   * @param root 層の木
   * @param path 表のパス（点で区切る）
   * @param key 除くキー
   * @return キーを除いた木
   */
  private static Node.Table without(Node.Table root, String path, String key) {
    int dot = path.indexOf('.');
    String head = dot < 0 ? path : path.substring(0, dot);
    Node.Table child = (Node.Table) root.get(head);
    Node.Table changed = dot < 0 ? child.without(key) : without(child, path.substring(dot + 1), key);
    Assert.assertNotEquals(changed, child, path + "." + key);
    Map<String, Node> entries = new LinkedHashMap<>(root.entries());
    entries.put(head, changed);
    return new Node.Table(entries, root.origin());
  }
}
