package com.epsilon.prism.settings;

import com.epsilon.agent.eval.SeatingSettings;
import com.epsilon.agent.learn.GrpTraining;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.KeyInfo;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.app.PrismApp;
import com.epsilon.prism.core.ClassTables;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.grp.PhiSettings;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism.noise.LedgerSettings;
import com.epsilon.prism.noise.TempoSettings;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * PRISM の run の設定全体のテスト。jar にある全てのルールと全てのプリセットの組み合わせが読み込みを通る（jar に無いルールは拒む）。設定全体の
 * 検査（クラスの表、クラスのしきい値、専用のカーネルの計算の型、評価の卓とルール、Φ の組み合わせ、TEMPO と帳簿のモードの組み合わせ）は、
 * キーのパスを添えて拒む。ノイズの対策を 1 つずつ切った設定は、系譜が対策なしの経路を選ぶ値になる。
 */
public class PrismSettingsTest {

  /** プリセットの組み合わせ（重ねる順）。 */
  private static final List<List<String>> PRESETS =
      List.of(List.of(), List.of("production"), List.of("smoke"), List.of("production", "smoke"), List.of("smoke", "production"));

  @Test
  public void everyVariantAndPresetCombinationBinds() {
    int bound = 0;
    for (Variant variant : Variant.values()) {
      for (List<String> presets : PRESETS) {
        String label = variant + " " + presets;
        if (variant != Variant.TENHOU_4P) {
          SettingsException missing =
              Assert.expectThrows(SettingsException.class, () -> TestSettings.bind(presets, "run.variant=" + variant));
          Assert.assertTrue(
              missing.problems().toString().contains("No PRISM module for " + variant + " on the classpath"), label + ": "
                  + missing.problems());
          continue;
        }
        PrismSettings settings = TestSettings.bind(presets, "run.variant=" + variant).value();
        bound++;
        Assert.assertEquals(settings.run().variant(), variant, label);
        boolean production = presets.contains("production");
        Assert.assertEquals(settings.inference().precision(), production ? Precision.BFLOAT16 : Precision.FLOAT32, label);
        Assert.assertEquals(settings.inference().kernels(), production ? Kernels.NATIVE : Kernels.EAGER, label);
        Assert.assertEquals(settings.inference().maxBatch(), production ? 2048 : 8, label);
        Assert.assertEquals(settings.learner().precision(), production ? Precision.BFLOAT16 : Precision.FLOAT32, label);
        boolean smoke = presets.contains("smoke");
        Assert.assertEquals(settings.schedule().iterations(), smoke ? 2 : 1000, label);
        Assert.assertEquals(settings.fit().maxSteps(), smoke ? 8 : 2000, label);
        Assert.assertEquals(settings.evaluation().tables().keySet(), smoke ? Set.of("smoke") : Set.of(), label);
      }
    }
    Assert.assertEquals(bound, PRESETS.size());
    Assert.assertThrows(SettingsException.class, () -> TestSettings.bind(List.of("no-such-preset")));
  }

  /** 三人打ちの系譜のプリセットは四人打ちの jar にも入るが、四人打ちの run では席の分け方（3 席ぶん）を英語の理由で拒む。 */
  @Test
  public void theThreePlayerLineagePresetIsRefusedOnFourPlayerRuns() {
    for (List<String> presets :
        List.of(
            List.of("three-player-lineage"),
            List.of("production", "three-player-lineage"),
            List.of("three-player-lineage", "production"))) {
      SettingsException refused = Assert.expectThrows(SettingsException.class, () -> TestSettings.bind(presets));
      List<String> splits =
          refused.problems().stream()
              .filter(problem -> problem.contains("seating.splits") && problem.contains("TENHOU_4P has 4"))
              .toList();
      Assert.assertTrue(
          splits.stream().anyMatch(problem -> problem.contains("evaluation.tables.champion.seating.splits")),
          presets + " -> " + refused.problems());
      for (String problem : splits) {
        Assert.assertTrue(problem.contains("preset three-player-lineage"), problem);
        Assert.assertTrue(problem.chars().allMatch(c -> c < 0x80), "not English: " + problem);
      }
    }
  }

  @Test
  public void wholeConfigChecksNameTheKey() throws IOException {
    String table = "evaluation.tables.t.";
    String[] heuristic = TestSettings.table("t", "heuristic", 8, 1, 1);
    // 1 つずつ確かめる: 拒まれる上書きと、エラーメッセージに入るキーのパスと理由
    // 補正の R² でゲインを上げる TEMPO には、補正する帳簿のモードが要る
    rejects("noise.tempo.noiseGain", "needs ledger mode APPLY", "noise.ledger.mode=RECORD");
    rejects(
        "inference.precision",
        "native kernels compute in BFLOAT16",
        "inference.devices=[\"gpu0\"]",
        "inference.kernels=NATIVE",
        "inference.precision=FLOAT32");
    rejects(table + "walls", "multiple of 4", TestSettings.concat(heuristic, table + "walls=6"));
    // 系譜の卓は、PRISM のチェックポイントや系譜のバージョンも相手にできる。相手が version:N の卓だけは、昇格したバージョンも測るかを書く
    for (String opponent : List.of("prism:/tmp/x", "champion", "version:1")) {
      String[] written = TestSettings.concat(heuristic, table + "opponent=" + opponent, table + "from=2");
      if (opponent.startsWith("version:")) {
        written = TestSettings.concat(written, table + "onPromotion=false");
      }
      Assert.assertEquals(TestSettings.of(written).evaluation().tables().get("t").opponent(), opponent);
    }
    // 全ての席の割り当てを打つ卓は、分け方（1v3 など）の席の数の和が卓の席の数と同じときだけ通る。1 半荘ずつの卓は、計画した牌山を全て打つときだけ通る
    String[] placements =
        TestSettings.concat(heuristic, table + "seating={kind = \"EVERY_PLACEMENT\", splits = [\"1v3\", \"3v1\"]}");
    Assert.assertEquals(
        TestSettings.of(placements).evaluation().tables().get("t").seating(),
        new SeatingSettings.EveryPlacement(List.of("1v3", "3v1")));
    rejects(
        table + "seating.splits",
        "seats 3",
        TestSettings.concat(heuristic, table + "seating={kind = \"EVERY_PLACEMENT\", splits = [\"1v2\"]}"));
    rejects(
        table + "stop",
        "stops FIXED",
        TestSettings.concat(
            heuristic,
            table + "walls=1024",
            table + "stop={kind = \"PRECISION\", error = 0.01, alpha = 0.01, checkEvery = 512}"));
    rejects("evaluation.anchor", "anchor", TestSettings.concat(heuristic, "evaluation.anchor=t"));
    // Φ の組み合わせ: 自前の GRP の運用のキーは自前の GRP の設定でしか使えず、固定の GRP には GRP が要る
    rejects("phi.grp", "", "phi.source=SCORE_RANK", "phi.grp.monitor=SCORE");
    rejects("phi.grp", "", "phi.source=FIXED_GRP");
    // 正しい卓は通る
    Assert.assertEquals(TestSettings.of(heuristic).evaluation().tables().get("t").walls(), 8);

    // クラスの表は、ルールのクラスをちょうど 1 回ずつ持つ（ルールの層から 1 行を抜くと、足りないクラスを名前で拒む）
    String layer = variantLayer();
    String dropped = layer.replaceFirst("(?m)^CALL = 0\\.03\\r?\\n", "");
    Assert.assertNotEquals(dropped, layer);
    List<Layer> layers = new ArrayList<>();
    layers.add(Layer.parse("variant without CALL", Layer.Source.BUNDLED, dropped));
    layers.add(Layer.assignments("test run", List.of("run.variant=TENHOU_4P", "run.seed=1")));
    SettingsException missing =
        Assert.expectThrows(SettingsException.class, () -> Settings.bind(PrismSettings.class, layers));
    String problems = missing.problems().toString();
    Assert.assertTrue(problems.contains("labels.rollin") && problems.contains("missing [CALL]"), problems);
    // 全ての手を測るクラスは、ルールのクラスから選ぶ
    Assert.assertEquals(TestSettings.of("noise.rates.allActions=[CALL]").noise().rates().allActions(), Set.of(DecisionClass.CALL));
  }

  /** PRISM の王者の決闘は良いだけで昇格させ、確かめの卓を打たないので、確かめの卓を書いた設定は拒む。 */
  @Test
  public void aConfirmTableIsRefusedBecauseThePrismDuelPlaysNone() {
    String champion = "evaluation.tables.champion.";
    String outside = "evaluation.tables.outside.";
    String[] duel = {
      champion + "opponent=champion",
      champion + "seating={kind = \"EVERY_PLACEMENT\", splits = [\"1v3\", \"3v1\"]}",
      champion + "draws=SEAT_STREAMS",
      champion + "wallSet=FRESH",
      champion + "walls=1024",
      champion
          + "stop={kind = \"TEST\", evenMargin = 0.05, alpha = 0.05, spending = 1.0, checkEvery = 512, promote = true}",
      champion + "score=RANK",
      champion + "every=1",
      champion + "from=2"
    };
    Assert.assertEquals(TestSettings.of(duel).evaluation().promoting().orElseThrow(), "champion");
    rejects(
        champion + "confirm",
        "plays no confirm table",
        TestSettings.concat(
            duel,
            champion + "confirm=outside",
            outside + "opponent=prism:/tmp/x",
            outside + "seating={kind = \"EVERY_PLACEMENT\", splits = [\"1v3\"]}",
            outside + "draws=SEAT_STREAMS",
            outside + "wallSet=FIXED",
            outside + "walls=8",
            outside + "stop=FIXED",
            outside + "score=RANK",
            outside + "confirmMargin=0.01",
            outside + "every=0",
            outside + "from=0"));
  }

  /**
   * 四人打ちの層は 9 クラスの表と 50 枚・25 枚のしきい値のまま。北抜きのクラスの行は四人打ちの表に書けず、全ての手を測るクラスにも選べない。
   */
  @Test
  public void theFourPlayerLayerKeepsItsClassesAndValues() {
    Set<DecisionClass> classes = ClassTables.classes(Variant.TENHOU_4P);
    Assert.assertEquals(classes.size(), 9);
    Assert.assertFalse(classes.contains(DecisionClass.NUKI) || classes.contains(DecisionClass.NUKI_RESPONSE), classes.toString());
    PrismSettings settings = TestSettings.of();
    Map<DecisionClass, Double> rollin = new EnumMap<>(DecisionClass.class);
    Map<DecisionClass, Double> temperature = new EnumMap<>(DecisionClass.class);
    Map<DecisionClass, Double> rates = new EnumMap<>(DecisionClass.class);
    for (DecisionClass decision : classes) {
      boolean discard = decision.name().startsWith("DISCARD_");
      rollin.put(decision, discard ? 0.002 : 0.03);
      temperature.put(decision, discard ? 0.01 : 0.02);
      rates.put(decision, 0.104);
    }
    rates.putAll(
        Map.of(
            DecisionClass.DISCARD_EARLY, 0.013,
            DecisionClass.DISCARD_MID, 0.039,
            DecisionClass.DISCARD_LATE, 0.0312,
            DecisionClass.FINAL_END, 1.0));
    Assert.assertEquals(settings.labels().rollin(), rollin);
    Assert.assertEquals(settings.fit().temperature(), temperature);
    Assert.assertEquals(settings.noise().rates().table(), rates);
    for (String table : List.of("labels.rollin", "fit.temperature", "noise.rates.table")) {
      for (DecisionClass nuki : List.of(DecisionClass.NUKI, DecisionClass.NUKI_RESPONSE)) {
        // 学習の温度は打牌以外のクラスで 1 つの値なので、その値で書く
        rejects(table, "not of the variant [" + nuki + "]", table + "." + nuki + "=0.02");
      }
    }
    rejects("noise.rates.allActions", "classes of TENHOU_4P", "noise.rates.allActions=[NUKI]");
  }

  @Test
  public void eachNoiseMeasureSwitchedOffTakesTheRawPath() {
    PrismSettings defaults = TestSettings.of();
    Assert.assertTrue(defaults.noise().tempo() instanceof TempoSettings.On);
    Assert.assertTrue(defaults.noise().ledger().corrects());
    Assert.assertEquals(defaults.noise().crn().draws(), DrawOrder.SEAT_STREAMS);

    // TEMPO を切ると、教師は補正した Y のままで、アンカーなし・ゲイン 1・学習率の倍率 1 になる（R² にも反復にも依らない）
    TempoSettings tempo = TestSettings.of("noise.tempo.enabled=false").noise().tempo();
    Assert.assertTrue(tempo instanceof TempoSettings.Off, tempo.toString());
    for (int t : new int[] {1, 2, 40}) {
      for (double r2 : new double[] {0, 0.3}) {
        Assert.assertEquals(tempo.gain(t, r2), 1.0);
        Assert.assertEquals(tempo.learningRateScale(t, r2), 1.0);
      }
    }
    Assert.assertFalse(tempo.anchored());
    Assert.assertFalse(tempo.resetsOnPhiKind());
    Assert.assertTrue(TestSettings.of("noise.tempo.enabled=false").noise().ledger().corrects(), "TEMPO off keeps Ỹ");

    // 帳簿を OFF にすると、書かず、当てはめず、補正もしない。RECORD は書くが、当てはめも補正もしない（教師は生の Y）。AUDIT は当てはめるが
    // 補正しない。TEMPO のゲインは、補正の R² を使わない設定にしておく
    String noGain = "noise.tempo.noiseGain=NONE";
    LedgerSettings off = TestSettings.of("noise.ledger.mode=OFF", noGain).noise().ledger();
    Assert.assertEquals(List.of(off.records(), off.fits(), off.corrects()), List.of(false, false, false));
    LedgerSettings record = TestSettings.of("noise.ledger.mode=RECORD", noGain).noise().ledger();
    Assert.assertEquals(List.of(record.records(), record.fits(), record.corrects()), List.of(true, false, false));
    LedgerSettings audit = TestSettings.of("noise.ledger.mode=AUDIT", noGain).noise().ledger();
    Assert.assertEquals(List.of(audit.records(), audit.fits(), audit.corrects()), List.of(true, true, false));
    // 切った機能でしか使わないキーを書くと拒む（RECORD の帳簿の ridge、切った TEMPO の gain）
    Assert.assertThrows(
        SettingsException.class, () -> TestSettings.of("noise.ledger.mode=RECORD", noGain, "noise.ledger.ridge=0.01"));
    Assert.assertThrows(SettingsException.class, () -> TestSettings.of("noise.tempo.enabled=false", "noise.tempo.gain=0.5"));

    // 席ごとのツモの列を切ると、半荘を始める呼び出しに牌山の順のツモが渡る
    Assert.assertEquals(TestSettings.of("noise.crn.seatStreams=false").noise().crn().draws(), DrawOrder.WALL_ORDER);
  }

  /**
   * 自前の GRP を学習し直すときの設定は、Φ の自前の GRP の設定の中の grp.training に書く。classic の GRP の学習と同じ record で、既定値も
   * 同じ。ほかの Φ の出どころでは書けない。
   */
  @Test
  public void theSelfGrpRefitsLearnByTheGrpTrainingKeys() {
    String[] self = {"phi.source=SELF_GRP", "phi.start.kind=SCORE_RANK"};
    GrpTraining defaults = ((PhiSettings.SelfGrp) TestSettings.of(self).phi()).grp().training();
    Assert.assertEquals(defaults, Settings.section(GrpTraining.class, "grp.training", List.of()).value());
    Assert.assertEquals(defaults, new GrpTraining(4096, 10.0, 2, 4, 1e-3, 16, 3e-4, 1));
    PhiSettings.SelfGrp written =
        (PhiSettings.SelfGrp) TestSettings.of(TestSettings.concat(self, "phi.grp.training.warmEpochs=3")).phi();
    Assert.assertEquals(written.grp().training().warmEpochs(), 3);
    rejects(
        "phi.grp.training.stopPatience",
        "below halvingPatience",
        TestSettings.concat(self, "phi.grp.training.stopPatience=1"));
    rejects("phi.grp", "", "phi.source=SCORE_RANK", "phi.grp.training.warmEpochs=3");
  }

  /**
   * バージョン 1 の初期化したままのネットワークは、network のセクションでどの形のアームからでも始められる。既定は v2 の作りの深さ 2。
   * v3 の作りのアームは深さ 3 だけなので、深さのキーを書くと拒む。形の名前はアームの定数の綴りだけを受ける。
   */
  @Test
  public void theNetworkStartsFromAnyShapeArm() {
    Assert.assertEquals(TestSettings.of().network().shape(), PrismShape.standard(2));
    Assert.assertEquals(TestSettings.of("network.depth=3").network().shape(), PrismShape.standard(3));
    for (ShapeArm arm : ShapeArm.values()) {
      PrismShape expected = arm == ShapeArm.V2 ? PrismShape.standard(2) : arm.shape(ShapeArm.V3_DEPTH);
      Assert.assertEquals(TestSettings.of("network.shape=" + arm.name()).network().shape(), expected, arm.name());
    }
    rejects("network.depth", "does not apply", "network.shape=V3_ONE_STAGE", "network.depth=3");
    rejects("network.shape", "one of", "network.shape=v3");
  }

  /**
   * 同値な打牌の切り方は行動方策の節のキーで、既定はツモ切り。収集から効く（COLLECT）。手出しに替えられ、ほかの値はキーのパスを添えて拒む。
   * 系列の入口が使う既定の節も、ツモ切り。
   */
  @Test
  public void twinDiscardsAreTsumogiriByDefault() {
    Assert.assertEquals(TestSettings.of().behaviour().twinDiscard(), TwinDiscard.TSUMOGIRI);
    Assert.assertEquals(BehaviourSettings.defaults().twinDiscard(), TwinDiscard.TSUMOGIRI);
    Assert.assertEquals(TestSettings.of("behaviour.twinDiscard=TEDASHI").behaviour().twinDiscard(), TwinDiscard.TEDASHI);
    KeyInfo key =
        Settings.describe(PrismSettings.class).stream()
            .filter(info -> info.path().equals("behaviour.twinDiscard"))
            .findFirst()
            .orElseThrow();
    Assert.assertEquals(key.value(), "TSUMOGIRI");
    Assert.assertEquals(key.change(), Change.COLLECT);
    rejects("behaviour.twinDiscard", "", "behaviour.twinDiscard=SIDEWAYS");
  }

  /** 上書きを重ねた設定が拒まれ、どれかのエラーメッセージがキーのパスと理由を含むことを確かめる。 */
  private static void rejects(String path, String because, String... assignments) {
    SettingsException failure = Assert.expectThrows(SettingsException.class, () -> TestSettings.of(assignments));
    boolean named = failure.problems().stream().anyMatch(problem -> problem.contains(path) && problem.contains(because));
    Assert.assertTrue(named, String.join(" ", assignments) + " -> " + failure.problems());
  }

  /** 四人打ちのルールの層の TOML を返す。 */
  private static String variantLayer() throws IOException {
    try (InputStream in = PrismApp.class.getClassLoader().getResourceAsStream("prism/variants/TENHOU_4P.toml")) {
      Assert.assertNotNull(in);
      return new String(in.readAllBytes(), StandardCharsets.UTF_8);
    }
  }
}
