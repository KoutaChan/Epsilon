package com.epsilon.reviewer.engine;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.Series;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.review.ReviewModel;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.model.CatalogModel;
import com.epsilon.reviewer.model.ModelCatalog;
import com.epsilon.reviewer.model.ModelFamilies;
import com.epsilon.reviewer.model.ModelKey;
import com.epsilon.reviewer.model.ModelLease;
import com.epsilon.reviewer.model.ModelRegistry;
import com.google.gson.JsonArray;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

/**
 * 三人打ちの PRISM を Reviewer で使うテスト。テストの中で、乱数で初期化した小さなネットワーク（v2 の形）を四人打ちと三人打ちで 1 つずつ
 * 保存し、models.json の項目から、クラスパスにある系列の入口を使って CPU で開く。三人打ちのモデルは、エンジンで打った三人打ちの半荘と
 * 手で並べた局の全ての判断を評価する。変更の無いもしもは、全ての判断で解析と同じ O と候補になる。カタログは席の数で一覧を分け、ルールの
 * 違う牌譜は英語の理由を付けて断る。
 */
public class ThreePlayerPrismReviewTest {
  private static final Variant VARIANT = Variant.TENHOU_3P;

  private static final ModelKey FOUR = new ModelKey("prism-4p", "r1");
  private static final ModelKey THREE = new ModelKey("prism-3p", "r1");
  private static final ModelKey RULES = new ModelKey("rules-3p", "r1");

  /** 三人打ちのネットワークを四人打ちと書いた項目。カタログには載るが、開くと断られる。 */
  private static final ModelKey WRONG = new ModelKey("prism-3p-as-4p", "r1");

  private static Fixture fixture;

  /** 保存したネットワークのフォルダーと、それを開くレジストリ。最初に使うテストが作る。 */
  private record Fixture(Path directory, ModelCatalog catalog, ModelRegistry registry) {}

  @AfterClass(alwaysRun = true)
  public void close() {
    synchronized (ThreePlayerPrismReviewTest.class) {
      if (fixture == null) return;
      try {
        fixture.registry().close();
      } finally {
        TestFiles.delete(fixture.directory());
        fixture = null;
      }
    }
  }

  /** 全ての牌譜の全ての判断を評価する。北抜きの手と、抜いた北への応答にも候補と価値が付き、確率の和は 1 になる。 */
  @Test(groups = "native", timeOut = 600_000)
  public void theThreePlayerPrismReviewsEveryDecisionWithValues() throws Exception {
    Fixture fixture = fixture();
    int nukiCandidates = 0;
    int nukiResponses = 0;
    try (ModelLease lease = fixture.registry().acquire(THREE)) {
      Assert.assertTrue(lease.valued());
      Assert.assertEquals(lease.variant(), VARIANT);
      Map<String, LoadedRecord> records = new LinkedHashMap<>(PlayedRecords.threePlayer());
      records.putAll(PlayedRecords.threePlayerScenes());
      for (Map.Entry<String, LoadedRecord> entry : records.entrySet()) {
        String name = entry.getKey();
        var model = new Recorded(lease);
        Format6.Document document = Format6.analyze(entry.getValue(), model);
        Assert.assertEquals(
            document.json().get("temperature").getAsDouble(), lease.temperature(), name);
        List<Format6.Located> decisions = document.decisions();
        Assert.assertFalse(decisions.isEmpty(), name);
        Assert.assertEquals(model.inputs.size(), decisions.size(), name);
        for (Format6.Round round : document.rounds()) {
          Map<Integer, String> events = new HashMap<>();
          for (Format6.Step step : round.steps()) {
            Assert.assertEquals(step.table().players().size(), 3, name);
            if (!step.decision()) {
              events.put(step.eventIndex(), step.event());
              continue;
            }
            String where = name + " event " + step.eventIndex() + " seat " + step.actor();
            assertValued(step, where);
            if (step.offers(Action.Type.NUKI)) nukiCandidates++;
            if ("kita".equals(events.get(step.cause()))) {
              nukiResponses++;
              Assert.assertTrue(step.offers(Action.Type.PASS), where);
            }
          }
        }
      }
    }
    Assert.assertTrue(nukiCandidates > 0, "no decision offers a nuki");
    Assert.assertTrue(nukiResponses > 0, "no response to a nuki");
  }

  /** 変更の無いもしもは、全ての判断で解析と同じ O を方策に渡し、同じ候補・選んだ手・確率を返す。 */
  @Test(groups = "native", timeOut = 1_200_000)
  public void anUnchangedScenarioMatchesTheReviewAtEveryDecision() throws Exception {
    Fixture fixture = fixture();
    Map<String, LoadedRecord> records = new LinkedHashMap<>();
    PlayedRecords.threePlayer()
        .forEach(
            (name, record) -> {
              records.put(name, record);
              records.put(name + " mjai", PlayedRecords.mjai(record));
            });
    records.putAll(PlayedRecords.threePlayerScenes());
    Set<EngineDecisionKind> kinds = EnumSet.noneOf(EngineDecisionKind.class);
    try (ModelLease lease = fixture.registry().acquire(THREE)) {
      for (Map.Entry<String, LoadedRecord> entry : records.entrySet()) {
        String name = entry.getKey();
        LoadedRecord record = entry.getValue();
        var review = new Recorded(lease);
        Format6.Document document = Format6.analyze(record, review);
        int decision = 0;
        for (Format6.Located located : document.decisions()) {
          Format6.Step step = located.step();
          String where =
              name
                  + " round "
                  + located.round()
                  + " event "
                  + step.eventIndex()
                  + " seat "
                  + step.actor();
          var scenario = new Recorded(lease);
          ScenarioAnswer answer = scenario(record, located, scenario);
          Assert.assertEquals(scenario.inputs.size(), 1, where);
          Assert.assertEquals(scenario.inputs.getFirst(), review.inputs.get(decision), where);
          Assert.assertEquals(Format6.candidateSet(answer), step.candidateSet(), where);
          kinds.add(ObservedInput.of(VARIANT, review.inputs.get(decision)).kind());
          decision++;
        }
        Assert.assertEquals(decision, review.inputs.size(), name);
      }
    }
    Assert.assertEquals(kinds, EnumSet.allOf(EngineDecisionKind.class));
  }

  /** 一覧は席の数を持ち、四人打ちと三人打ちのモデルが分かれる。ネットワークのルールと違う項目は開けない。 */
  @Test(groups = "native", timeOut = 300_000)
  public void theCatalogSplitsModelsBySeats() throws Exception {
    Fixture fixture = fixture();
    List<CatalogModel> models = fixture.catalog().view().models();
    Assert.assertTrue(models.stream().allMatch(CatalogModel::available));
    Assert.assertEquals(ids(models, 4), List.of(FOUR.modelId(), WRONG.modelId()));
    Assert.assertEquals(ids(models, 3), List.of(THREE.modelId(), RULES.modelId()));
    ReviewerException wrong =
        Assert.expectThrows(ReviewerException.class, () -> fixture.registry().acquire(WRONG));
    Assert.assertEquals(wrong.code(), ErrorCode.MODEL_UNAVAILABLE);
    try (ModelLease four = fixture.registry().acquire(FOUR);
        ModelLease three = fixture.registry().acquire(THREE)) {
      Assert.assertEquals(four.variant(), Variant.TENHOU_4P);
      Assert.assertEquals(three.variant(), VARIANT);
      // 同じ JVM で、四人打ちのモデルは四人打ちの牌譜をそのまま解析できる。
      Assert.assertFalse(
          Format6.analyze(PlayedRecords.read(Variant.TENHOU_4P, 0), four).decisions().isEmpty());
    }
  }

  /** ルールの違う牌譜は、解析でももしもでも、model_unavailable と英語の理由で断る。四人打ちと三人打ちのモデルの両方で確かめる。 */
  @Test(groups = "native", timeOut = 300_000)
  public void aModelRefusesRecordsOfTheOtherRules() throws Exception {
    Fixture fixture = fixture();
    LoadedRecord threeRecord = PlayedRecords.read(VARIANT, 0);
    LoadedRecord fourRecord = PlayedRecords.read(Variant.TENHOU_4P, 0);
    try (ModelLease four = fixture.registry().acquire(FOUR);
        ModelLease three = fixture.registry().acquire(THREE)) {
      Format6.Located threeDecision = Format6.analyze(threeRecord, three).decisions().getFirst();
      Format6.Located fourDecision = Format6.analyze(fourRecord, four).decisions().getFirst();
      String fourReadsThree = "The model reads TENHOU_4P records, not TENHOU_3P.";
      String threeReadsFour = "The model reads TENHOU_3P records, not TENHOU_4P.";
      assertRefused(() -> Format6.analyze(threeRecord, four), fourReadsThree);
      assertRefused(() -> Format6.analyze(fourRecord, three), threeReadsFour);
      assertRefused(() -> scenario(threeRecord, threeDecision, four), fourReadsThree);
      assertRefused(() -> scenario(fourRecord, fourDecision, three), threeReadsFour);
    }
  }

  private static void assertRefused(Assert.ThrowingRunnable call, String message) {
    ReviewerException refused = Assert.expectThrows(ReviewerException.class, call);
    Assert.assertEquals(refused.code(), ErrorCode.MODEL_UNAVAILABLE);
    Assert.assertEquals(refused.getMessage(), message);
  }

  private static ScenarioAnswer scenario(
      LoadedRecord record, Format6.Located located, ReviewModel model) {
    Format6.Step step = located.step();
    return Scenarios.evaluate(
        record,
        new ScenarioQuery(located.round(), step.eventIndex(), step.actor(), TestEdits.NONE, null),
        model);
  }

  /** 全ての候補が有限の価値の損失を持ち、最善の候補の損失は 0 で、損失の小さい順に並ぶ。 */
  private static void assertValued(Format6.Step step, String where) {
    JsonArray candidates = step.candidates();
    Assert.assertFalse(candidates.isEmpty(), where);
    float previous = 0f;
    for (int i = 0; i < candidates.size(); i++) {
      JsonArray candidate = candidates.get(i).getAsJsonArray();
      Assert.assertTrue(candidate.size() >= 2, where);
      float loss = candidate.get(1).getAsFloat();
      Assert.assertTrue(Float.isFinite(loss), where);
      if (i == 0) Assert.assertEquals(loss, 0f, 1e-6f, where + ": the best candidate has no loss");
      else Assert.assertTrue(previous <= loss, where);
      previous = loss;
    }
  }

  private static List<String> ids(List<CatalogModel> models, int seats) {
    return models.stream()
        .filter(model -> model.seats() == seats)
        .map(CatalogModel::modelId)
        .toList();
  }

  private static synchronized Fixture fixture() throws Exception {
    if (fixture != null) return fixture;
    Path directory = TestFiles.directory("reviewer-prism-3p-");
    try {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
        for (Variant variant : List.of(Variant.TENHOU_4P, VARIANT)) {
          PrismNetwork network = new PrismNetwork(InputFormat.of(variant), PrismShape.standard(2));
          network.initialize(manager);
          Snapshot.write(directory.resolve(variant.name()), network, PolicyParams.of(1));
        }
      }
      Path models = directory.resolve("models.json");
      Files.writeString(
          models,
          "{\"models\":["
              + entry(FOUR, Series.PRISM, Variant.TENHOU_4P, Variant.TENHOU_4P.name())
              + ","
              + entry(THREE, Series.PRISM, VARIANT, VARIANT.name())
              + ","
              + entry(RULES, Series.RULE_BOT, VARIANT, null)
              + ","
              + entry(WRONG, Series.PRISM, Variant.TENHOU_4P, VARIANT.name())
              + "]}");
      ModelFamilies families = ModelFamilies.load();
      ModelCatalog catalog = ModelCatalog.load(models, families);
      var registry = new ModelRegistry(catalog, families, 4, Duration.ofMinutes(10));
      fixture = new Fixture(directory, catalog, registry);
      return fixture;
    } catch (Exception | Error failure) {
      TestFiles.delete(directory);
      throw failure;
    }
  }

  /**
   * models.json の項目 1 つ。
   *
   * @param checkpoint models.json からの相対のチェックポイント。無ければ null
   */
  private static String entry(ModelKey key, Series series, Variant variant, String checkpoint) {
    return "{\"modelId\":\""
        + key.modelId()
        + "\",\"revision\":\""
        + key.revision()
        + "\",\"displayName\":\"Model\",\"version\":\"1\",\"series\":\""
        + series.id()
        + "\",\"variant\":\""
        + variant.name()
        + "\""
        + (checkpoint == null
            ? ""
            : ",\"checkpoint\":\"" + checkpoint + "\",\"options\":{\"device\":\"cpu\"}")
        + "}";
  }

  /**
   * 借りたモデルの評価をそのまま返し、受け取った O の写しを順に取っておく。評価ごとに、確率が合法手の上で和が 1 になること、価値が
   * 全ての合法手にあること、最善の手が合法手の中にあることを確かめる。
   */
  private static final class Recorded implements ReviewModel {
    /** 受け取った O の写し。受け取った順に並ぶ。 */
    final List<long[]> inputs = new ArrayList<>();

    private final ModelLease lease;
    private final ObservationBatch batch = new ObservationBatch(VARIANT, 256);

    Recorded(ModelLease lease) {
      this.lease = lease;
    }

    @Override
    public Evaluation evaluate(Observation observation) {
      batch.clear();
      int index = batch.add(observation);
      int start = batch.start(index);
      inputs.add(Arrays.copyOfRange(batch.words(), start, start + batch.longs(index)));
      Evaluation evaluation = lease.evaluate(observation);
      int legal = observation.legalActions();
      Assert.assertEquals(evaluation.probabilities().length, legal);
      double sum = 0;
      for (float probability : evaluation.probabilities()) sum += probability;
      Assert.assertEquals(sum, 1.0, Evaluation.SUM_TOLERANCE);
      Assert.assertNotNull(evaluation.values(), "PRISM gives values");
      Assert.assertEquals(evaluation.values().length, legal);
      Assert.assertTrue(evaluation.greedy() >= 0 && evaluation.greedy() < legal);
      return evaluation;
    }

    @Override
    public boolean valued() {
      return lease.valued();
    }

    @Override
    public double temperature() {
      return lease.temperature();
    }

    @Override
    public Variant variant() {
      return lease.variant();
    }
  }
}
