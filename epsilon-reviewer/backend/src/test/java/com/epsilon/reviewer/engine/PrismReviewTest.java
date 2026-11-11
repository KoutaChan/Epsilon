package com.epsilon.reviewer.engine;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.Series;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.engine.review.ReviewModel;
import com.epsilon.reviewer.model.ModelCatalog;
import com.epsilon.reviewer.model.ModelFamilies;
import com.epsilon.reviewer.model.ModelKey;
import com.epsilon.reviewer.model.ModelLease;
import com.epsilon.reviewer.model.ModelRegistry;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * テストの中で作った小さな PRISM のバージョンを、モデルの一覧から CPU で開き、天鳳の半荘の全ての席の判断を評価する。開くときは、
 * クラスパスにある PRISM の系列の入口を使う。確率は合法手の上で和が 1 になり、価値は全ての候補に付く。解析の文書では、候補が価値の
 * 損失の小さい順に並び、先頭は PRISM が打つ手になる。
 */
public class PrismReviewTest {
  private static final ModelKey KEY = new ModelKey("prism", "r1");

  @Test(groups = "native")
  public void prismSnapshotReviewsEverySeatWithValues() throws Exception {
    Path directory = TestFiles.directory("reviewer-prism-");
    try {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
        PrismNetwork network = new PrismNetwork(InputFormat.of(Variant.TENHOU_4P), PrismShape.standard(2));
        network.initialize(manager);
        Snapshot.write(directory.resolve("snapshot"), network, PolicyParams.of(1));
      }
      Path models = directory.resolve("models.json");
      Files.writeString(
          models,
          "{\"models\":[{\"modelId\":\"prism\",\"revision\":\"r1\",\"displayName\":\"Prism\","
              + "\"version\":\"1\",\"series\":\""
              + Series.PRISM.id()
              + "\",\"variant\":\""
              + Variant.TENHOU_4P.name()
              + "\",\"checkpoint\":\"snapshot\",\"options\":{\"device\":\"cpu\"}}]}");
      ModelFamilies families = ModelFamilies.load();
      Assert.assertEquals(
          families.find(Series.PRISM).orElseThrow().getClass().getName(),
          "com.epsilon.prism.play.PrismFamily");
      ModelCatalog catalog = ModelCatalog.load(models, families);
      Assert.assertTrue(catalog.view().models().getFirst().available());
      AtomicInteger evaluated = new AtomicInteger();
      List<Integer> greedy = new ArrayList<>();
      try (var registry = new ModelRegistry(catalog, families, 1, Duration.ofMinutes(10));
          ModelLease lease = registry.acquire(KEY)) {
        Assert.assertTrue(lease.valued());
        // models.json に温度が無いので、PRISM の方策の温度になる
        Assert.assertEquals(lease.temperature(), 0.002);
        Format6.Document document =
            Format6.analyze(PlayedRecords.read(Variant.TENHOU_4P, 0), checked(lease, evaluated, greedy));
        Assert.assertEquals(
            document.json().get("temperature").getAsDouble(), lease.temperature());
        Set<Integer> seats = new HashSet<>();
        List<Format6.Located> decisions = document.decisions();
        for (int d = 0; d < decisions.size(); d++) {
          Format6.Step step = decisions.get(d).step();
          seats.add(step.actor());
          Assert.assertEquals(value(step, 0), 0f, 1e-6f, "the best candidate has no loss");
          // 同じ損失の候補の中では、PRISM が打つ手（同値な打牌は切る方の手）が先頭に来る
          Assert.assertEquals(step.actionId(0), greedy.get(d).intValue(), "the first candidate is the played action");
          for (int i = 1; i < step.candidates().size(); i++)
            Assert.assertTrue(value(step, i - 1) <= value(step, i));
        }
        Assert.assertEquals(seats, Set.of(0, 1, 2, 3));
        Assert.assertEquals(evaluated.get(), document.decisions().size());
      }
    } finally {
      TestFiles.delete(directory);
    }
  }

  /** 評価ごとに、確率と価値の形を確かめ、貪欲な手の行動の番号を {@code greedy} に足してから返す。 */
  private static ReviewModel checked(ModelLease lease, AtomicInteger evaluated, List<Integer> greedy) {
    return new ReviewModel() {
      @Override
      public Evaluation evaluate(Observation observation) {
        Evaluation evaluation = lease.evaluate(observation);
        int legal = observation.legalActions();
        Assert.assertEquals(evaluation.probabilities().length, legal);
        double sum = 0;
        for (float probability : evaluation.probabilities()) sum += probability;
        Assert.assertEquals(sum, 1.0, Evaluation.SUM_TOLERANCE);
        Assert.assertNotNull(evaluation.values(), "PRISM gives values");
        Assert.assertEquals(evaluation.values().length, legal);
        Assert.assertTrue(evaluation.greedy() >= 0 && evaluation.greedy() < legal);
        greedy.add(observation.action(evaluation.greedy()).code());
        evaluated.incrementAndGet();
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
    };
  }

  private static float value(Format6.Step step, int candidate) {
    return step.candidates().get(candidate).getAsJsonArray().get(1).getAsFloat();
  }
}
