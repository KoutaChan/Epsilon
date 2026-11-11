package com.epsilon.prism3p;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;
import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.CrnScheme;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.grp.Phi;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.Limits;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.backend.WeightLayout;
import com.epsilon.prism.collect.Collector;
import com.epsilon.prism.collect.RowSink;
import com.epsilon.prism.collect.Sampling;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.hip.HipArchitecture;
import com.epsilon.prism.hip.HipCounts;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.noise.UraSettings;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism3p.input.InputSchema;
import com.epsilon.prism3p.input.InputWriter;
import com.epsilon.prism3p.input.Tokenizer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちと四人打ちの食い違いを拒むテスト。三人打ちの入力のライターに四人打ちの O を渡したとき、収集で Φ かコホートが四人打ちのとき、
 * 方策に四人打ちの O を渡したとき、系列の入口で三人打ちのネットワークを四人打ちとして開いたとき、HIP の表に三人打ちの形式と数の合わない
 * 構成を渡したときに拒む。
 */
public class VariantMismatchTest {

  private static final PolicyOptions CPU = PolicyOptions.defaults();

  private static final BatchPlan.Capacity ONE_ROW =
      new BatchPlan.Capacity(1, Limits.MAX_EVENT_TOKENS, Limits.MAX_TRANSITIONS, Limits.MAX_LEGAL_ACTIONS);

  /** 乱数の種を決めた四人打ちの局で、最初の判断の O を返す。 */
  private static Observation fourPlayerObservation() {
    ObservationWriter writer = new ObservationWriter(TENHOU_4P);
    ObservationBatch batch = new ObservationBatch(TENHOU_4P, 1 << 12);
    Observation reader = new Observation(TENHOU_4P);
    boolean[] captured = {false};
    new TableWalk(TENHOU_4P, TableWalk.Policy.GREEDY)
        .run(
            31,
            1,
            (engine, slot) -> {
              if (!captured[0]) {
                Decisions decisions = engine.decisions();
                batch.clear();
                reader.bind(batch, writer.capture(engine.view(decisions.seat(slot)), decisions, slot, batch));
                captured[0] = true;
              }
            });
    Assert.assertTrue(captured[0]);
    return reader;
  }

  private static HostBatch threePlayerRow(Observation o) {
    HostBatch batch =
        new HostBatch(InputSchema.FORMAT, ONE_ROW, ByteBuffer.allocate(BatchPlan.maxBytes(InputSchema.FORMAT, ONE_ROW)));
    batch.plan().add(o);
    batch.plan().seal();
    return batch;
  }

  @Test
  public void theWriterRefusesAFourPlayerObservation() {
    Observation four = fourPlayerObservation();
    HostBatch batch = threePlayerRow(four);
    IllegalArgumentException refused =
        Assert.expectThrows(IllegalArgumentException.class, () -> new Tokenizer().write(four, new InputWriter().bind(batch, 0)));
    Assert.assertEquals(refused.getMessage(), "Variant mismatch: expected TENHOU_3P but found TENHOU_4P");
  }

  @Test
  public void collectionRefusesAFourPlayerPhiOrCohort() throws IOException {
    assertCollectionRefused(TENHOU_3P, TENHOU_4P);
    assertCollectionRefused(TENHOU_4P, TENHOU_3P);
    assertCollectionRefused(TENHOU_4P, TENHOU_4P);
  }

  private static void assertCollectionRefused(Variant cohort, Variant phi) throws IOException {
    Path directory = Files.createTempDirectory("prism-3p-mismatch");
    var plan =
        new Collector.Cohort(
            cohort,
            1,
            2,
            List.of(10L),
            DrawOrder.SEAT_STREAMS,
            Sampling.OFF,
            null,
            new UraSettings.Realized(0.05),
            new Behaviour(0.05, 0, CrnScheme.SINGLE_RACE),
            TwinDiscard.TSUMOGIRI,
            null);
    try (var table = new TableLoop(new TableSettings(1, 1, 1_000_000L, 1));
        var writer = new CohortWriter(directory, TENHOU_3P, 1, 0, 2, CohortWriter.Tap.NONE)) {
      PolicyService unused = new UnusedPolicy();
      IllegalStateException refused =
          Assert.expectThrows(
              IllegalStateException.class,
              () -> Collector.collect(table, plan, unused, Phi.scoreRank(phi, RankUtility.TENHOU), new RowSink(writer)));
      Assert.assertTrue(refused.getMessage().startsWith("Variant mismatch: "), refused.getMessage());
    } finally {
      Snapshot.deleteTree(directory);
    }
  }

  /** 食い違いのテストでは呼ばれない方策。 */
  private static final class UnusedPolicy implements PolicyService {
    @Override
    public int version() {
      return 1;
    }

    @Override
    public boolean inline() {
      return true;
    }

    @Override
    public void evaluateInline(PolicyRequest request) {
      throw new AssertionError("the collection must be refused before any decision");
    }
  }

  private static Path writeSnapshot(Path root, NDManager manager) throws IOException {
    Path snapshot = root.resolve("1");
    PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, PrismShape.standard(2));
    network.initialize(manager);
    Snapshot.write(snapshot, network, PolicyParams.of(1));
    return snapshot;
  }

  @Test(timeOut = 300_000)
  public void thePolicyAndTheSeriesEntryRefuseAnotherVariant() throws Exception {
    Path root = Files.createTempDirectory("prism-3p-family");
    try {
      Path snapshot;
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        snapshot = writeSnapshot(root, manager);
      }
      Observation four = fourPlayerObservation();
      ModelFamily family = ModelFamily.of(Series.PRISM);
      try (Policy policy = family.open(snapshot, TENHOU_3P, CPU)) {
        IllegalArgumentException refused = Assert.expectThrows(IllegalArgumentException.class, () -> policy.evaluate(four));
        Assert.assertEquals(refused.getMessage(), "Variant mismatch: expected TENHOU_3P but found TENHOU_4P");
      }
      // 四人打ちのモジュールが無い classpath では、四人打ちとしては開けない
      IllegalStateException refused =
          Assert.expectThrows(IllegalStateException.class, () -> family.open(snapshot, TENHOU_4P, CPU));
      Assert.assertTrue(refused.getMessage().contains("TENHOU_4P"), refused.getMessage());
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void theHipTableRefusesAnArchitectureOfOtherCounts() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, PrismShape.standard(2));
      network.initialize(manager);
      HipArchitecture three = WeightLayout.of(network).architecture();
      HipCounts four = new HipCounts(4, 34, three.counts().keyCodes(), three.counts().tileRelations(), 110);
      HipArchitecture other =
          new HipArchitecture(
              three.hidden(),
              three.epsilon(),
              three.scale(),
              three.tau(),
              three.layers(),
              three.candidateFfn(),
              three.stages(),
              three.gate(),
              three.gateKinds(),
              three.roles(),
              four);
      IllegalArgumentException refused =
          Assert.expectThrows(
              IllegalArgumentException.class,
              () -> WeightLayout.of(InputSchema.FORMAT, other, network.inferenceParameters()));
      Assert.assertEquals(
          refused.getMessage(), "the TENHOU_3P input has the counts " + three.counts() + ", the architecture has " + four);
    }
  }
}
