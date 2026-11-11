package com.epsilon.agent.policy;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 系列の実装は ServiceLoader で見つかる。開いた方策は、判断の記録を読んで手を返す。実装の無い系列は例外になる。呼び出し側が渡した
 * ランタイムは、方策を開くメソッドにそのまま届く。渡さなければ null が届く。
 */
public class ModelFamilyTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final PolicyOptions CPU = PolicyOptions.defaults();

  @Test
  public void serviceLoaderFindsTheFamilyOfASeries() throws IOException {
    ModelFamily family = ModelFamily.of(Series.RULE_BOT);
    assertTrue(family instanceof FirstLegalFamily, family.getClass().getName());
    Observation observation = firstDecision();
    try (Policy policy = family.open(null, VARIANT, CPU)) {
      assertEquals(policy.variant(), VARIANT);
      assertEquals(policy.label(), "heuristic");
      InlinePolicy inline = (InlinePolicy) policy;
      assertEquals(inline.act(observation), 0);
      Evaluation evaluation = policy.evaluate(observation);
      assertEquals(evaluation.probabilities().length, observation.legalActions());
      assertEquals(evaluation.greedy(), 0);
    }
  }

  @Test
  public void aSeriesWithoutAFamilyIsRejected() {
    IllegalStateException missing = expectThrows(IllegalStateException.class, () -> ModelFamily.of(Series.NANO));
    assertTrue(missing.getMessage().contains("NANO"), missing.getMessage());
  }

  @Test
  public void theCallersRuntimeReachesTheFamily() throws IOException {
    List<NetworkRuntime> received = new ArrayList<>();
    ModelFamily family =
        new ModelFamily() {
          @Override
          public Series series() {
            return Series.RULE_BOT;
          }

          @Override
          public Policy open(Path checkpoint, Variant variant, PolicyOptions options, NetworkRuntime runtime) {
            received.add(runtime);
            return new FirstLegalFamily().open(checkpoint, variant, options, runtime);
          }
        };
    try (NetworkRuntime runtime = new NetworkRuntime(1, 1, 1)) {
      family.open(null, VARIANT, CPU).close();
      family.open(null, VARIANT, CPU, runtime).close();
      assertEquals(received, Arrays.asList(null, runtime));
    }
  }

  /** 半荘の最初の判断（親の手番）を記録したもの。 */
  static Observation firstDecision() {
    Engine engine = new Engine(VARIANT);
    RuleWorkspace ws = new RuleWorkspace(VARIANT);
    engine.startGame(7, DrawOrder.WALL_ORDER);
    assertEquals(engine.advance(ws), Boundary.DECISIONS);
    Decisions decisions = engine.decisions();
    ObservationBatch batch = new ObservationBatch(VARIANT, 64);
    int index = new ObservationWriter(VARIANT).capture(engine.view(decisions.seat(0)), decisions, 0, batch);
    return new Observation(VARIANT).bind(batch, index);
  }
}
