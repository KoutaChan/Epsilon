package com.epsilon.classic.learn;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotEquals;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.RandomCheckpoints;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.classic.model.ClassicNetwork;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 自己対局の学習器を、乱数の重みの小さな checkpoint とランダムな自己対局の軌跡を使って、CPU の FP32 で回す。次のことを確かめる。
 * 1 ステップの学習はシードだけで決まる。状態を書いて続きから学んだ結果は、途切れずに学んだ結果とビットで同じ。KL が上限を超えたステップは
 * 方策の更新だけを捨てる。2 つのレーンのデータ並列は、1 つのデバイスと許容誤差の内で同じステップを踏む。
 */
public class SelfPlayLearnerTest {

  private static final Logger LOG = LoggerFactory.getLogger(SelfPlayLearnerTest.class);
  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final int MEMORIZE_STEPS = 8;
  private static final int MEMORIZE_MACROS = 25;

  private Path root;
  private Path start;
  private TrajectoryBatch batch;

  /** パラメーターの名前と値の組。並びは {@code getParameters()} の順。 */
  private record Weights(List<String> names, List<float[]> values) {

    static Weights of(ClassicNetwork network) {
      List<String> names = new ArrayList<>();
      List<float[]> values = new ArrayList<>();
      for (Pair<String, Parameter> pair : network.getParameters()) {
        names.add(pair.getKey());
        values.add(pair.getValue().getArray().toFloatArray());
      }
      return new Weights(names, values);
    }

    /** 価値の組（true）か方策の組（false）のパラメーターが、ほかの重みとビットで同じなら true。 */
    boolean sameGroup(Weights other, boolean value) {
      for (int index = 0; index < names.size(); index++) {
        if (ClassicNetwork.valueParameter(names.get(index)) == value
            && !Arrays.equals(values.get(index), other.values.get(index))) {
          return false;
        }
      }
      return true;
    }

    boolean same(Weights other) {
      return sameGroup(other, true) && sameGroup(other, false);
    }
  }

  @BeforeClass(groups = "native")
  public void prepare() throws IOException {
    root = Files.createTempDirectory("selfplay-learner");
    start = root.resolve("start");
    RandomCheckpoints.write(start, Architecture.NANO, VARIANT, 7);
    batch = Trajectories.play(VARIANT, 3, 3, 120);
    LOG.info("Learner test batch: games={} decisions={}", batch.games().size(), batch.size());
  }

  @AfterClass(groups = "native", alwaysRun = true)
  public void cleanUp() throws IOException {
    if (root != null) {
      try (Stream<Path> paths = Files.walk(root)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
          Files.deleteIfExists(path);
        }
      }
    }
  }

  /**
   * 1 ステップの macro を同じ checkpoint・同じシードから 2 回学ぶと、記録（秒数を除く）も重みもビットで同じ。価値と方策のどちらの組も
   * 動く。
   */
  @Test(groups = "native", timeOut = 900_000)
  public void oneStepFromARandomCheckpointIsDeterministic() throws IOException {
    SelfPlayLearner.Settings settings = settings(1, 0.01, 1e-4);
    Weights initial;
    Weights[] after = new Weights[2];
    SelfPlayLearner.Macro[] macros = new SelfPlayLearner.Macro[2];
    for (int run = 0; run < 2; run++) {
      try (Model model = model(start);
          SelfPlayLearner learner = new SelfPlayLearner(model, NetworkDevices.of(Device.cpu()), null, settings)) {
        initial = Weights.of(network(model));
        macros[run] = learner.learn(batch, 5);
        after[run] = Weights.of(network(model));
        assertFalse(after[run].sameGroup(initial, false), "the policy moved");
        assertFalse(after[run].sameGroup(initial, true), "the value moved");
        assertEquals(learner.macros(), 1);
        assertEquals(learner.actorSteps(), 1);
        assertEquals(learner.valueSteps(), 1);
      }
    }
    assertEquals(withoutSeconds(macros[1]), withoutSeconds(macros[0]));
    assertTrue(after[1].same(after[0]), "the weights differ between two runs");
    SelfPlayLearner.Macro macro = macros[0];
    assertEquals(macro.games(), batch.games().size());
    assertEquals(macro.decisions(), batch.size());
    assertEquals(macro.actorRows(), actorRows(batch));
    SelfPlayLearner.Step step = macro.steps().getFirst();
    assertEquals(step.rows(), batch.size());
    assertEquals(step.actorRows(), macro.actorRows());
    assertTrue(step.actorUpdated());
    assertTrue(step.kl() >= 0 && step.kl() < 1e-5, "first step KL " + step.kl());
    assertTrue(step.clipFraction() < 1e-3, "clipped " + step.clipFraction());
    assertTrue(Double.isFinite(step.actorNorm()) && step.actorNorm() > 0, "actor norm " + step.actorNorm());
    assertTrue(Double.isFinite(step.valueNorm()) && step.valueNorm() > 0, "value norm " + step.valueNorm());
    assertTrue(macro.credit() > 0.1 - 1e-6 && macro.credit() <= 1, "credit " + macro.credit());
    assertEquals(macro.rate().action(), KlLearningRate.Action.WARMUP);
    LOG.info("One step: {}", macro);
  }

  /**
   * macro 1 の後に状態と重みを書き、新しいモデルで続きから macro 2 を学ぶと、途切れずに 2 つ学んだときと、ステップの記録も重みも
   * ビットで同じ。既にあるディレクトリには書かず、形の違うネットワークでは続けない。
   */
  @Test(groups = "native", timeOut = 1_800_000)
  public void resumingMatchesAnUnbrokenRun() throws IOException {
    SelfPlayLearner.Settings settings = settings(4, 0.01, 1e-4);
    Path middle = root.resolve("middle");
    Path state = root.resolve("state");
    SelfPlayLearner.Macro unbrokenSecond;
    Weights unbroken;
    try (Model model = model(start);
        SelfPlayLearner learner = new SelfPlayLearner(model, NetworkDevices.of(Device.cpu()), null, settings)) {
      learner.learn(batch, 11);
      learner.save(state);
      assertThrows(FileAlreadyExistsException.class, () -> learner.save(state));
      ClassicCheckpoint.write(middle, network(model), 1);
      unbrokenSecond = learner.learn(batch, 12);
      unbroken = Weights.of(network(model));
    }
    assertFalse(Files.exists(root.resolve("state.tmp")), "the temporary directory is left");
    assertNotEquals(unbrokenSecond.rate().action(), KlLearningRate.Action.WARMUP);
    try (Model model = model(middle);
        SelfPlayLearner learner =
            SelfPlayLearner.resume(model, NetworkDevices.of(Device.cpu()), null, settings, state)) {
      assertEquals(learner.macros(), 1);
      assertEquals(learner.actorRate().state().macros(), 1);
      SelfPlayLearner.Macro resumedSecond = learner.learn(batch, 12);
      assertEquals(withoutSeconds(resumedSecond), withoutSeconds(unbrokenSecond));
      assertTrue(Weights.of(network(model)).same(unbroken), "the resumed weights differ");
      assertEquals(learner.macros(), 2);
    }
    Path major = root.resolve("major");
    RandomCheckpoints.write(major, Architecture.MAJOR, VARIANT, 7);
    try (Model model = model(major)) {
      assertThrows(
          IllegalArgumentException.class,
          () -> SelfPlayLearner.resume(model, NetworkDevices.of(Device.cpu()), null, settings, state));
    }
  }

  /**
   * KL の上限をほぼ 0 にすると、更新の前の KL が上限を超えたステップは方策の更新を捨て、価値だけを更新する。全てのステップで捨てれば、
   * 方策の重みはビットで変わらず、学習率の制御もその macro を受け入れない。上限が大きければ、全てのステップが方策を更新する。
   */
  @Test(groups = "native", timeOut = 1_800_000)
  public void stepsOverTheKlLimitDropOnlyThePolicyUpdate() throws IOException {
    double limit = 1e-30;
    try (Model model = model(start);
        SelfPlayLearner learner =
            new SelfPlayLearner(model, NetworkDevices.of(Device.cpu()), null, settings(2, limit, 1e-3))) {
      Weights initial = Weights.of(network(model));
      double rate = learner.actorRate().rate();
      SelfPlayLearner.Macro macro = learner.learn(batch, 21);
      Weights after = Weights.of(network(model));
      int accepted = 0;
      for (SelfPlayLearner.Step step : macro.steps()) {
        assertEquals(step.actorUpdated(), step.actorRows() > 0 && step.kl() <= limit, step.toString());
        assertEquals(Double.isNaN(step.actorNorm()), !step.actorUpdated(), step.toString());
        accepted += step.actorUpdated() ? 1 : 0;
      }
      LOG.info("KL guard steps: {}", macro.steps());
      assertTrue(accepted < macro.steps().size(), "no step was rejected");
      assertEquals(learner.actorSteps(), accepted);
      assertEquals(learner.valueSteps(), macro.steps().size());
      assertFalse(after.sameGroup(initial, true), "the value moved");
      if (accepted == 0) {
        assertTrue(after.sameGroup(initial, false), "the policy must not move");
        assertEquals(macro.rate().action(), KlLearningRate.Action.REJECTED);
        assertEquals(learner.actorRate().rate(), rate);
        assertEquals(learner.actorRate().state().macros(), 0);
      }
    }
    try (Model model = model(start);
        SelfPlayLearner learner =
            new SelfPlayLearner(model, NetworkDevices.of(Device.cpu()), null, settings(2, 1e9, 1e-3))) {
      SelfPlayLearner.Macro macro = learner.learn(batch, 21);
      assertEquals(macro.actorSteps(), 2);
      assertEquals(learner.actorRate().state().macros(), 1);
    }
  }

  /** 2 つの CPU のレーンで学ぶと、1 つのデバイスと許容誤差の内で同じステップを踏む（更新の前の指標、勾配のノルム、重みの動きを比べる）。 */
  @Test(groups = "native", timeOut = 1_800_000)
  public void twoLanesTakeTheStepsOfOneDevice() throws IOException {
    SelfPlayLearner.Settings settings = settings(3, 0.01, 1e-4);
    SelfPlayLearner.Macro[] macros = new SelfPlayLearner.Macro[2];
    Weights[] moved = new Weights[2];
    Weights initial = null;
    NetworkDevices[] devices = {NetworkDevices.of(Device.cpu()), NetworkDevices.of(Device.cpu(), Device.cpu())};
    for (int run = 0; run < 2; run++) {
      try (Model model = model(start);
          SelfPlayLearner learner = new SelfPlayLearner(model, devices[run], null, settings)) {
        initial = Weights.of(network(model));
        macros[run] = learner.learn(batch, 31);
        moved[run] = Weights.of(network(model));
      }
    }
    SelfPlayLearner.Macro one = macros[0];
    SelfPlayLearner.Macro two = macros[1];
    LOG.info("One lane: {}", one.steps());
    LOG.info("Two lanes: {}", two.steps());
    assertEquals(two.actorRows(), one.actorRows());
    assertEquals(two.credit(), one.credit(), 1e-6);
    assertEquals(two.advantage(), one.advantage(), 1e-6);
    for (int index = 0; index < one.steps().size(); index++) {
      SelfPlayLearner.Step expected = one.steps().get(index);
      SelfPlayLearner.Step actual = two.steps().get(index);
      assertEquals(actual.rows(), expected.rows());
      assertEquals(actual.actorRows(), expected.actorRows());
      assertEquals(actual.actorUpdated(), expected.actorUpdated());
      if (index == 0) {
        assertEquals(actual.kl(), expected.kl(), 1e-7);
        close(actual.surrogate(), expected.surrogate(), 1e-4, "surrogate");
        close(actual.entropy(), expected.entropy(), 1e-5, "entropy");
        close(actual.valueLoss(), expected.valueLoss(), 1e-5, "value loss");
        close(actual.actorNorm(), expected.actorNorm(), 1e-4, "actor norm");
        close(actual.valueNorm(), expected.valueNorm(), 1e-4, "value norm");
      } else {
        close(actual.valueLoss(), expected.valueLoss(), 1e-3, "value loss of step " + index);
        close(actual.entropy(), expected.entropy(), 1e-3, "entropy of step " + index);
      }
    }
    for (boolean value : new boolean[] {false, true}) {
      double difference = 0;
      double movement = 0;
      for (int tensor = 0; tensor < initial.names().size(); tensor++) {
        if (ClassicNetwork.valueParameter(initial.names().get(tensor)) != value) {
          continue;
        }
        float[] from = initial.values().get(tensor);
        float[] a = moved[0].values().get(tensor);
        float[] b = moved[1].values().get(tensor);
        for (int index = 0; index < from.length; index++) {
          difference += Math.abs((double) a[index] - b[index]);
          movement += Math.abs((double) a[index] - from[index]);
        }
      }
      LOG.info("{} weights: movement {} difference {}", value ? "value" : "policy", movement, difference);
      assertTrue(movement > 0, "nothing moved");
      assertTrue(difference <= 1e-3 * movement, (value ? "value" : "policy") + " " + difference + " of " + movement);
    }
  }

  /**
   * 1 つのバッチを覚え切る。選んだ手の補正の下限と V-trace の割合がどちらも 1 なら、価値の目標はどれも局末の価値になって動かない。そのため、同じ 96 判断の macro を
   * 繰り返すと、価値の交差エントロピーは教師の分布のエントロピー（下限）へ近づく。方策の学習率はほぼ 0 にする。
   */
  @Test(groups = "native", timeOut = 1_800_000)
  public void aSmallNetworkMemorizesOneBatch() throws IOException {
    TrajectoryBatch small = Trajectories.play(VARIANT, 9, 2, 48);
    SelfPlayLearner.Settings settings =
        new SelfPlayLearner.Settings(
            new SelfPlayLearner.Ppo(MEMORIZE_STEPS, 0.2, 1.0, 0, 1e9, 0, 100),
            new KlLearningRate.Settings(1e-9, 1e-9, 1e-9, 1, 1, 1.25, 1.25, 1.02, 0, 0.5),
            new SelfPlayLearner.ValueRate(3e-3, 0, 100),
            new SelfPlayLearner.VTrace(1),
            new SelfPlayLearner.Batches(64, 16_384));
    try (Model model = model(start);
        SelfPlayLearner learner = new SelfPlayLearner(model, NetworkDevices.of(Device.cpu()), null, settings)) {
      double floor = targetEntropy(network(model), small);
      double first = Double.NaN;
      double last = Double.NaN;
      for (int macro = 0; macro < MEMORIZE_MACROS; macro++) {
        SelfPlayLearner.Macro learned = learner.learn(small, macro);
        double loss = 0;
        for (SelfPlayLearner.Step step : learned.steps()) {
          loss += step.valueLoss() * step.rows();
        }
        last = loss / small.size();
        first = macro == 0 ? last : first;
        if (macro % 10 == 0 || macro == MEMORIZE_MACROS - 1) {
          LOG.info("Memorize macro {}: value loss {} (floor {})", macro, last, floor);
        }
      }
      assertTrue(first - floor > 0.5, "the start is already at the floor: " + first + " vs " + floor);
      assertTrue(last - floor <= 0.1 * (first - floor), "value loss " + first + " -> " + last + " (floor " + floor + ")");
    }
  }

  /** 局末の価値を HL-Gauss の分布にしたときの、エントロピーの平均（価値の交差エントロピーの下限）。 */
  private static double targetEntropy(ClassicNetwork network, TrajectoryBatch batch) {
    float[] ends = new float[batch.size()];
    for (int game = 0; game < batch.games().size(); game++) {
      GameTrajectory trajectory = batch.games().get(game);
      for (int decision = 0; decision < trajectory.size(); decision++) {
        ends[batch.start(game) + decision] =
            (float) trajectory.endValue(trajectory.hand(decision), trajectory.seat(decision), network.utility());
      }
    }
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      float[] targets = network.support().targets(manager.create(ends)).toFloatArray();
      double entropy = 0;
      for (float probability : targets) {
        entropy -= probability > 0 ? probability * Math.log(probability) : 0;
      }
      return entropy / ends.length;
    }
  }

  /** ルールの違う軌跡は拒む。判断の無い軌跡では何も更新せず、学習率の制御もその macro を受け入れない。 */
  @Test(groups = "native", timeOut = 600_000)
  public void emptyAndForeignBatches() throws IOException {
    try (Model model = model(start);
        SelfPlayLearner learner =
            new SelfPlayLearner(model, NetworkDevices.of(Device.cpu()), null, settings(4, 0.01, 1e-4))) {
      Weights initial = Weights.of(network(model));
      assertThrows(IllegalArgumentException.class, () -> learner.learn(TrajectoryBatch.of(Variant.TENHOU_3P, List.of()), 1));
      SelfPlayLearner.Macro empty = learner.learn(TrajectoryBatch.of(VARIANT, List.of()), 1);
      assertEquals(empty.decisions(), 0);
      assertEquals(empty.steps(), List.of());
      assertEquals(empty.rate().action(), KlLearningRate.Action.REJECTED);
      assertTrue(Weights.of(network(model)).same(initial));
      assertEquals(learner.actorSteps(), 0);
      assertEquals(learner.valueSteps(), 0);
    }
  }

  /** ステップ数・KL の上限・方策の学習率を指定した設定（ウォームアップ 1 回、価値の学習率 1e-4、1 バッチ 64 行）。 */
  static SelfPlayLearner.Settings settings(int steps, double klLimit, double actorRate) {
    return new SelfPlayLearner.Settings(
        new SelfPlayLearner.Ppo(steps, 0.2, 0.10, 0.001, klLimit, 1e-4, 0.5),
        new KlLearningRate.Settings(actorRate, actorRate / 100, actorRate * 10, 1, 1, 1.25, 1.25, 1.02, 0, 0.5),
        new SelfPlayLearner.ValueRate(1e-4, 1e-4, 0.5),
        new SelfPlayLearner.VTrace(0.95),
        new SelfPlayLearner.Batches(64, 16_384));
  }

  /** 方策の損失に入れる判断（合法手が 2 つ以上で、選び替えで局の進みが変わりえた）の数。 */
  static int actorRows(TrajectoryBatch batch) {
    Observation reader = new Observation(batch.variant());
    int rows = 0;
    for (GameTrajectory game : batch.games()) {
      for (int decision = 0; decision < game.size(); decision++) {
        rows += game.causal(decision) && game.bind(decision, reader).legalActions() > 1 ? 1 : 0;
      }
    }
    return rows;
  }

  static Model model(Path checkpoint) throws IOException {
    Model model = Model.newInstance("learner-test", Device.cpu(), "PyTorch");
    try {
      model.setBlock(ClassicCheckpoint.read(checkpoint, VARIANT, model.getNDManager()));
      return model;
    } catch (IOException | RuntimeException | Error failure) {
      model.close();
      throw failure;
    }
  }

  static ClassicNetwork network(Model model) {
    return (ClassicNetwork) model.getBlock();
  }

  private static SelfPlayLearner.Macro withoutSeconds(SelfPlayLearner.Macro macro) {
    return new SelfPlayLearner.Macro(
        macro.games(), macro.decisions(), macro.actorRows(), macro.steps(), macro.kl(), macro.credit(),
        macro.advantage(), macro.rate(), 0, 0);
  }

  private static void close(double actual, double expected, double relative, String name) {
    assertTrue(
        Math.abs(actual - expected) <= relative * Math.max(1e-6, Math.abs(expected)),
        name + ": " + actual + " vs " + expected);
  }
}
