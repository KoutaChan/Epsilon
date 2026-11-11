package com.epsilon.classic.model;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.ClassicHeader;
import com.epsilon.classic.input.Buckets;
import com.epsilon.classic.input.Buckets.Bucket;
import com.epsilon.classic.input.ClassicExtras;
import com.epsilon.classic.input.ClassicInputs;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.input.ClassicTokenizer;
import com.epsilon.classic.input.ClassicVocabulary.ActionType;
import com.epsilon.classic.input.VocabularyCodec;
import com.epsilon.mahjong.rules.RankUtility;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 重みを小さな乱数にした Nano と Major を、ランダムな自己対局で集めた判断（ロン・ツモ・九種九牌・鳴き・槓・立直を含む）にかけて
 * CPU で動かし、次のことを確かめる。合法手の確率の和は 1 で、合法手の無い位置は 0 になる。貪欲に選んだ手は、スコアからグラフを
 * たどって選んだ手と同じになる。1 行だけを最小の区分に書いても、最大の区分にまとめて書いても、出力は許容誤差の範囲で一致する。
 * checkpoint に書いて読み直しても出力は変わらない。BF16 の出力は FP32 と許容誤差の範囲で一致する（CPU が BF16 を扱えるときだけ）。
 */
public class ClassicNetworkTest {

  private static final Logger LOG = LoggerFactory.getLogger(ClassicNetworkTest.class);

  /** どの判断も入る区分。 */
  private static final Bucket BUCKET = Buckets.required(ClassicSchema.MAX_LEGAL, ClassicSchema.MAX_TRANSITIONS);

  private static final int PER_KIND = 4;
  private static final int STEPS = 400_000;

  private NDManager manager;
  private PinnedUploads uploads;
  private SampleDecisions samples;
  private final Map<Architecture, Fixture> fixtures = new EnumMap<>(Architecture.class);

  /** アーキテクチャごとのネットワークと、全ての判断を書いた入力。 */
  private record Fixture(ClassicNetwork network, ClassicInputs inputs, float[][] priors) {}

  @BeforeClass(groups = "native")
  public void build() {
    manager = NDManager.newBaseManager(Device.cpu(), "PyTorch");
    uploads = PinnedUploads.on(manager);
    samples = new SampleDecisions(41, PER_KIND, STEPS);
    LOG.info("Sampled {} decisions: {}", samples.size(), samples.counts());
    for (Architecture architecture : Architecture.values()) {
      RankUtility utility = architecture == Architecture.NANO ? RankUtility.TENHOU : RankUtility.AVERAGE_RANK;
      ClassicNetwork network = new ClassicNetwork(architecture, SampleDecisions.VARIANT, utility);
      network.initialize(manager);
      randomize(network, 7L + architecture.ordinal());
      ClassicInputs inputs = new ClassicInputs(ClassicSchema.of(architecture), BUCKET, samples.size());
      float[][] priors = new float[samples.size()][];
      Random random = new Random(13);
      ClassicTokenizer tokenizer = new ClassicTokenizer(architecture, SampleDecisions.VARIANT);
      Observation o = new Observation(SampleDecisions.VARIANT);
      for (int row = 0; row < samples.size(); row++) {
        priors[row] = row % 2 == 0 ? null : prior(random);
        write(tokenizer, samples.bind(o, row), priors[row], inputs, row);
      }
      fixtures.put(architecture, new Fixture(network, inputs, priors));
    }
  }

  @AfterClass(groups = "native", alwaysRun = true)
  public void close() {
    if (manager != null) {
      manager.close();
    }
  }

  @Test(groups = "native")
  public void probabilitiesSumToOneOverLegalActionsAndIllegalSlotsGetZero() {
    Observation o = new Observation(SampleDecisions.VARIANT);
    for (Fixture fixture : fixtures.values()) {
      Result result = run(fixture.network(), fixture.inputs(), samples.size(), Precision.FLOAT32);
      for (int row = 0; row < samples.size(); row++) {
        int legal = samples.bind(o, row).legalActions();
        double sum = 0;
        for (int slot = 0; slot < BUCKET.actions(); slot++) {
          float probability = result.probabilities()[row * BUCKET.actions() + slot];
          if (slot < legal) {
            assertTrue(probability >= 0 && probability <= 1, "probability " + probability);
            sum += probability;
          } else {
            assertEquals(probability, 0f, "illegal slot " + slot + " of row " + row);
          }
        }
        assertEquals(sum, 1.0, 1e-4, "row " + row);
        float utility = fixture.network().support().decode(result.values(), row * HlGauss.BINS);
        assertTrue(Float.isFinite(utility) && utility >= -1.5f && utility <= 1.5f, "utility " + utility);
      }
    }
  }

  @Test(groups = "native")
  public void greedyFollowsThePolicyGraph() {
    Observation o = new Observation(SampleDecisions.VARIANT);
    PolicyGraph.Greedy greedy = new PolicyGraph.Greedy();
    for (Fixture fixture : fixtures.values()) {
      Result result = run(fixture.network(), fixture.inputs(), samples.size(), Precision.FLOAT32);
      int actions = BUCKET.actions();
      for (int row = 0; row < samples.size(); row++) {
        int chosen = greedy.select(fixture.inputs(), row, result.probabilities(), row * actions);
        int expected = graphChoice(samples.bind(o, row), result, row);
        assertEquals(chosen, expected, fixture.network().architecture() + " row " + row);
      }
    }
  }

  @Test(groups = "native")
  public void aRowAloneInItsSmallestBucketMatchesTheBatch() {
    Observation o = new Observation(SampleDecisions.VARIANT);
    for (Map.Entry<Architecture, Fixture> entry : fixtures.entrySet()) {
      Fixture fixture = entry.getValue();
      Result batch = run(fixture.network(), fixture.inputs(), samples.size(), Precision.FLOAT32);
      ClassicTokenizer tokenizer = new ClassicTokenizer(entry.getKey(), SampleDecisions.VARIANT);
      int actions = BUCKET.actions();
      for (int row = 0; row < samples.size(); row++) {
        Bucket smallest = Buckets.required(samples.bind(o, row));
        ClassicInputs single = new ClassicInputs(ClassicSchema.of(entry.getKey()), smallest, 1);
        write(tokenizer, o, fixture.priors()[row], single, 0);
        Result alone = run(fixture.network(), single, 1, Precision.FLOAT32);
        for (int slot = 0; slot < smallest.actions(); slot++) {
          assertEquals(
              alone.probabilities()[slot], batch.probabilities()[row * actions + slot], 1e-4f, smallest + " row " + row);
        }
        for (int bin = 0; bin < HlGauss.BINS; bin++) {
          assertEquals(alone.values()[bin], batch.values()[row * HlGauss.BINS + bin], 1e-3f, "row " + row);
        }
      }
    }
  }

  @Test(groups = "native")
  public void aCheckpointRoundTripKeepsTheOutputs() throws IOException {
    Path root = Files.createTempDirectory("classic-checkpoint");
    try {
      for (Fixture fixture : fixtures.values()) {
        ClassicNetwork network = fixture.network();
        Path directory = root.resolve(network.architecture().name().toLowerCase(Locale.ROOT));
        ClassicCheckpoint.write(directory, network, 17);
        ClassicHeader header = ClassicCheckpoint.header(directory);
        assertEquals(header, network.header(17));
        ClassicNetwork read = ClassicCheckpoint.read(directory, SampleDecisions.VARIANT, manager);
        assertEquals(read.getParameters().keys(), network.getParameters().keys());
        for (Pair<String, Parameter> pair : network.getParameters()) {
          assertEquals(
              read.getParameters().get(pair.getKey()).getArray().toFloatArray(),
              pair.getValue().getArray().toFloatArray(),
              pair.getKey());
        }
        // 重みのビットが同じでも、別に確保した配列では CPU の行列積の最後の桁が変わることがある（メモリのアラインメントが違うため）
        Result written = run(network, fixture.inputs(), samples.size(), Precision.FLOAT32);
        Result loaded = run(read, fixture.inputs(), samples.size(), Precision.FLOAT32);
        assertEquals(loaded.probabilities(), written.probabilities(), 1e-4f);
        assertEquals(loaded.values(), written.values(), 1e-4f);
        assertEquals(loaded.alternatives(), written.alternatives(), 1e-4f);
      }
    } finally {
      try (Stream<Path> files = Files.walk(root)) {
        for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
          Files.delete(file);
        }
      }
    }
  }

  @Test(groups = "native")
  public void bfloat16AgreesWithFloat32() {
    try (NDManager probe = manager.newSubManager()) {
      NDArray ones = probe.ones(new Shape(2, 2), DataType.BFLOAT16);
      ones.matMul(ones).toType(DataType.FLOAT32, false).toFloatArray();
    } catch (RuntimeException unavailable) {
      throw new SkipException("BF16 is not available on this CPU: " + unavailable.getMessage());
    }
    for (Fixture fixture : fixtures.values()) {
      Result full = run(fixture.network(), fixture.inputs(), samples.size(), Precision.FLOAT32);
      Result half = run(fixture.network(), fixture.inputs(), samples.size(), Precision.BFLOAT16);
      double worst = 0;
      double total = 0;
      for (int index = 0; index < full.probabilities().length; index++) {
        double difference = Math.abs(full.probabilities()[index] - half.probabilities()[index]);
        worst = Math.max(worst, difference);
        total += difference;
      }
      double mean = total / (samples.size() * BUCKET.actions());
      LOG.info("{} BF16 probabilities: max |diff| {} mean |diff| {}", fixture.network().architecture(), worst, mean);
      assertTrue(worst < 0.1, "max probability difference " + worst);
      assertTrue(mean < 0.01, "mean probability difference " + mean);
      for (int row = 0; row < samples.size(); row++) {
        float a = fixture.network().support().decode(full.values(), row * HlGauss.BINS);
        float b = fixture.network().support().decode(half.values(), row * HlGauss.BINS);
        assertEquals(b, a, 0.05f, "utility of row " + row);
      }
    }
  }

  @Test(groups = "native")
  public void parametersKeepTheConvertedShapes() {
    for (Fixture fixture : fixtures.values()) {
      ClassicNetwork network = fixture.network();
      boolean major = network.architecture() == Architecture.MAJOR;
      int hidden = network.architecture().hidden();
      int context = major ? 128 : 64;
      ClassicSchema schema = ClassicSchema.of(network.architecture());
      assertShape(network, "01stateEncoder_01stateFeatureEmbedding_embedding", schema.stateDictionary(), 8);
      assertShape(network, "01stateEncoder_02canonicalTileEmbedding_embedding", 35, hidden);
      assertShape(network, "04policyHead_01actionFeatureEmbedding_embedding", schema.actionDictionary(), 8);
      assertShape(network, "04policyHead_02transitionTileRelationEmbedding_embedding", 1601, 32 + context);
      assertShape(
          network, "01stateEncoder_06tileRelationEncoder_01block0_04relationKeyEmbedding_embedding", 23, major ? 128 : 64);
      if (major) {
        assertShape(network, "01stateEncoder_12publicHistoryEncoder_03relationTable_embedding", 76, 4 + context);
        assertShape(network, "04policyHead_23ronGate_02directFeatureProjection_weight", 256, 6);
        assertShape(network, "04policyHead_21alternativeHidden_weight", hidden, 2 * hidden);
        assertShape(network, "07valueHead_weight", HlGauss.BINS, hidden);
      } else {
        assertShape(network, "04policyHead_03transitionWaitYakuProjection_weight", 16, 42);
        assertShape(network, "04policyHead_10transitionWaitKeyValueProjection_weight", 32 + context, 38);
        assertShape(network, "04policyHead_21alternativeHidden_weight", hidden, 4 * hidden);
        assertShape(network, "06valueHead_weight", HlGauss.BINS, hidden);
      }
    }
  }

  /** ホストに読み出した、1 回の順伝播の出力。 */
  private record Result(float[] probabilities, float[] values, float[] alternatives, float[] actions, float[] riichi) {}

  private Result run(ClassicNetwork network, ClassicInputs inputs, int rows, Precision precision) {
    try (NDManager working = manager.newSubManager()) {
      ClassicBatch batch = ClassicBatch.send(working, uploads, inputs, rows);
      ClassicNetwork.Output output =
          network.forward(new ParameterStore(working, false), batch, precision, false, ClassicNetwork.Heads.BOTH);
      NDArray logProbabilities = PolicyGraph.logProbabilities(output.policy(), batch);
      float[] probabilities = PolicyGraph.probabilities(logProbabilities, batch).toFloatArray();
      return new Result(
          probabilities,
          output.value().toFloatArray(),
          floats(output.policy().alternatives()),
          floats(output.policy().actions()),
          floats(output.policy().riichi()));
    }
  }

  private static float[] floats(NDArray array) {
    return array.toType(DataType.FLOAT32, false).toFloatArray();
  }

  /** グラフのノードごとにロジットの大きい側をたどって選んだ手。確率を使わずにスコアだけから、別の道筋で求める。 */
  private static int graphChoice(Observation o, Result result, int row) {
    int legal = o.legalActions();
    int actions = BUCKET.actions();
    float[] scores = result.actions();
    int base = row * actions;
    boolean response = false;
    for (int action = 0; action < legal; action++) {
      response |= SampleDecisions.type(o, action).group().response();
    }
    if (response) {
      if (has(o, ActionType.RON_AGARI)
          && (!hasOther(o, ActionType.RON_AGARI) || alternative(result, row, Alternative.RON) >= 0)) {
        return best(o, scores, base, ActionType.RON_AGARI);
      }
      boolean meld = has(o, ActionType.CHI) || has(o, ActionType.PON) || has(o, ActionType.DAIMINKAN);
      if (meld && (!has(o, ActionType.PASS) || alternative(result, row, Alternative.CALL) >= 0)) {
        return best(o, scores, base, heaviest(o, result, row, Alternative.CHI, Alternative.PON, Alternative.DAIMINKAN));
      }
      return best(o, scores, base, ActionType.PASS);
    }
    if (has(o, ActionType.TSUMO_AGARI)
        && (!hasOther(o, ActionType.TSUMO_AGARI) || alternative(result, row, Alternative.TSUMO) >= 0)) {
      return best(o, scores, base, ActionType.TSUMO_AGARI);
    }
    if (has(o, ActionType.KYUSHU_KYUHAI)
        && (!hasOther(o, ActionType.TSUMO_AGARI, ActionType.KYUSHU_KYUHAI)
            || alternative(result, row, Alternative.KYUSHU) >= 0)) {
      return best(o, scores, base, ActionType.KYUSHU_KYUHAI);
    }
    boolean kan = has(o, ActionType.ANKAN) || has(o, ActionType.KAKAN);
    boolean discard = has(o, ActionType.DAHAI) || has(o, ActionType.RIICHI_DAHAI);
    if (kan && (!discard || alternative(result, row, Alternative.KAN) >= 0)) {
      return best(o, scores, base, heaviest(o, result, row, Alternative.ANKAN, Alternative.KAKAN));
    }
    int identity = -1;
    for (int action = 0; action < legal; action++) {
      ActionType type = SampleDecisions.type(o, action);
      boolean discarding = type == ActionType.DAHAI || type == ActionType.RIICHI_DAHAI;
      if (discarding && (identity < 0 || scores[base + action] > scores[base + identity])) {
        identity = action;
      }
    }
    int key = VocabularyCodec.discardIdentity(o.action(identity));
    int dama = -1;
    int riichi = -1;
    for (int action = 0; action < legal; action++) {
      if (VocabularyCodec.discardIdentity(o.action(action)) == key) {
        if (SampleDecisions.type(o, action) == ActionType.RIICHI_DAHAI) {
          riichi = action;
        } else {
          dama = action;
        }
      }
    }
    if (riichi >= 0 && (dama < 0 || result.riichi()[base + riichi] >= 0)) {
      return riichi;
    }
    return dama;
  }

  private static float alternative(Result result, int row, Alternative alternative) {
    return result.alternatives()[row * Alternative.values().length + alternative.ordinal()];
  }

  /** 合法手のある種類のうち、種類を選ぶ分岐のスコアが最大のもの（同じなら前の方）。 */
  private static ActionType heaviest(Observation o, Result result, int row, Alternative... kinds) {
    ActionType heaviest = null;
    float score = Float.NEGATIVE_INFINITY;
    for (Alternative kind : kinds) {
      if (has(o, kind.type()) && alternative(result, row, kind) > score) {
        score = alternative(result, row, kind);
        heaviest = kind.type();
      }
    }
    return heaviest;
  }

  private static boolean has(Observation o, ActionType type) {
    for (int action = 0; action < o.legalActions(); action++) {
      if (SampleDecisions.type(o, action) == type) {
        return true;
      }
    }
    return false;
  }

  /** 渡した種類のどれでもない合法手があるか。 */
  private static boolean hasOther(Observation o, ActionType... excluded) {
    for (int action = 0; action < o.legalActions(); action++) {
      boolean other = true;
      for (ActionType type : excluded) {
        other &= SampleDecisions.type(o, action) != type;
      }
      if (other) {
        return true;
      }
    }
    return false;
  }

  /** その種類の合法手のうち、スコアが最大の手（同じなら前の方）。 */
  private static int best(Observation o, float[] scores, int base, ActionType type) {
    int best = -1;
    for (int action = 0; action < o.legalActions(); action++) {
      if (SampleDecisions.type(o, action) == type && (best < 0 || scores[base + action] > scores[base + best])) {
        best = action;
      }
    }
    assertFalse(best < 0, "no " + type);
    return best;
  }

  private static void write(ClassicTokenizer tokenizer, Observation o, float[] prior, ClassicInputs into, int row) {
    if (prior == null) {
      tokenizer.write(o, ClassicExtras.empty(), into, row);
    } else {
      tokenizer.write(o, ClassicExtras.empty(), prior, into, row);
    }
  }

  /** 席 × 順位の周辺確率（席ごとに和が 1）。 */
  private static float[] prior(Random random) {
    int seats = ClassicSchema.SEATS;
    float[] prior = new float[seats * seats];
    for (int seat = 0; seat < seats; seat++) {
      float sum = 0;
      for (int rank = 0; rank < seats; rank++) {
        prior[seat * seats + rank] = 0.05f + random.nextFloat();
        sum += prior[seat * seats + rank];
      }
      for (int rank = 0; rank < seats; rank++) {
        prior[seat * seats + rank] /= sum;
      }
    }
    return prior;
  }

  /**
   * パラメーターをシードから決まる乱数で埋める（0 で初期化する層にも値が入るように）。重みは N(0, 1/入力の幅)、LayerNorm のゲインは
   * 1 ＋ N(0, 0.1²)、ほかの 1 次元のパラメーターは N(0, 0.1²)。
   */
  private static void randomize(ClassicNetwork network, long seed) {
    Random random = new Random(seed);
    for (Pair<String, Parameter> pair : network.getParameters()) {
      NDArray array = pair.getValue().getArray();
      Shape shape = array.getShape();
      float[] values = new float[Math.toIntExact(shape.size())];
      boolean gain = pair.getKey().endsWith("_gamma");
      double scale = shape.dimension() == 1 ? 0.1 : 1 / Math.sqrt(shape.get(shape.dimension() - 1));
      for (int index = 0; index < values.length; index++) {
        values[index] = (float) ((gain ? 1 : 0) + scale * random.nextGaussian());
      }
      array.set(values);
    }
  }

  private static void assertShape(ClassicNetwork network, String name, long... dimensions) {
    Parameter parameter = network.getParameters().get(name);
    assertTrue(parameter != null, "no parameter " + name + " in " + network.getParameters().keys());
    assertEquals(parameter.getArray().getShape(), new Shape(dimensions), name);
  }
}
