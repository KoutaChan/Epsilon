package com.epsilon.classic.learn;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.settings.Settings;
import com.epsilon.classic.app.ClassicSettings;
import com.epsilon.classic.input.VocabularyCodec;
import com.epsilon.classic.learn.Exploration.Kind;
import com.epsilon.classic.learn.UncertaintyCdf.Key;
import com.epsilon.classic.model.SampleDecisions;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 振る舞いの方策 μ を確かめる。どの合法手の確率も下限以上で、和は 1。倍率 0 なら一様分布を混ぜず、下限の式だけを使う。打牌だけの判断
 * と、ロンか見送りかの判断は、節点ごとに一様分布を混ぜる式のとおりになる。倍率は迷いの百分位から [最小, 最大] の中に決まる。ほかに、
 * 迷いの式、判断の種類、百分位を引く分布の選び方（種類 → 手番か応答か → 全体）、数える表の往復とスレッドごとの足し合わせも確かめる。
 */
public class ExplorationTest {

  private static final Variant VARIANT = SampleDecisions.VARIANT;
  private static final Exploration.Settings SETTINGS = Settings.bind(ClassicSettings.class, List.of()).value().explore();
  private static final double FLOOR = SETTINGS.leafFloor();

  private final List<Observation> decisions = new ArrayList<>();

  @BeforeClass
  public void collect() {
    SampleDecisions samples = new SampleDecisions(41, 8, 400_000);
    for (int index = 0; index < samples.size(); index++) {
      Observation o = samples.bind(new Observation(VARIANT), index);
      if (o.legalActions() > 1) {
        decisions.add(o);
      }
    }
    assertTrue(decisions.size() > 30, "decisions " + decisions.size());
  }

  /** 1 つの手だけに確率がある方策で、倍率や節点の割合を 0 にしても、どの合法手の確率も下限を下回らず、和は 1。 */
  @Test
  public void everyLegalActionKeepsTheFloorAndTheSumIsOne() {
    Exploration.Settings unmixed = new Exploration.Settings(0, 0, FLOOR, 0.1, 5, 4.5, 0.5);
    Random random = new Random(1);
    float[] behaviour = new float[64];
    List<Exploration> explorations =
        List.of(new Exploration(SETTINGS, UncertaintyCdf.empty()), new Exploration(unmixed, UncertaintyCdf.empty()));
    for (Exploration exploration : explorations) {
      for (Observation o : decisions) {
        for (float[] policy : policies(o.legalActions(), random)) {
          for (double scale : new double[] {0, 0.1, 1, SETTINGS.maximumScale()}) {
            exploration.mix(o, policy, scale, behaviour);
            double sum = 0;
            for (int action = 0; action < o.legalActions(); action++) {
              assertTrue(behaviour[action] >= FLOOR * (1 - 1e-6), "action " + action + " " + behaviour[action]);
              sum += behaviour[action];
            }
            assertEquals(sum, 1.0, 1e-5);
          }
        }
      }
    }
  }

  /** 倍率 0: μ(a) = f + (1 − n·f)·π(a)（π は和で割り、0 の手は 1e-8 にしてから割り直す）。 */
  @Test
  public void scaleZeroOnlyAppliesTheFloor() {
    Exploration exploration = new Exploration(SETTINGS, UncertaintyCdf.empty());
    Random random = new Random(2);
    float[] behaviour = new float[64];
    for (Observation o : decisions) {
      int legal = o.legalActions();
      for (float[] policy : policies(legal, random)) {
        exploration.mix(o, policy, 0, behaviour);
        double[] normalized = normalized(policy, legal);
        for (int action = 0; action < legal; action++) {
          assertEquals(behaviour[action], FLOOR + (1 - legal * FLOOR) * normalized[action], 1e-6);
        }
      }
    }
  }

  /** 打牌だけの手番: 同じ牌の打牌の組 g ごとに μ_グラフ(a) = ((1 − ε)·M_g + ε/K)·π(a)/M_g（K は組の数、ε は打牌の割合 × 倍率）。 */
  @Test
  public void aDiscardOnlyTurnMixesTheTileChoiceUniformly() {
    Exploration exploration = new Exploration(SETTINGS, UncertaintyCdf.empty());
    Random random = new Random(3);
    float[] behaviour = new float[64];
    int checked = 0;
    for (Observation o : decisions) {
      if (Exploration.kind(o) != Kind.DISCARD || response(o)) {
        continue;
      }
      int legal = o.legalActions();
      Map<Integer, List<Integer>> groups = new HashMap<>();
      for (int action = 0; action < legal; action++) {
        assertEquals(o.action(action).type(), Action.Type.DAHAI);
        groups.computeIfAbsent(VocabularyCodec.discardIdentity(o.action(action)), key -> new ArrayList<>()).add(action);
      }
      for (double scale : new double[] {1, SETTINGS.maximumScale()}) {
        double exploring = SETTINGS.discardMass() * scale;
        float[] policy = policies(legal, random).getFirst();
        double[] normalized = normalized(policy, legal);
        exploration.mix(o, policy, scale, behaviour);
        for (List<Integer> group : groups.values()) {
          double mass = 0;
          for (int action : group) {
            mass += normalized[action];
          }
          for (int action : group) {
            double graph = ((1 - exploring) * mass + exploring / groups.size()) * normalized[action] / mass;
            assertEquals(behaviour[action], FLOOR + (1 - legal * FLOOR) * graph, 1e-5);
          }
        }
      }
      checked++;
    }
    assertTrue(checked > 0, "no discard-only turn");
  }

  /** ロンか見送りかだけの応答: μ_グラフ(ロン) = (1 − ε)·π(ロン) + ε/2（ε は鳴きなどの割合 × 倍率）。 */
  @Test
  public void aRonOrPassResponseMixesTheBinaryChoice() {
    Exploration exploration = new Exploration(SETTINGS, UncertaintyCdf.empty());
    Random random = new Random(4);
    float[] behaviour = new float[64];
    int checked = 0;
    for (Observation o : decisions) {
      if (o.legalActions() != 2 || !response(o)) {
        continue;
      }
      int ron = -1;
      for (int action = 0; action < 2; action++) {
        if (o.action(action).type() == Action.Type.RON_AGARI) {
          ron = action;
        }
      }
      if (ron < 0 || o.action(1 - ron).type() != Action.Type.PASS) {
        continue;
      }
      assertEquals(Exploration.kind(o), Kind.AGARI_OR_ABORT);
      double scale = SETTINGS.maximumScale();
      double exploring = SETTINGS.gateMass() * scale;
      float[] policy = {random.nextFloat(), random.nextFloat()};
      double[] normalized = normalized(policy, 2);
      exploration.mix(o, policy, scale, behaviour);
      double graph = (1 - exploring) * normalized[ron] + exploring / 2;
      assertEquals(behaviour[ron], FLOOR + (1 - 2 * FLOOR) * graph, 1e-6);
      assertEquals(behaviour[1 - ron], FLOOR + (1 - 2 * FLOOR) * (1 - graph), 1e-6);
      checked++;
    }
    assertTrue(checked > 0, "no ron-or-pass response");
  }

  /** 倍率 m = 最小 + (最大 − 最小)·p^k。百分位 p は前の macro の分布から引き、引けなければ m = 1。 */
  @Test
  public void theScaleFollowsTheUncertaintyPercentile() {
    long[][] counts = new long[Key.values().length][UncertaintyCdf.BINS];
    Arrays.fill(counts[Key.GLOBAL.ordinal()], 1);
    Exploration calibrated = new Exploration(SETTINGS, UncertaintyCdf.of(counts));
    Exploration empty = new Exploration(SETTINGS, UncertaintyCdf.empty());
    Random random = new Random(5);
    float[] behaviour = new float[64];
    float[] mixed = new float[64];
    for (Observation o : decisions) {
      int legal = o.legalActions();
      for (float[] policy : policies(legal, random)) {
        double tie = random.nextDouble();
        double uncertainty = Exploration.uncertainty(policy, legal, SETTINGS.entropyWeight());
        double percentile = (UncertaintyCdf.bin(uncertainty) + tie) / UncertaintyCdf.BINS;
        double expected =
            SETTINGS.minimumScale()
                + (SETTINGS.maximumScale() - SETTINGS.minimumScale()) * Math.pow(percentile, SETTINGS.exponent());
        double scale = calibrated.behaviour(o, policy, tie, behaviour);
        assertEquals(scale, expected, 1e-9);
        assertTrue(scale >= SETTINGS.minimumScale() && scale <= SETTINGS.maximumScale(), "scale " + scale);
        calibrated.mix(o, policy, scale, mixed);
        assertEquals(Arrays.copyOf(behaviour, legal), Arrays.copyOf(mixed, legal));
        assertEquals(empty.behaviour(o, policy, tie, behaviour), 1.0);
      }
    }
    assertEquals(calibrated.calibration().counts()[Key.GLOBAL.ordinal()].length, UncertaintyCdf.BINS);
    long counted = Arrays.stream(calibrated.calibration().counts()[Key.GLOBAL.ordinal()]).sum();
    assertEquals(counted, 3L * decisions.size());
  }

  /** 百分位は、種類の分布 → 手番か応答かの分布 → 全体の分布の順に見て、数えた数が 1,024 以上ある最初の分布から引く。 */
  @Test
  public void percentilesFallBackToWiderDistributions() {
    long[][] counts = new long[Key.values().length][UncertaintyCdf.BINS];
    counts[Key.DISCARD.ordinal()][0] = 1_024;
    counts[Key.RESPONSE.ordinal()][UncertaintyCdf.BINS - 1] = 2_000;
    counts[Key.TURN.ordinal()][0] = 1_023;
    counts[Key.GLOBAL.ordinal()][0] = 1_500;
    UncertaintyCdf cdf = UncertaintyCdf.of(counts);
    assertEquals(cdf.percentile(Kind.DISCARD, false, 0.5, 0.3), 1.0);
    assertEquals(cdf.percentile(Kind.DISCARD, false, 0.0, 0.3), 0.3, 1e-12);
    assertEquals(cdf.percentile(Kind.CALL, true, 0.5, 0.3), 0.0);
    assertEquals(cdf.percentile(Kind.CALL, true, 1.0, 0.25), 0.25, 1e-12);
    assertEquals(cdf.percentile(Kind.RIICHI, false, 0.5, 0.3), 1.0, "the turn table is short, so the global one");
    counts[Key.GLOBAL.ordinal()][0] = 1_023;
    assertTrue(Double.isNaN(UncertaintyCdf.of(counts).percentile(Kind.KAN, false, 0.5, 0.3)));
    assertTrue(Double.isNaN(UncertaintyCdf.empty().percentile(Kind.DISCARD, false, 0.5, 0.3)));
    assertThrows(IllegalArgumentException.class, () -> cdf.percentile(Kind.DISCARD, false, 1.5, 0.3));
  }

  @Test
  public void countsRoundTripAndThreadsAddUp() throws Exception {
    Random random = new Random(6);
    long[][] counts = new long[Key.values().length][UncertaintyCdf.BINS];
    for (long[] table : counts) {
      for (int bin = 0; bin < table.length; bin++) {
        table[bin] = random.nextInt(5);
      }
    }
    assertTrue(Arrays.deepEquals(UncertaintyCdf.of(counts).counts(), counts));
    assertTrue(Arrays.deepEquals(UncertaintyCdf.empty().counts(), new long[Key.values().length][UncertaintyCdf.BINS]));
    assertThrows(IllegalArgumentException.class, () -> UncertaintyCdf.of(new long[3][UncertaintyCdf.BINS]));
    assertThrows(IllegalArgumentException.class, () -> UncertaintyCdf.of(new long[Key.values().length][7]));
    counts[2][9] = -1;
    assertThrows(IllegalArgumentException.class, () -> UncertaintyCdf.of(counts));

    UncertaintyCdf.Histogram histogram = new UncertaintyCdf.Histogram();
    int perThread = 5_000;
    try (ExecutorService threads = Executors.newFixedThreadPool(4)) {
      List<Future<?>> done = new ArrayList<>();
      for (int thread = 0; thread < 4; thread++) {
        int offset = thread;
        done.add(
            threads.submit(
                () -> {
                  for (int index = 0; index < perThread; index++) {
                    Kind kind = Kind.values()[(index + offset) % Kind.values().length];
                    histogram.observe(kind, index % 2 == 0, (index % 100) / 99.0);
                  }
                }));
      }
      for (Future<?> future : done) {
        future.get();
      }
    }
    long[][] frozen = histogram.freeze().counts();
    long total = 4L * perThread;
    assertEquals(Arrays.stream(frozen[Key.GLOBAL.ordinal()]).sum(), total);
    assertEquals(
        Arrays.stream(frozen[Key.TURN.ordinal()]).sum() + Arrays.stream(frozen[Key.RESPONSE.ordinal()]).sum(), total);
    long kinds = 0;
    for (Kind kind : Kind.values()) {
      kinds += Arrays.stream(frozen[Key.of(kind).ordinal()]).sum();
    }
    assertEquals(kinds, total);
    assertEquals(frozen[Key.GLOBAL.ordinal()][UncertaintyCdf.BINS - 1], 4L * perThread / 100);
  }

  /** 迷い u = w·H/log n + (1 − w)·2π₂/(π₁ + π₂)。和で割ってから測る。 */
  @Test
  public void uncertaintyFollowsTheFormula() {
    assertEquals(Exploration.uncertainty(new float[] {1, 1, 1, 1}, 4, 0.5), 1.0, 1e-6);
    assertEquals(Exploration.uncertainty(new float[] {0, 3, 0}, 3, 0.5), 0.0, 1e-12);
    assertEquals(Exploration.uncertainty(new float[] {2, 2, 0, 0}, 4, 0.5), 0.5 * 0.5 + 0.5, 1e-6);
    assertEquals(Exploration.uncertainty(new float[] {0.6f, 0.2f, 0.2f}, 3, 1), entropy(0.6, 0.2, 0.2) / Math.log(3), 1e-6);
    assertEquals(Exploration.uncertainty(new float[] {0.6f, 0.2f, 0.2f}, 3, 0), 2 * 0.2 / 0.8, 1e-6);
  }

  @Test
  public void kindsFollowTheActionsOnOffer() {
    for (Observation o : decisions) {
      boolean agari = false;
      boolean call = false;
      boolean kan = false;
      boolean riichi = false;
      for (int action = 0; action < o.legalActions(); action++) {
        switch (o.action(action).type()) {
          case RON_AGARI, TSUMO_AGARI, KYUSHU_KYUHAI -> agari = true;
          case CHI, PON, DAIMINKAN -> call = true;
          case ANKAN, KAKAN -> kan = true;
          case RIICHI_DAHAI -> riichi = true;
          default -> {}
        }
      }
      Kind expected =
          agari ? Kind.AGARI_OR_ABORT : call ? Kind.CALL : kan ? Kind.KAN : riichi ? Kind.RIICHI : Kind.DISCARD;
      assertEquals(Exploration.kind(o), expected);
    }
  }

  @Test
  public void badInputsAreRejected() {
    Exploration exploration = new Exploration(SETTINGS, UncertaintyCdf.empty());
    Observation o = decisions.getFirst();
    float[] behaviour = new float[64];
    float[] uniform = new float[o.legalActions()];
    Arrays.fill(uniform, 1f);
    assertThrows(IllegalArgumentException.class, () -> exploration.mix(o, uniform, -0.1, behaviour));
    assertThrows(IllegalArgumentException.class, () -> exploration.mix(o, uniform, 5.01, behaviour));
    float[] negative = uniform.clone();
    negative[0] = -1f;
    assertThrows(IllegalArgumentException.class, () -> exploration.mix(o, negative, 1, behaviour));
    float[] notFinite = uniform.clone();
    notFinite[0] = Float.NaN;
    assertThrows(IllegalArgumentException.class, () -> exploration.mix(o, notFinite, 1, behaviour));
    assertThrows(IllegalArgumentException.class, () -> exploration.mix(o, new float[o.legalActions()], 1, behaviour));
    assertThrows(IllegalArgumentException.class, () -> new Exploration.Settings(0.004, 0.25, 1e-4, 0.1, 5, 4.5, 0.5));
    assertThrows(IllegalArgumentException.class, () -> new Exploration.Settings(0.004, 0.01, 1.0 / 64, 0.1, 5, 4.5, 0.5));
    assertThrows(IllegalArgumentException.class, () -> new Exploration.Settings(0.004, 0.01, 1e-4, 0, 5, 4.5, 0.5));
    assertThrows(IllegalArgumentException.class, () -> new Exploration.Settings(0.004, 0.01, 1e-4, 0.1, 5, 4.5, 1.5));
  }

  /** 3 つの方策: 全ての手が正の乱数、1 つの手だけ 1（ほかは 0）、1 つの手に寄ったもの。 */
  private static List<float[]> policies(int legal, Random random) {
    float[] positive = new float[legal];
    float[] single = new float[legal];
    float[] peaked = new float[legal];
    int favourite = random.nextInt(legal);
    for (int action = 0; action < legal; action++) {
      positive[action] = 0.01f + random.nextFloat();
      peaked[action] = action == favourite ? 50f : random.nextFloat() * 0.01f;
    }
    single[favourite] = 1f;
    return List.of(positive, single, peaked);
  }

  /** 和で割り、0 の手を 1e-8 にしてから割り直した方策。 */
  private static double[] normalized(float[] policy, int legal) {
    double sum = 0;
    for (int action = 0; action < legal; action++) {
      sum += policy[action];
    }
    float[] floored = new float[legal];
    double positive = 0;
    for (int action = 0; action < legal; action++) {
      floored[action] = (float) Math.max(1e-8f, policy[action] / sum);
      positive += floored[action];
    }
    double[] normalized = new double[legal];
    for (int action = 0; action < legal; action++) {
      normalized[action] = floored[action] / positive;
    }
    return normalized;
  }

  private static double entropy(double... probabilities) {
    double entropy = 0;
    for (double probability : probabilities) {
      entropy -= probability * Math.log(probability);
    }
    return entropy;
  }

  private static boolean response(Observation o) {
    EngineDecisionKind kind = o.decisionKind();
    return kind == EngineDecisionKind.RESPONSE || kind == EngineDecisionKind.CHANKAN
        || kind == EngineDecisionKind.NUKI_RESPONSE;
  }
}
