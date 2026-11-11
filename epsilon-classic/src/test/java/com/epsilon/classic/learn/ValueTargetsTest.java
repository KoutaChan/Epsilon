package com.epsilon.classic.learn;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 選んだ手の補正の式と、局の中の V-trace の価値の目標と優位を確かめる。手で計算した値と、V-trace の和の形
 * v_s = V_s + Σ_t λ^(t−s)·(Π_{i=s..t} c_i)·(V_{t+1} − V_t) に照合する。局末の価値は、半荘が続く局末なら、次の局の始まりの周辺確率で
 * 求めた Φ。最後の局と、周辺確率の無い半荘では、終局の順位の効用。判断の鎖は席ごと・局ごとに切れる。
 */
public class ValueTargetsTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final RankUtility UTILITY = RankUtility.TENHOU;
  private static final double TOLERANCE = 1e-5;

  /** 終局の点（席 0 から順に 2 位・1 位・4 位・3 位）。 */
  private static final int[] SCORES = {30_000, 40_000, 10_000, 20_000};

  private Trajectories.Pool pool;

  @BeforeClass
  public void collect() {
    pool = new Trajectories.Pool(VARIANT, 5, 8);
  }

  @Test
  public void creditFollowsTheFormula() {
    assertEquals(ValueTargets.credit(0.1, 0.3, 0.6), 0.1 + 0.9 * 0.5, 1e-12);
    assertEquals(ValueTargets.credit(0.1, 0.6, 0.3), 1.0, 0.0);
    assertEquals(ValueTargets.credit(0.1, 0.0, 0.5), 0.1, 1e-12);
    assertEquals(ValueTargets.credit(1.0, 0.01, 0.9), 1.0, 0.0);
  }

  /** 1 つの局で席 1 がした 3 つの判断（λ = 0.5）。半荘の最後の局なので、局末の価値は終局で 1 位の席 1 の効用 2/3。 */
  @Test
  public void targetsMatchAHandComputation() {
    GameTrajectory.Builder builder = new GameTrajectory.Builder(VARIANT, 1);
    builder.hand(null);
    for (int index = 0; index < 3; index++) {
      add(builder, 1, index);
    }
    TrajectoryBatch batch = TrajectoryBatch.of(VARIANT, List.of(builder.finish(SCORES)));
    float[] values = {0.2f, 0.4f, 0.1f};
    float[] credits = {0.5f, 1f, 0.8f};
    float[] targets = new float[3];
    float[] advantages = new float[3];
    ValueTargets.compute(batch, UTILITY, values, credits, 0.5, targets, advantages);
    double end = 2.0 / 3;
    // t = 2: q = G、v = 0.1 + 0.8·(G − 0.1)
    double q2 = end;
    double v2 = 0.1 + 0.8 * (q2 - 0.1);
    // t = 1: q = 0.5·V_2 + 0.5·v_2、v = 0.4 + 1·(q − 0.4)
    double q1 = 0.5 * 0.1 + 0.5 * v2;
    double v1 = q1;
    // t = 0: q = 0.5·V_1 + 0.5·v_1、v = 0.2 + 0.5·(q − 0.2)
    double q0 = 0.5 * 0.4 + 0.5 * v1;
    double v0 = 0.2 + 0.5 * (q0 - 0.2);
    assertEquals(targets[2], v2, TOLERANCE);
    assertEquals(targets[1], v1, TOLERANCE);
    assertEquals(targets[0], v0, TOLERANCE);
    assertEquals(advantages[2], q2 - 0.1, TOLERANCE);
    assertEquals(advantages[1], q1 - 0.4, TOLERANCE);
    assertEquals(advantages[0], q0 - 0.2, TOLERANCE);
  }

  /** 席を混ぜた 3 局の半荘（周辺確率あり・なし）で、目標と優位が、席ごと・局ごとの鎖で計算した V-trace の和の形と一致する。 */
  @Test
  public void interleavedSeatsAndHandsMatchTheVtraceSum() {
    Random random = new Random(3);
    for (boolean priors : new boolean[] {false, true}) {
      for (double lambda : new double[] {0, 0.5, 0.95, 1}) {
        GameTrajectory.Builder builder = new GameTrajectory.Builder(VARIANT, 2);
        int[] used = new int[VARIANT.seats()];
        int[][] pattern = {{0, 1, 0, 2, 1, 0}, {3, 3, 1}, {2, 0, 2, 2}};
        for (int[] hand : pattern) {
          builder.hand(priors ? Trajectories.prior(VARIANT, random) : null);
          for (int seat : hand) {
            add(builder, seat, used[seat]++);
          }
        }
        GameTrajectory game = builder.finish(SCORES);
        TrajectoryBatch batch = TrajectoryBatch.of(VARIANT, List.of(game));
        int size = batch.size();
        float[] values = new float[size];
        float[] credits = new float[size];
        for (int decision = 0; decision < size; decision++) {
          values[decision] = (float) (random.nextDouble() * 2 - 1);
          credits[decision] = (float) (0.1 + 0.9 * random.nextDouble());
        }
        float[] targets = new float[size];
        float[] advantages = new float[size];
        ValueTargets.compute(batch, UTILITY, values, credits, lambda, targets, advantages);
        for (int hand = 0; hand < game.hands(); hand++) {
          for (int seat = 0; seat < VARIANT.seats(); seat++) {
            List<Integer> chain = new ArrayList<>();
            for (int decision = 0; decision < size; decision++) {
              if (game.hand(decision) == hand && game.seat(decision) == seat) {
                chain.add(decision);
              }
            }
            check(chain, game.endValue(hand, seat, UTILITY), values, credits, lambda, targets, advantages);
          }
        }
      }
    }
  }

  /** λ = 1 で補正が全て 1 なら目標はどれも局末の価値、λ = 0 なら次の判断の価値（最後の判断は局末の価値）。 */
  @Test
  public void lambdaEndsGiveMonteCarloAndOneStepTargets() {
    GameTrajectory.Builder builder = new GameTrajectory.Builder(VARIANT, 3);
    builder.hand(null);
    for (int index = 0; index < 4; index++) {
      add(builder, 2, index);
    }
    TrajectoryBatch batch = TrajectoryBatch.of(VARIANT, List.of(builder.finish(SCORES)));
    float[] values = {0.3f, -0.2f, 0.5f, 0.1f};
    float[] credits = {1f, 1f, 1f, 1f};
    float[] targets = new float[4];
    float[] advantages = new float[4];
    double end = UTILITY.value(VARIANT, 3);
    ValueTargets.compute(batch, UTILITY, values, credits, 1, targets, advantages);
    for (int decision = 0; decision < 4; decision++) {
      assertEquals(targets[decision], end, TOLERANCE);
      assertEquals(advantages[decision], end - values[decision], TOLERANCE);
    }
    ValueTargets.compute(batch, UTILITY, values, credits, 0, targets, advantages);
    for (int decision = 0; decision < 4; decision++) {
      double next = decision == 3 ? end : values[decision + 1];
      assertEquals(targets[decision], next, TOLERANCE);
      assertEquals(advantages[decision], next - values[decision], TOLERANCE);
    }
    assertThrows(
        IllegalArgumentException.class,
        () -> ValueTargets.compute(batch, UTILITY, values, credits, 1.5, targets, advantages));
  }

  /** 局末の価値: 続く局末は次の局の周辺確率で効用を平均した値、最後の局と周辺確率の無い半荘は終局の順位の効用。 */
  @Test
  public void handEndValuesUseTheNextHandPriorOrTheFinalRank() {
    float[] first = new float[16];
    float[] second = new float[16];
    for (int seat = 0; seat < 4; seat++) {
      for (int rank = 0; rank < 4; rank++) {
        first[seat * 4 + rank] = 0.25f;
        second[seat * 4 + rank] = rank == seat ? 1f : 0f;
      }
    }
    second[2 * 4 + 2] = 0.5f;
    second[2 * 4 + 3] = 0.5f;
    GameTrajectory.Builder builder = new GameTrajectory.Builder(VARIANT, 4);
    builder.hand(first);
    add(builder, 0, 0);
    builder.hand(second);
    add(builder, 2, 0);
    GameTrajectory game = builder.finish(SCORES);
    assertEquals(game.endValue(0, 0, UTILITY), UTILITY.value(VARIANT, 0), 1e-6);
    assertEquals(game.endValue(0, 2, UTILITY), 0.5 * UTILITY.value(VARIANT, 2) + 0.5 * UTILITY.value(VARIANT, 3), 1e-6);
    assertEquals(game.endValue(0, 3, UTILITY), UTILITY.value(VARIANT, 3), 1e-6);
    int[] finalRanks = {1, 0, 3, 2};
    for (int seat = 0; seat < 4; seat++) {
      assertEquals(game.finalRank(seat), finalRanks[seat]);
      assertEquals(game.endValue(1, seat, UTILITY), UTILITY.value(VARIANT, finalRanks[seat]), 0.0);
    }

    GameTrajectory.Builder plain = new GameTrajectory.Builder(VARIANT, 5);
    plain.hand(null);
    add(plain, 0, 0);
    plain.hand(null);
    GameTrajectory without = plain.finish(SCORES);
    for (int seat = 0; seat < 4; seat++) {
      assertEquals(without.endValue(0, seat, UTILITY), UTILITY.value(VARIANT, finalRanks[seat]), 0.0);
    }
  }

  /** V-trace の和の形で、1 つの鎖（起きた順の判断の番号）の目標と優位を確かめる。 */
  private static void check(
      List<Integer> chain,
      double end,
      float[] values,
      float[] credits,
      double lambda,
      float[] targets,
      float[] advantages) {
    int length = chain.size();
    double[] value = new double[length + 1];
    for (int t = 0; t < length; t++) {
      value[t] = values[chain.get(t)];
    }
    value[length] = end;
    double[] target = new double[length + 1];
    target[length] = end;
    for (int s = 0; s < length; s++) {
      double sum = 0;
      double trace = 1;
      for (int t = s; t < length; t++) {
        trace *= (t == s ? 1 : lambda) * credits[chain.get(t)];
        sum += trace * (value[t + 1] - value[t]);
      }
      target[s] = value[s] + sum;
    }
    for (int s = 0; s < length; s++) {
      int decision = chain.get(s);
      double lookahead = (1 - lambda) * value[s + 1] + lambda * target[s + 1];
      assertEquals(targets[decision], target[s], TOLERANCE, "target of decision " + decision);
      assertEquals(advantages[decision], lookahead - value[s], TOLERANCE, "advantage of decision " + decision);
    }
  }

  /** その席の {@code index} 番目の判断を、選んだ手 0・振る舞いの確率 1 で足す。 */
  private void add(GameTrajectory.Builder builder, int seat, int index) {
    Observation o = pool.bind(seat, index, new Observation(VARIANT));
    builder.decision(o, 0, 1f, true);
  }
}
