package com.epsilon.classic.campaign;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.GameObserver;
import com.epsilon.agent.play.PolicySeat;
import com.epsilon.agent.play.Query;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSeat;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.settings.Settings;
import com.epsilon.classic.app.ClassicSettings;
import com.epsilon.classic.learn.Exploration;
import com.epsilon.classic.learn.GameTrajectory;
import com.epsilon.classic.learn.GrpPretrain;
import com.epsilon.classic.learn.TrajectoryBatch;
import com.epsilon.classic.learn.UncertaintyCdf;
import com.epsilon.classic.rulebot.RuleBot;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpShape;
import com.epsilon.grp.GrpTensor;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.grp.RankPermutations;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import org.testng.annotations.Test;

/**
 * 自己対局の卓の処理（{@link SelfPlayHooks}）を、4 人打ちの純 Java の席で回す。学習器は方策を判断の記録の合法手と観測席から作って探索を混ぜた振る舞いの方策を
 * 書き、相手はルールベースの打ち手。軌跡は種だけで決まり、ワーカーの数に依らない。学習器は牌山の番号 mod 席の数の席に座り、判断ごとの確率は
 * 振る舞いの方策の選んだ手の値で、振る舞いの方策はどの合法手も下限以上で和が 1。選んだ手は合法手の中にある。局末の価値は、GRP が無ければ終局の順位の効用、あれば次の
 * 局の始まりの周辺確率で平均した効用で、周辺確率は観測した局の始まりの卓から GRP で求めた値と同じ。
 */
public class SelfPlayHooksTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final long RUN_SEED = 20_261_110L, MACRO = 5, KEY = 0x5EEDL;
  private static final int GAMES = 8;
  private static final RankUtility UTILITY = RankUtility.TENHOU;
  private static final Exploration.Settings EXPLORE = Settings.bind(ClassicSettings.class, List.of()).value().explore();

  @Test(timeOut = 600_000)
  public void theTrajectoriesDependOnTheSeedsAloneWhateverTheWorkers() {
    GrpNetwork grp = randomGrp(VARIANT, 7);
    long single = digest(play(new TableSettings(1, 1, 1_000L, 4), grp, MACRO, null).trajectories());
    long wide = digest(play(new TableSettings(8, 4, 1_000L, 4), grp, MACRO, null).trajectories());
    long other = digest(play(new TableSettings(3, 2, 1_000L, 4), grp, MACRO + 1, null).trajectories());
    assertEquals(wide, single);
    assertNotEquals(other, single, "another macro plays other walls");
  }

  /**
   * 学習器の席の判断ごとに、確率は振る舞いの方策の選んだ手の値を和で割った値。振る舞いの方策は判断の記録から作り直せ、どの合法手も下限以上で和が 1。合法手が 1 つの判断は確率 1 で、
   * 選び替えで局の進みが変わりえない。合法手が 2 つ以上で局の進みが変わらなかった判断は、応答だけ。
   */
  @Test(timeOut = 600_000)
  public void everyLearnerDecisionRecordsTheDrawnShareOfItsBehaviour() {
    double floor = EXPLORE.leafFloor();
    Collect.Played played = play(new TableSettings(8, 4, 1_000L, 4), null, MACRO, null);
    assertEquals(played.table().games(), GAMES);
    Exploration fresh = new Exploration(EXPLORE, UncertaintyCdf.empty());
    Observation reader = new Observation(VARIANT);
    Observation copied = new Observation(VARIANT);
    int queried = 0;
    int preempted = 0;
    for (GameTrajectory game : played.trajectories().games()) {
      int learner = (int) (game.id() % VARIANT.seats());
      assertTrue(game.size() > 0, "game " + game.id());
      for (int decision = 0; decision < game.size(); decision++) {
        Observation o = game.bind(decision, reader);
        String where = "game " + game.id() + " decision " + decision;
        assertEquals(game.seat(decision), learner, where);
        assertEquals(o.observer(), learner, where);
        int legal = o.legalActions();
        int chosen = game.chosen(decision);
        assertTrue(chosen >= 0 && chosen < legal, where + " chose " + chosen + " of " + legal);
        if (legal == 1) {
          assertFalse(game.causal(decision), where);
          assertEquals(game.behaviour(decision), 1f, where);
          continue;
        }
        if (!game.causal(decision)) {
          preempted++;
          assertTrue(response(o.decisionKind()), where + " " + o.decisionKind());
        }
        queried++;
        ObservationBatch copy = new ObservationBatch(VARIANT, 64);
        copy.add(o);
        float[] behaviour = behaviour(fresh, copied.bind(copy, 0), game.id(), copy.words(), copy.start(0));
        double sum = 0;
        for (float probability : behaviour) {
          assertTrue(probability >= floor * (1 - 1e-6), where + " " + Arrays.toString(behaviour));
          sum += probability;
        }
        assertEquals(sum, 1.0, 1e-5, where);
        assertEquals(game.behaviour(decision), (float) (behaviour[chosen] / sum), 1e-7f, where);
      }
    }
    assertTrue(queried > 0, "no learner decision was queried");
    assertTrue(preempted > 0, "no learner response was preempted");
  }

  @Test(timeOut = 600_000)
  public void handEndValuesFollowTheRankUtility() {
    int seats = VARIANT.seats();
    for (GrpNetwork grp : Arrays.asList(null, randomGrp(VARIANT, 11))) {
      Tables tables = new Tables();
      Collect.Played played = play(new TableSettings(8, 4, 1_000L, 4), grp, MACRO, tables);
      for (GameTrajectory game : played.trajectories().games()) {
        String where = "game " + game.id() + (grp == null ? "" : " with GRP");
        List<TableState> ends = tables.ends(game.id());
        assertEquals(game.hands(), ends.size(), where);
        assertEquals(game.hasPriors(), grp != null, where);
        TableState last = ends.getLast();
        assertTrue(last.gameOver(), where);
        for (int seat = 0; seat < seats; seat++) {
          assertEquals(game.finalRank(seat), last.rank(seat), where);
        }
        for (int hand = 0; hand < game.hands(); hand++) {
          if (grp != null) {
            TableState start = hand == 0 ? new TableState(VARIANT) : ends.get(hand - 1);
            assertTrue(Arrays.equals(game.prior(hand), marginals(grp, start)), where + " hand " + hand);
          }
          for (int seat = 0; seat < seats; seat++) {
            double expected =
                grp == null || hand + 1 == game.hands()
                    ? UTILITY.value(VARIANT, last.rank(seat))
                    : phi(game.prior(hand + 1), seat);
            assertEquals(game.endValue(hand, seat, UTILITY), expected, 1e-12, where + " hand " + hand);
          }
        }
      }
    }
  }

  /** 純 Java の席で macro の半荘を打つ。 */
  private static Collect.Played play(TableSettings settings, GrpNetwork grp, long macro, GameObserver observer) {
    Collect collect =
        new Collect(VARIANT, RUN_SEED, new Collect.Settings(GAMES, 0.7), EXPLORE, grp, PolicyOptions.defaults(), null);
    try (TableLoop table = new TableLoop(settings);
        HashedLearner learner = new HashedLearner();
        PolicySeat opponent = new PolicySeat(new RuleBot(VARIANT))) {
      return collect.play(table, macro, learner, opponent, observer);
    }
  }

  /** 方策を判断の記録の合法手と観測席だけから作り、探索を混ぜた振る舞いの方策を書く、その場で答える学習器の席。 */
  private static final class HashedLearner implements TableSeat<Query> {

    private final Exploration exploration = new Exploration(EXPLORE, UncertaintyCdf.empty());
    private final ThreadLocal<Observation> readers = ThreadLocal.withInitial(() -> new Observation(VARIANT));

    @Override
    public boolean inline() {
      return true;
    }

    @Override
    public void evaluateInline(Query query) {
      Observation o = query.bind(readers.get());
      query.sigma = behaviour(exploration, o, query.wall(), query.recordWords(), query.recordBase());
    }

    @Override
    public int admission(Observation observation) {
      return 0;
    }
  }

  /** 学習器の席の振る舞いの方策（迷いの百分位を散らす一様値は {@link LearnerSeat#tie}）。 */
  private static float[] behaviour(Exploration exploration, Observation o, long wall, long[] words, int base) {
    float[] policy = new float[o.legalActions()];
    for (int action = 0; action < policy.length; action++) {
      long bits = SeedMixer.mix64(o.action(action).code() * 31L + o.observer());
      policy[action] = (float) (0.05 + (bits >>> 11) * 0x1.0p-53);
    }
    float[] behaviour = new float[policy.length];
    exploration.behaviour(o, policy, LearnerSeat.tie(KEY, wall, words, base, o.longs()), behaviour);
    return behaviour;
  }

  /** 主経路の局末ごとに、精算した後の卓（半荘が続くなら次の局の始まり、終わるなら終局）を牌山の番号ごとに写して持つ。 */
  private static final class Tables implements GameObserver {

    private final Map<Long, List<TableState>> ends = new ConcurrentHashMap<>();

    @Override
    public void roundEnded(GameKey game, FullView full, RoundEnd end) {
      TableState copy = new TableState(VARIANT);
      copy.copyFrom(end.next());
      ends.computeIfAbsent(game.wall(), wall -> Collections.synchronizedList(new ArrayList<>())).add(copy);
    }

    @Override
    public void gameEnded(GameKey game, TableState last) {}

    List<TableState> ends(long wall) {
      return ends.get(wall);
    }
  }

  /** 軌跡の中身（O・選んだ手・確率・選び替えの効き・席・局・周辺確率・終局の順位）のハッシュ。 */
  static long digest(TrajectoryBatch batch) {
    Variant variant = batch.variant();
    Observation reader = new Observation(variant);
    long hash = 0;
    for (GameTrajectory game : batch.games()) {
      hash = mix(hash, game.id());
      hash = mix(hash, game.size());
      hash = mix(hash, game.hands());
      for (int seat = 0; seat < variant.seats(); seat++) {
        hash = mix(hash, game.finalRank(seat));
      }
      for (int hand = 0; hand < game.hands(); hand++) {
        float[] prior = game.prior(hand);
        hash = mix(hash, prior == null ? -1 : prior.length);
        if (prior != null) {
          for (float probability : prior) {
            hash = mix(hash, Float.floatToRawIntBits(probability));
          }
        }
      }
      for (int decision = 0; decision < game.size(); decision++) {
        hash = mix(hash, game.seat(decision));
        hash = mix(hash, game.hand(decision));
        hash = mix(hash, game.chosen(decision));
        hash = mix(hash, Float.floatToRawIntBits(game.behaviour(decision)));
        hash = mix(hash, game.causal(decision) ? 1 : 0);
        ObservationBatch copy = new ObservationBatch(variant, 64);
        copy.add(game.bind(decision, reader));
        for (int index = copy.start(0); index < copy.start(0) + copy.longs(0); index++) {
          hash = mix(hash, copy.words()[index]);
        }
      }
    }
    return hash;
  }

  /** 乱数の重みの小さな GRP。 */
  static GrpNetwork randomGrp(Variant variant, long seed) {
    GrpShape shape = new GrpShape(variant, GrpPretrain.FIELDS, 8, false);
    float[] weights = new float[GrpTensor.starts(shape)[GrpTensor.values().length]];
    Random random = new Random(seed);
    for (int index = 0; index < weights.length; index++) {
      weights[index] = (float) (0.3 * random.nextGaussian());
    }
    return GrpNetwork.of(shape, weights);
  }

  /** 卓の、GRP の絶対席 × 順位の周辺確率。 */
  private static float[] marginals(GrpNetwork grp, TableState start) {
    RankPermutations permutations = RankPermutations.of(grp.variant());
    double[] probabilities = new double[permutations.count()];
    grp.permutations(start, GrpWorkspace.of(grp.shape()), probabilities);
    int seats = grp.variant().seats();
    double[] marginals = new double[seats * seats];
    permutations.marginals(probabilities, start.dealer(), marginals);
    float[] prior = new float[marginals.length];
    for (int index = 0; index < prior.length; index++) {
      prior[index] = (float) marginals[index];
    }
    return prior;
  }

  /** Φ_s = Σ_r P[s][r]·u_r。 */
  private static double phi(float[] prior, int seat) {
    int seats = VARIANT.seats();
    double phi = 0;
    for (int rank = 0; rank < seats; rank++) {
      phi += prior[seat * seats + rank] * UTILITY.value(VARIANT, rank);
    }
    return phi;
  }

  private static boolean response(EngineDecisionKind kind) {
    return kind == EngineDecisionKind.RESPONSE || kind == EngineDecisionKind.CHANKAN
        || kind == EngineDecisionKind.NUKI_RESPONSE;
  }

  private static long mix(long hash, long value) {
    return SeedMixer.mix64(hash ^ SeedMixer.mix64(value));
  }
}
