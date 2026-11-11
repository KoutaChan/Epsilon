package com.epsilon.classic.campaign;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.StoreEvent.Initial;
import com.epsilon.agent.eval.StoreEvent.Promoted;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.play.PolicySeat;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.settings.Settings;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.ClassicFamily;
import com.epsilon.classic.ClassicPolicy;
import com.epsilon.classic.RandomCheckpoints;
import com.epsilon.classic.app.ClassicSettings;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.learn.Exploration;
import com.epsilon.classic.learn.GameTrajectory;
import com.epsilon.classic.learn.TrajectoryBatch;
import com.epsilon.classic.learn.UncertaintyCdf;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 乱数の重みの小さなチェックポイントで、CPU の小さな卓の収集を回す。学習器の席の判断ごとの確率は、学習器の方策に探索を混ぜた振る舞いの方策の
 * 選んだ手の値で、振る舞いの方策はどの合法手も下限以上で和が 1。macro の収集は種だけで決まりワーカーの数に依らず、相手は候補の 1 人、迷いは学習器が答えた
 * 判断ごとに数える。
 */
public class CollectTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final Architecture ARCHITECTURE = Architecture.NANO;
  private static final PolicyOptions CPU = PolicyOptions.defaults();
  private static final Exploration.Settings EXPLORE = Settings.bind(ClassicSettings.class, List.of()).value().explore();
  private static final long RUN_SEED = 77_031L, KEY = 0x7E57L;
  private static final int GAMES = 4;

  private Path root;
  private Path learner;
  private Path champion;
  private Path store;

  /** native のグループだけを回しても準備が走るよう、グループを付けない。 */
  @BeforeClass(alwaysRun = true)
  public void writeCheckpoints() throws IOException {
    root = Files.createTempDirectory("classic-collect");
    learner = root.resolve("learner");
    RandomCheckpoints.write(learner, ARCHITECTURE, VARIANT, 31);
    Path past = root.resolve("past");
    RandomCheckpoints.write(past, ARCHITECTURE, VARIANT, 32);
    champion = root.resolve("champion");
    RandomCheckpoints.write(champion, ARCHITECTURE, VARIANT, 33);
    store = root.resolve("store");
    ChampionStore champions = ChampionStore.open(store);
    champions.append(new Initial(Champion.of("v00000", past, 0)));
    champions.append(new Promoted(Champion.of("v00010", champion, 10), "v00010-champion"));
  }

  @AfterClass(alwaysRun = true)
  public void deleteCheckpoints() throws IOException {
    if (root == null) {
      return;
    }
    try (Stream<Path> files = Files.walk(root)) {
      for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(file);
      }
    }
  }

  /** 学習器の席が書く振る舞いの方策は、学習器の方策と判断の記録だけから作り直せる。 */
  @Test(groups = "native", timeOut = 1_800_000)
  public void theLearnerSeatPlaysTheBehaviourOfItsPolicy() throws IOException {
    Collect collect = new Collect(VARIANT, RUN_SEED, new Collect.Settings(GAMES, 0.7), EXPLORE, null, CPU, null);
    Exploration exploration = new Exploration(EXPLORE, UncertaintyCdf.empty());
    Collect.Played played;
    try (TableLoop table = new TableLoop(new TableSettings(4, 2, 1_000L, 4));
        LearnerSeat seat =
            new LearnerSeat(ClassicFamily.open(ARCHITECTURE, learner, VARIANT, CPU, null), exploration, KEY);
        PolicySeat opponent =
            new PolicySeat(ClassicFamily.open(ARCHITECTURE, champion, VARIANT, CPU, null).asPolicy())) {
      assertTrue(seat.inline());
      played = collect.play(table, 0, seat, opponent, null);
    }
    assertEquals(played.trajectories().games().size(), GAMES);
    assertEquals(counted(exploration.calibration()), 3L * queried(played.trajectories()));
    double floor = EXPLORE.leafFloor();
    Exploration fresh = new Exploration(EXPLORE, UncertaintyCdf.empty());
    Observation reader = new Observation(VARIANT);
    Observation copied = new Observation(VARIANT);
    float[] policy = new float[ClassicSchema.MAX_LEGAL];
    ClassicPolicy reference = ClassicFamily.open(ARCHITECTURE, learner, VARIANT, CPU, null);
    try {
      for (GameTrajectory game : played.trajectories().games()) {
        for (int decision = 0; decision < game.size(); decision++) {
          String where = "game " + game.id() + " decision " + decision;
          Observation o = game.bind(decision, reader);
          int legal = o.legalActions();
          int chosen = game.chosen(decision);
          assertTrue(chosen >= 0 && chosen < legal, where + " chose " + chosen + " of " + legal);
          assertEquals(game.seat(decision), (int) (game.id() % VARIANT.seats()), where);
          if (legal == 1) {
            assertEquals(game.behaviour(decision), 1f, where);
            continue;
          }
          ObservationBatch copy = new ObservationBatch(VARIANT, 64);
          copy.add(o);
          Observation bound = copied.bind(copy, 0);
          reference.probabilities(bound, policy);
          float[] behaviour = new float[legal];
          fresh.behaviour(
              bound, policy, LearnerSeat.tie(KEY, game.id(), copy.words(), copy.start(0), bound.longs()), behaviour);
          double sum = 0;
          for (float probability : behaviour) {
            assertTrue(probability >= floor * (1 - 1e-6), where + " " + Arrays.toString(behaviour));
            sum += probability;
          }
          assertEquals(sum, 1.0, 1e-5, where);
          assertEquals(game.behaviour(decision), (float) (behaviour[chosen] / sum), 1e-6f, where);
        }
      }
    } finally {
      reference.close();
    }
  }

  /** macro の収集は、run の種と macro の番号だけで決まる（ワーカーと同時に進める半荘の数に依らない）。 */
  @Test(groups = "native", timeOut = 1_800_000)
  public void aMacroOnTheCpuDependsOnTheSeedsAlone() throws IOException {
    Collect collect =
        new Collect(
            VARIANT,
            RUN_SEED,
            new Collect.Settings(GAMES, 0.7),
            EXPLORE,
            SelfPlayHooksTest.randomGrp(VARIANT, 5),
            CPU,
            null);
    OpponentPool pool = OpponentPool.of(ChampionStore.open(store));
    Collect.Result wide;
    Collect.Result single;
    try (TableLoop table = new TableLoop(new TableSettings(4, 2, 1_000L, 4))) {
      wide = collect.macro(table, 2, learner, pool, UncertaintyCdf.empty());
    }
    try (TableLoop table = new TableLoop(new TableSettings(1, 1, 1_000L, 4))) {
      single = collect.macro(table, 2, learner, pool, UncertaintyCdf.empty());
    }
    assertEquals(SelfPlayHooksTest.digest(wide.trajectories()), SelfPlayHooksTest.digest(single.trajectories()));
    assertEquals(wide.opponent(), single.opponent());
    assertTrue(wide.opponent().equals(pool.champion()) || pool.past().contains(wide.opponent()), wide.opponent().label());
    assertEquals(wide.trajectories().games().size(), GAMES);
    assertTrue(wide.trajectories().games().getFirst().hasPriors());
    assertTrue(Arrays.deepEquals(wide.calibration().counts(), single.calibration().counts()));
    assertEquals(counted(wide.calibration()), 3L * queried(wide.trajectories()));
  }

  /**
   * 持ち主が置いた初めの王者（run の外のチェックポイント、バージョン 0）を相手に選んだら、開く前に、置いたときの SHA-256 と照合する。置いた後に
   * 同じディレクトリへ同じ形の別のネットワークを書き出し直すと、収集はそのネットワークを開かずに止まる。
   */
  @Test(groups = "native", timeOut = 1_800_000)
  public void aPlacedChampionRewrittenAfterItWasPlacedIsNotOpened() throws IOException {
    Path placed = root.resolve("placed");
    RandomCheckpoints.write(placed, ARCHITECTURE, VARIANT, 34);
    Path placedStore = root.resolve("placed-store");
    ChampionStore.open(placedStore).append(new Initial(Champion.placed("nano-stage8", placed, 0)));
    DurableFiles.delete(placed);
    RandomCheckpoints.write(placed, ARCHITECTURE, VARIANT, 35);

    Collect collect = new Collect(VARIANT, RUN_SEED, new Collect.Settings(GAMES, 0.7), EXPLORE, null, CPU, null);
    OpponentPool pool = OpponentPool.of(ChampionStore.open(placedStore));
    try (TableLoop table = new TableLoop(new TableSettings(4, 2, 1_000L, 4))) {
      IOException refused =
          expectThrows(IOException.class, () -> collect.macro(table, 1, learner, pool, UncertaintyCdf.empty()));
      assertTrue(
          refused.getMessage().contains("but the store recorded " + pool.champion().sha256()), refused.getMessage());
    }
  }

  /** 迷いの表の数の和。判断ごとに種類・手番か応答か・全体の 3 つの表に数える。 */
  private static long counted(UncertaintyCdf calibration) {
    long sum = 0;
    for (long[] table : calibration.counts()) {
      for (long count : table) {
        sum += count;
      }
    }
    return sum;
  }

  /** 合法手が 2 つ以上の判断（学習器の席に問い合わせた判断）の数。 */
  private static long queried(TrajectoryBatch batch) {
    Observation reader = new Observation(batch.variant());
    long queried = 0;
    for (GameTrajectory game : batch.games()) {
      for (int decision = 0; decision < game.size(); decision++) {
        queried += game.bind(decision, reader).legalActions() > 1 ? 1 : 0;
      }
    }
    return queried;
  }
}
