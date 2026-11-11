package com.epsilon.classic.learn;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 軌跡を書き足すときに拒むもの（局を始める前の判断、周辺確率の長さや値の誤りと局ごとの有無の食い違い、範囲の外の確率と手、終えた後
 * の書き足し）と、読み戻しを確かめる。軌跡の束の並びが半荘 ID の順で、足した順やスレッドに依らないこと、通しの番号から半荘を引ける
 * ことも確かめる。
 */
public class TrajectoryBatchTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final int[] SCORES = {25_000, 25_000, 25_000, 25_000};

  private Trajectories.Pool pool;

  @BeforeClass
  public void collect() {
    pool = new Trajectories.Pool(VARIANT, 9, 6);
  }

  @Test
  public void aGameKeepsWhatWasWritten() {
    Observation reader = new Observation(VARIANT);
    float[] prior = Trajectories.prior(VARIANT, new Random(1));
    GameTrajectory.Builder builder = new GameTrajectory.Builder(VARIANT, 77);
    builder.hand(prior);
    Observation first = pool.bind(2, 0, reader);
    int firstLegal = first.legalActions();
    builder.decision(first, firstLegal - 1, 0.25f, true);
    builder.hand(prior);
    Observation second = pool.bind(1, 1, reader);
    long[] words = words(second);
    builder.decision(second, 0, 1f, false);
    prior[0] = 9f;
    GameTrajectory game = builder.finish(SCORES);

    assertEquals(game.id(), 77);
    assertEquals(game.size(), 2);
    assertEquals(game.hands(), 2);
    assertTrue(game.hasPriors());
    assertEquals(game.seat(0), 2);
    assertEquals(game.seat(1), 1);
    assertEquals(game.hand(0), 0);
    assertEquals(game.hand(1), 1);
    assertEquals(game.chosen(0), firstLegal - 1);
    assertEquals(game.behaviour(0), 0.25f);
    assertTrue(game.causal(0));
    assertFalse(game.causal(1));
    assertTrue(game.prior(0)[0] < 1f, "the prior is copied");
    assertEquals(words(game.bind(1, new Observation(VARIANT))), words);
    assertEquals(game.finalRank(3), 3);
    assertThrows(IllegalStateException.class, () -> builder.hand(null));
    assertThrows(IllegalStateException.class, () -> builder.finish(SCORES));
  }

  @Test
  public void theBuilderRejectsBadInput() {
    Observation o = pool.bind(0, 0, new Observation(VARIANT));
    GameTrajectory.Builder builder = new GameTrajectory.Builder(VARIANT, 1);
    assertThrows(IllegalStateException.class, () -> builder.decision(o, 0, 1f, true));
    assertThrows(IllegalStateException.class, () -> builder.finish(SCORES));
    assertThrows(IllegalArgumentException.class, () -> builder.hand(new float[15]));
    float[] negative = new float[16];
    negative[3] = -0.1f;
    assertThrows(IllegalArgumentException.class, () -> builder.hand(negative));
    builder.hand(null);
    assertThrows(IllegalArgumentException.class, () -> builder.hand(new float[16]));
    assertThrows(IllegalArgumentException.class, () -> builder.decision(o, 0, 0f, true));
    assertThrows(IllegalArgumentException.class, () -> builder.decision(o, 0, 1.5f, true));
    assertThrows(IndexOutOfBoundsException.class, () -> builder.decision(o, o.legalActions(), 1f, true));
    assertThrows(IllegalArgumentException.class, () -> builder.finish(new int[3]));
    GameTrajectory game = builder.finish(SCORES);
    assertFalse(game.hasPriors());
    assertNull(game.prior(0));
    assertEquals(game.size(), 0);
  }

  @Test
  public void batchesOrderGamesByIdWhateverTheOrderAdded() throws Exception {
    List<GameTrajectory> games = new ArrayList<>();
    for (int id = 0; id < 12; id++) {
      games.add(game(100 - 7L * id, id % 3));
    }
    List<GameTrajectory> shuffled = new ArrayList<>(games);
    Collections.shuffle(shuffled, new Random(4));
    TrajectoryBatch.Collector collector = new TrajectoryBatch.Collector(VARIANT);
    try (ExecutorService threads = Executors.newFixedThreadPool(4)) {
      List<Future<?>> added = new ArrayList<>();
      for (GameTrajectory game : shuffled) {
        added.add(threads.submit(() -> collector.add(game)));
      }
      for (Future<?> future : added) {
        future.get();
      }
    }
    assertEquals(collector.games(), games.size());
    TrajectoryBatch batch = collector.build();
    TrajectoryBatch direct = TrajectoryBatch.of(VARIANT, games);
    assertEquals(batch.games(), direct.games());
    int decisions = 0;
    for (int game = 0; game < batch.games().size(); game++) {
      GameTrajectory trajectory = batch.games().get(game);
      if (game > 0) {
        assertTrue(batch.games().get(game - 1).id() < trajectory.id());
      }
      assertEquals(batch.start(game), decisions);
      for (int local = 0; local < trajectory.size(); local++) {
        assertEquals(batch.gameIndex(decisions + local), game);
      }
      decisions += trajectory.size();
    }
    int total = decisions;
    assertEquals(batch.size(), total);
    assertThrows(IndexOutOfBoundsException.class, () -> batch.gameIndex(total));

    List<GameTrajectory> twice = new ArrayList<>(games);
    twice.add(game(100, 1));
    assertThrows(IllegalArgumentException.class, () -> TrajectoryBatch.of(VARIANT, twice));
    assertThrows(IllegalArgumentException.class, () -> TrajectoryBatch.of(Variant.TENHOU_3P, games));
    assertThrows(
        IllegalArgumentException.class,
        () -> new TrajectoryBatch.Collector(Variant.TENHOU_3P).add(games.getFirst()));
  }

  /** 判断を {@code decisions} 個（0 個でもよい）持つ半荘。 */
  private GameTrajectory game(long id, int decisions) {
    Observation reader = new Observation(VARIANT);
    GameTrajectory.Builder builder = new GameTrajectory.Builder(VARIANT, id);
    builder.hand(null);
    for (int index = 0; index < decisions; index++) {
      builder.decision(pool.bind(index % 4, index, reader), 0, 1f, true);
    }
    return builder.finish(SCORES);
  }

  /** 判断の記録を表す long のワードの列。 */
  private static long[] words(Observation o) {
    ObservationBatch copy = new ObservationBatch(VARIANT, 64);
    int index = copy.add(o);
    return Arrays.copyOfRange(copy.words(), copy.start(index), copy.start(index) + copy.longs(index));
  }
}
