package com.epsilon.grp.train;

import com.epsilon.grp.GrpKey;
import com.epsilon.grp.GrpState;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 学習の行の分け方（学習用、早期終了用の検証、{@code GUARDED} のときの関門用）を確かめる。行の中身と重み、GrpKey で表せない局末の
 * 扱いも見る。
 */
public class GrpRowsTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  @Test
  public void rowsFollowTheSlicesAndTheWindow() {
    RecencyWindow window = new RecencyWindow(12, 1, 64);
    Map<Integer, List<GameRecord>> cohorts = cohorts(window);
    for (GateCohort gate : GateCohort.values()) {
      GrpRows.Builder builder = GrpRows.builder(VARIANT, window, gate);
      cohorts.forEach(builder::add);
      GrpRows.Split split = builder.build();
      List<String> train = new ArrayList<>();
      List<String> stop = new ArrayList<>();
      for (int cohort = window.first(); cohort <= window.newest(); cohort++) {
        boolean gateCohort = gate == GateCohort.GUARDED && cohort == window.newest();
        for (GameRecord game : cohorts.get(cohort)) {
          GrpSlice slice = gateCohort ? GrpSlice.guarded(game.gameId()) : GrpSlice.of(game.gameId());
          List<String> into = switch (slice) {
            case TRAIN -> train;
            case STOP -> cohort > window.newest() - GrpRows.STOP_COHORTS ? stop : null;
            case GUARD -> null;
          };
          for (int handEnd = 0; into != null && handEnd < game.handEnds().size(); handEnd++) {
            into.add(row(game.handEnds().get(handEnd).key(), game.realizedPermutation(handEnd), cohort));
          }
        }
      }
      Assert.assertEquals(rows(split.train()), train, gate.toString());
      Assert.assertEquals(rows(split.stop()), stop, gate.toString());
      Assert.assertEquals(split.train().excluded(), 0);
      for (int row = 0; row < split.train().size(); row++) {
        int cohort = split.train().cohort(row);
        Assert.assertEquals(split.train().weight(row), (float) window.weight(cohort));
        GrpState state = GrpState.ofKey(VARIANT, split.train().key(row));
        Assert.assertEquals(state.key(), split.train().key(row));
      }
    }
  }

  @Test
  public void guardedCohortsKeepTheirGuardGamesForTheGate() {
    RecencyWindow window = new RecencyWindow(9, 1, 64);
    List<GameRecord> games = TrainFixtures.simulated(VARIANT, 400, 9_000, 3);
    List<GameRecord> gate = GateCohort.GUARDED.gateGames(games);
    Assert.assertEquals(GateCohort.GUARDED.cohort(window), 9);
    Assert.assertEquals(GateCohort.NEXT.cohort(window), 10);
    Assert.assertEquals(GateCohort.NEXT.gateGames(games), games);
    Assert.assertTrue(gate.size() > 50 && gate.size() < 150, gate.size() + " guard games");
    for (GameRecord game : gate) {
      Assert.assertEquals(GrpSlice.guarded(game.gameId()), GrpSlice.GUARD);
    }
    GrpRows.Split split = GrpRows.builder(VARIANT, window, GateCohort.GUARDED).add(9, games).build();
    int guarded = 0;
    for (GameRecord game : gate) {
      guarded += game.handEnds().size();
    }
    int all = 0;
    for (GameRecord game : games) {
      all += game.handEnds().size();
    }
    Assert.assertEquals(split.train().size() + split.stop().size(), all - guarded);
  }

  @Test
  public void tablesOutsideTheKeyAreCountedAndLeftOut() {
    GrpState deep = new GrpState(VARIANT, 5, 64, 0, new int[] {25_000, 25_000, 25_000, 25_000});
    GrpState plain = new GrpState(VARIANT, 6, 0, 0, new int[] {30_000, 25_000, 25_000, 20_000});
    Assert.assertEquals(deep.key(), GrpKey.NONE);
    int[] finalScores = {40_000, 30_000, 20_000, 10_000};
    GameRecord game = new GameRecord(VARIANT, idOf(GrpSlice.TRAIN), List.of(deep, plain), finalScores);
    GrpRows rows = GrpRows.builder(VARIANT, new RecencyWindow(5, 1, 8), GateCohort.NEXT)
        .add(5, List.of(game))
        .build()
        .train();
    Assert.assertEquals(rows.size(), 1);
    Assert.assertEquals(rows.excluded(), 1);
    Assert.assertEquals(rows.key(0), plain.key());
    Assert.assertEquals(rows.totalWeight(), 1.0);
  }

  @Test
  public void buildersCheckTheirCohortsAndGames() {
    RecencyWindow window = new RecencyWindow(10, 1, 64);
    GrpRows.Builder builder = GrpRows.builder(VARIANT, window, GateCohort.NEXT);
    builder.add(10, List.of());
    Assert.assertThrows(IllegalArgumentException.class, () -> builder.add(10, List.of()));
    Assert.assertThrows(IllegalArgumentException.class, () -> builder.add(5, List.of()));
    Assert.assertThrows(IllegalArgumentException.class, () -> builder.add(11, List.of()));
    GameRecord three = new GameRecord(Variant.TENHOU_3P, 1, List.of(), new int[3]);
    Assert.assertThrows(IllegalArgumentException.class, () -> builder.add(9, List.of(three)));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> GrpRows.builder(VARIANT, new RecencyWindow(2, 1, 8), GateCohort.NEXT));
  }

  /** 範囲のコホートごとに 120 半荘を打ち、コホートの昇順に並べて返す。 */
  private static Map<Integer, List<GameRecord>> cohorts(RecencyWindow window) {
    Map<Integer, List<GameRecord>> cohorts = new TreeMap<>();
    for (int cohort = window.first(); cohort <= window.newest(); cohort++) {
      cohorts.put(cohort, TrainFixtures.simulated(VARIANT, 120, cohort * 1_000L, cohort));
    }
    return cohorts;
  }

  private static List<String> rows(GrpRows rows) {
    List<String> listed = new ArrayList<>();
    for (int row = 0; row < rows.size(); row++) {
      listed.add(row(rows.key(row), rows.permutation(row), rows.cohort(row)));
    }
    return listed;
  }

  private static String row(long key, int permutation, int cohort) {
    return Long.toHexString(key) + "/" + permutation + "/" + cohort;
  }

  private static long idOf(GrpSlice slice) {
    long gameId = 0;
    while (GrpSlice.of(gameId) != slice) {
      gameId++;
    }
    return gameId;
  }
}
