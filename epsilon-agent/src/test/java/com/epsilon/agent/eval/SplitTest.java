package com.epsilon.agent.eval;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.policy.Policy;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 席の分け方（1v3 など）と、卓の計画の割り当てのテスト。割り当ての数は打ち手ごとの席の数から決まる多項係数で、どの割り当ても並びが
 * 違う。どの打ち手も各席に同じ回数ずつ座る。1v(S − 1) の割り当て p では候補が席 p に座る。1 半荘ずつの卓では、牌山の番号で候補の席を回す。
 */
public class SplitTest {

  /** 手を選ばない方策。計画を確かめるだけに使う。 */
  record Idle(Variant variant, String label) implements InlinePolicy {

    @Override
    public int act(Observation observation) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Evaluation evaluate(Observation observation) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }

  private static List<Policy> slots(Variant variant, int count) {
    List<Policy> slots = new ArrayList<>(count);
    for (int slot = 0; slot < count; slot++) {
      slots.add(new Idle(variant, "idle:" + slot));
    }
    return slots;
  }

  private static ArenaPlan plan(Variant variant, Seating seating, List<Split> splits, int slots, int walls) {
    return new ArenaPlan(
        variant,
        slots(variant, slots),
        seating,
        splits,
        DrawOrder.SEAT_STREAMS,
        WallSet.FIXED,
        11L,
        100L,
        walls,
        RankUtility.TENHOU);
  }

  private static void assertPlacements(Variant variant, String text, int slots, int expected) {
    Split split = Split.parse(text, slots, variant);
    Assert.assertEquals(split.toString(), text);
    Assert.assertEquals(split.placements(), expected, text);
    int seats = variant.seats();
    Set<List<Integer>> seen = new HashSet<>();
    int[][] sitting = new int[slots][seats];
    for (int placement = 0; placement < expected; placement++) {
      int[] slotOfSeat = new int[seats];
      split.slotsOf(placement, slotOfSeat);
      int[] perSlot = new int[slots];
      for (int seat = 0; seat < seats; seat++) {
        perSlot[slotOfSeat[seat]]++;
        sitting[slotOfSeat[seat]][seat]++;
      }
      for (int slot = 0; slot < slots; slot++) {
        Assert.assertEquals(perSlot[slot], split.seatsOf(slot), text + " placement " + placement);
      }
      Assert.assertTrue(seen.add(Arrays.stream(slotOfSeat).boxed().toList()), text + " repeats placement " + placement);
      if (slots == 2 && split.seatsOf(0) == 1) {
        Assert.assertEquals(slotOfSeat[placement], 0, text + " puts the candidate at seat " + placement);
      }
    }
    for (int slot = 0; slot < slots; slot++) {
      for (int seat = 0; seat < seats; seat++) {
        Assert.assertEquals(
            sitting[slot][seat], expected * split.seatsOf(slot) / seats, text + " slot " + slot + " seat " + seat);
      }
    }
  }

  @Test
  public void placementsAreEveryArrangementAndEverySlotSitsEverySeatEqually() {
    assertPlacements(Variant.TENHOU_4P, "1v3", 2, 4);
    assertPlacements(Variant.TENHOU_4P, "3v1", 2, 4);
    assertPlacements(Variant.TENHOU_4P, "2v2", 2, 6);
    assertPlacements(Variant.TENHOU_4P, "1v1v2", 3, 12);
    assertPlacements(Variant.TENHOU_4P, "1v1v1v1", 4, 24);
    assertPlacements(Variant.TENHOU_3P, "1v2", 2, 3);
    assertPlacements(Variant.TENHOU_3P, "2v1", 2, 3);
    assertPlacements(Variant.TENHOU_3P, "1v1v1", 3, 6);
    Assert.assertEquals(Split.oneAgainstRest(Variant.TENHOU_3P), Split.parse("1v2", 2, Variant.TENHOU_3P));
    Assert.assertEquals(Split.oneAgainstRest(Variant.TENHOU_4P), Split.parse("1v3", 2, Variant.TENHOU_4P));
  }

  @Test
  public void splitsAreValidated() {
    for (String text : List.of("1v2", "0v4", "1v", "v3", "1x3", "1v3v", "a1v3")) {
      Assert.assertThrows(IllegalArgumentException.class, () -> Split.parse(text, 2, Variant.TENHOU_4P));
    }
    Assert.assertThrows(IllegalArgumentException.class, () -> Split.parse("2v2", 3, Variant.TENHOU_4P));
    Assert.assertThrows(IllegalArgumentException.class, () -> Split.parse("1v3", 2, Variant.TENHOU_3P));
  }

  /** 分け方を並べた卓では、分け方の順に割り当てに通し番号を振り、牌山ごとに全ての割り当てを打つ。 */
  @Test
  public void everyPlacementPlansNumberThePlacementsThroughTheSplits() {
    for (Variant variant : List.of(Variant.TENHOU_4P, Variant.TENHOU_3P)) {
      int seats = variant.seats();
      List<Split> splits = List.of(Split.oneAgainstRest(variant), new Split(List.of(seats - 1, 1)));
      ArenaPlan plan = plan(variant, Seating.EVERY_PLACEMENT, splits, 2, 5);
      Assert.assertEquals(plan.layout().count(), 2 * seats);
      Assert.assertEquals(plan.gamesPerWall(), 2 * seats);
      int[][] table = plan.slotTable();
      int[] candidateSeats = new int[seats];
      for (int placement = 0; placement < table.length; placement++) {
        int split = placement / seats;
        Assert.assertEquals(plan.layout().splitOf(placement), split);
        int[] expected = new int[seats];
        splits.get(split).slotsOf(placement % seats, expected);
        Assert.assertEquals(table[placement], expected, "placement " + placement);
        Assert.assertEquals(plan.placement(105, placement), placement);
        for (int seat = 0; seat < seats; seat++) {
          candidateSeats[seat] += table[placement][seat] == 0 ? 1 : 0;
        }
      }
      // 候補は 1v(S − 1) で各席に 1 回、(S − 1)v1 で各席に S − 1 回座る
      for (int seat = 0; seat < seats; seat++) {
        Assert.assertEquals(candidateSeats[seat], seats, variant + " seat " + seat);
      }
      Assert.assertThrows(IndexOutOfBoundsException.class, () -> plan.placement(100, 2 * seats));
    }
    ArenaPlan pairs = plan(Variant.TENHOU_4P, Seating.EVERY_PLACEMENT, List.of(new Split(List.of(2, 2))), 2, 3);
    Assert.assertEquals(pairs.gamesPerWall(), 6);
    ArenaPlan four =
        plan(Variant.TENHOU_4P, Seating.EVERY_PLACEMENT, List.of(Split.parse("1v1v1v1", 4, Variant.TENHOU_4P)), 4, 2);
    Assert.assertEquals(four.gamesPerWall(), 24);
  }

  /**
   * 1 半荘ずつの卓は 1v(S − 1) だけを受け付け、牌山の番号を席の数で割った余りの席に候補を置く。牌山の数が S の倍数なら、候補は
   * 各席に同じ回数座る。
   */
  @Test
  public void byGamePlansRotateTheCandidateByTheWallNumber() {
    for (Variant variant : List.of(Variant.TENHOU_4P, Variant.TENHOU_3P)) {
      int seats = variant.seats();
      int walls = 4 * seats;
      ArenaPlan plan = plan(variant, Seating.BY_GAME, List.of(Split.oneAgainstRest(variant)), 2, walls);
      Assert.assertEquals(plan.gamesPerWall(), 1);
      int[][] table = plan.slotTable();
      int[] candidateSeats = new int[seats];
      for (long wall = plan.wallBase(); wall < plan.wallBase() + walls; wall++) {
        int placement = plan.placement(wall, 0);
        Assert.assertEquals(placement, (int) (wall % seats));
        for (int seat = 0; seat < seats; seat++) {
          Assert.assertEquals(table[placement][seat], seat == placement ? 0 : 1, "wall " + wall);
        }
        candidateSeats[placement]++;
      }
      for (int seat = 0; seat < seats; seat++) {
        Assert.assertEquals(candidateSeats[seat], walls / seats, variant + " seat " + seat);
      }
      Assert.assertThrows(
          IllegalArgumentException.class,
          () -> plan(variant, Seating.BY_GAME, List.of(new Split(List.of(seats - 1, 1))), 2, walls));
    }
    Assert.assertThrows(
        IllegalArgumentException.class,
        () ->
            plan(
                Variant.TENHOU_4P,
                Seating.EVERY_PLACEMENT,
                List.of(Split.oneAgainstRest(Variant.TENHOU_4P), Split.oneAgainstRest(Variant.TENHOU_4P)),
                2,
                1));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> plan(Variant.TENHOU_4P, Seating.EVERY_PLACEMENT, List.of(Split.parse("1v2", 2, Variant.TENHOU_3P)), 2, 1));
  }
}
