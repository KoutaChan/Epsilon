package com.epsilon.grp;

import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** RoundHistory を確かめる。行の読み書き、コピーの独立とバージョン、容量を超えた伸び、メモリを確保しないこと、拒む卓。 */
public class RoundHistoryTest {

  @Test
  public void rowsKeepTheTablesInOrder() {
    RoundHistory history = new RoundHistory(Variant.TENHOU_4P);
    Assert.assertEquals(history.size(), 0);
    Assert.assertEquals(history.stamp(), 0);
    history.add(new TableState(Variant.TENHOU_4P));
    history.add(new GrpState(Variant.TENHOU_4P, 5, 2, 1, new int[] {30_000, 20_000, 26_000, 23_000}));
    Assert.assertEquals(history.size(), 2);
    Assert.assertEquals(history.round(0), 0);
    Assert.assertEquals(history.honba(0), 0);
    Assert.assertEquals(history.kyotaku(0), 0);
    for (int seat = 0; seat < 4; seat++) {
      Assert.assertEquals(history.score(0, seat), 25_000);
    }
    Assert.assertEquals(history.round(1), 5);
    Assert.assertEquals(history.honba(1), 2);
    Assert.assertEquals(history.kyotaku(1), 1);
    Assert.assertEquals(history.score(1, 0), 30_000);
    Assert.assertEquals(history.score(1, 2), 26_000);
    Assert.assertEquals(history.score(1, 3), 23_000);
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> history.round(2));
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> history.score(0, 4));
  }

  /** コピーは同じ中身とバージョンを持ち、どちらに足しても、空にしても、もう片方は変わらない。足すたびにバージョンは新しくなる。 */
  @Test
  public void copiesAreIndependentAndShareTheStamp() {
    Random random = new Random(3);
    RoundHistory history = GrpFixtures.history(Variant.TENHOU_3P, random, 3);
    List<GrpState> rows = rows(history);
    RoundHistory copy = history.copy();
    Assert.assertEquals(copy.stamp(), history.stamp());
    Assert.assertEquals(rows(copy), rows);

    long before = history.stamp();
    history.add(GrpFixtures.state(Variant.TENHOU_3P, random));
    Assert.assertNotEquals(history.stamp(), before);
    Assert.assertEquals(copy.stamp(), before);
    Assert.assertEquals(rows(copy), rows);

    copy.add(GrpFixtures.state(Variant.TENHOU_3P, random));
    Assert.assertNotEquals(copy.stamp(), before);
    Assert.assertNotEquals(copy.stamp(), history.stamp());
    Assert.assertEquals(history.size(), 4);
    Assert.assertEquals(rows(history).subList(0, 3), rows);

    RoundHistory target = new RoundHistory(Variant.TENHOU_3P);
    target.add(GrpFixtures.state(Variant.TENHOU_3P, random));
    target.copyFrom(history);
    Assert.assertEquals(target.stamp(), history.stamp());
    Assert.assertEquals(rows(target), rows(history));
    List<GrpState> copied = rows(target);
    history.clear();
    Assert.assertEquals(history.size(), 0);
    Assert.assertEquals(history.stamp(), 0);
    Assert.assertEquals(rows(target), copied);
  }

  @Test
  public void historiesGrowPastTheInitialCapacity() {
    Random random = new Random(5);
    RoundHistory history = new RoundHistory(Variant.TENHOU_4P);
    List<GrpState> added = new ArrayList<>();
    for (int round = 0; round < 100; round++) {
      GrpState state = GrpFixtures.state(Variant.TENHOU_4P, random);
      history.add(state);
      added.add(state);
    }
    Assert.assertEquals(rows(history), added);
    Assert.assertEquals(rows(history.copy()), added);
    RoundHistory small = new RoundHistory(Variant.TENHOU_4P);
    small.copyFrom(history);
    Assert.assertEquals(rows(small), added);
  }

  /** 容量の内なら、局を足すこと・分岐のために写すこと・空にすることはメモリを確保しない。 */
  @Test
  public void addingAndForkingDoNotAllocate() {
    Random random = new Random(7);
    GrpState[] states = new GrpState[20];
    for (int index = 0; index < states.length; index++) {
      states[index] = GrpFixtures.state(Variant.TENHOU_4P, random);
    }
    RoundHistory history = new RoundHistory(Variant.TENHOU_4P);
    RoundHistory fork = new RoundHistory(Variant.TENHOU_4P);
    TableState table = new TableState(Variant.TENHOU_4P);
    Allocations.assertSteady(2 * states.length + " rounds with forks", 10, () -> {
      history.clear();
      for (GrpState state : states) {
        state.copyInto(table);
        history.add(table);
        history.add(state);
        fork.copyFrom(history);
      }
    });
    Assert.assertEquals(fork.size(), 2 * states.length);
  }

  @Test
  public void finishedTablesAndOtherVariantsAreRefused() {
    RoundHistory history = new RoundHistory(Variant.TENHOU_4P);
    TableState finished = GrpFixtures.finished();
    Assert.assertTrue(finished.gameOver(), finished.toString());
    Assert.assertThrows(IllegalArgumentException.class, () -> history.add(finished));
    Assert.assertThrows(IllegalArgumentException.class, () -> history.add(new TableState(Variant.TENHOU_3P)));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> history.add(new GrpState(Variant.TENHOU_3P, 0, 0, 0, new int[] {35_000, 35_000, 35_000})));
    Assert.assertThrows(IllegalArgumentException.class, () -> history.copyFrom(new RoundHistory(Variant.TENHOU_3P)));
    Assert.assertEquals(history.size(), 0);
  }

  /** 比べるために、履歴の行を GrpState にして並べる。 */
  private static List<GrpState> rows(RoundHistory history) {
    List<GrpState> rows = new ArrayList<>();
    int seats = history.variant().seats();
    for (int index = 0; index < history.size(); index++) {
      int[] scores = new int[seats];
      for (int seat = 0; seat < seats; seat++) {
        scores[seat] = history.score(index, seat);
      }
      rows.add(new GrpState(history.variant(), history.round(index), history.honba(index), history.kyotaku(index), scores));
    }
    return rows;
  }
}
