package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.mahjong.util.AllocationProbe;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 局の途中で状態全体をコピーすると、どのフィールドも元と等しく、元と何も共有しない。同じ手で進めれば同じ状態になり、コピーを進めても元は
 * 変わらない。
 */
public class StateCopyTest {

  @Test
  public void aCopyEqualsTheOriginalSharesNothingAndContinuesTheSame() {
    Random random = new Random(64);
    for (Variant variant : Variant.values()) {
      for (int trial = 0; trial < 30; trial++) {
        RandomRound original = new RandomRound(variant);
        original.start(
            RoundStart.first(variant),
            random.nextLong(),
            1 + trial,
            trial % 2 == 0 ? DrawOrder.SEAT_STREAMS : DrawOrder.WALL_ORDER);
        int turns = random.nextInt(25);
        for (int turn = 0; turn < turns; turn++) {
          original.turn(random);
        }
        RandomRound copy = new RandomRound(variant);
        copy.copyFrom(original);
        Fields.assertSameFields(original, copy);
        RandomRound snapshot = new RandomRound(variant);
        snapshot.copyFrom(original);

        long rest = random.nextLong();
        playIntoTheNextRound(copy, new Random(rest));
        Fields.assertSameFields(snapshot, original);
        playIntoTheNextRound(original, new Random(rest));
        Fields.assertSameFields(original, copy);
      }
    }
  }

  @Test
  public void copiesNeedTheSameVariant() {
    RandomRound four = new RandomRound(Variant.TENHOU_4P);
    RandomRound three = new RandomRound(Variant.TENHOU_3P);
    Assert.assertThrows(IllegalArgumentException.class, () -> four.wall.copyFrom(three.wall));
    Assert.assertThrows(IllegalArgumentException.class, () -> four.state.copyFrom(three.state));
    Assert.assertThrows(IllegalArgumentException.class, () -> four.chance.copyFrom(three.chance));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> four.state.dora().copyFrom(three.state.dora()));
  }

  /** 局を終えて裏ドラ表示牌をめくり、次の牌山を配って数手進める（コピーも牌山の epoch を引き継ぐ）。 */
  private static void playIntoTheNextRound(RandomRound round, Random random) {
    round.finish(random);
    round.revealUra();
    round.deal(RoundStart.first(round.variant));
    for (int turn = 0; turn < 5; turn++) {
      round.turn(random);
    }
  }

  private static final int WARM_UP = 20_000;
  private static final int MEASURED = 100_000;

  @Test
  public void stateChangesViewsAndCopiesAllocateNothingOnceWarm() {
    for (Variant variant : Variant.values()) {
      AllocationProbe.assertNoAllocation(WARM_UP, MEASURED, new Workload(variant)::run);
    }
  }

  /**
   * 1 回の処理で手番を 1 つ進める（ツモ・槓と嶺上・打牌・立直・鳴き・槓ドラ・判断）。局が終われば裏ドラ表示牌をめくり、次の牌山を配る。
   * 8 手番に 1 回は状態全体をコピーし、全ての席のビューは毎手番読む。
   */
  private static final class Workload {

    private final RandomRound round;
    private final RandomRound copy;
    private final RoundStart start;
    private final Random random = new Random(3);
    private final SeatView[] views;

    Workload(Variant variant) {
      round = new RandomRound(variant);
      copy = new RandomRound(variant);
      start = RoundStart.first(variant);
      round.start(start, 17, 1, DrawOrder.SEAT_STREAMS);
      views = new SeatView[variant.seats()];
      for (int seat = 0; seat < views.length; seat++) {
        views[seat] = new SeatReader(round.state, round.hands, round.decisions, seat);
      }
    }

    long run(int turns) {
      long checksum = 0;
      for (int turn = 0; turn < turns; turn++) {
        if (!round.turn(random)) {
          round.revealUra();
          round.deal(start);
        }
        if (turn % 8 == 0) {
          copy.copyFrom(round);
        }
        for (SeatView view : views) {
          checksum += read(view);
        }
        checksum += round.chance.unseen(turn % Tiles.KINDS) + round.wall.liveLeft();
      }
      return checksum;
    }

    private static long read(SeatView view) {
      long checksum = view.waits() + view.liveLeft() + view.events().size() + view.last().tile();
      checksum +=
          view.visible(Tiles.P5)
              + view.riverKinds(view.seat())
              + view.riichi(view.seat()).ordinal();
      checksum +=
          view.indicator(view.indicatorCount() - 1)
              + view.meldCount(view.seat())
              + (view.furiten() ? 1 : 0);
      return checksum;
    }
  }
}
