package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.mahjong.util.AllocationProbe;
import java.util.Random;
import org.testng.annotations.Test;

/** ウォームアップの後は、状態の書き換え・読み・コピーがメモリを確保しない。 */
public class StateAllocationTest {

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
      checksum += view.visible(Tiles.P5) + view.riverKinds(view.seat()) + view.riichi(view.seat()).ordinal();
      checksum += view.indicator(view.indicatorCount() - 1) + view.meldCount(view.seat()) + (view.furiten() ? 1 : 0);
      return checksum;
    }
  }
}
