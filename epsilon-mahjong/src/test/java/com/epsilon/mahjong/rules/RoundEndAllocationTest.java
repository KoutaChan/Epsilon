package com.epsilon.mahjong.rules;

import static com.epsilon.mahjong.rules.RoundOutcome.NO_SEAT;

import com.epsilon.mahjong.util.AllocationProbe;
import org.testng.annotations.Test;

/** 局の結果を書き、精算し、次局へ進め、点の配列で終局を決める処理は、ウォームアップの後はメモリを確保しない。 */
public class RoundEndAllocationTest {

  private static final int WARM_UP = 50_000;
  private static final int MEASURED = 200_000;

  @Test
  public void writingSettlingAndAdvancingDoNotAllocate() {
    TableState start = RoundFixtures.table(Variant.TENHOU_4P, 7, 1, 1, 31_000, 20_000, 26_000, 22_000);
    TableState next = new TableState(Variant.TENHOU_4P);
    RoundOutcome outcome = new RoundOutcome(Variant.TENHOU_4P);
    int[] delta = new int[Variant.TENHOU_4P.seats()];
    int[] judged = new int[Variant.TENHOU_4P.seats()];
    AllocationProbe.assertNoAllocation(WARM_UP, MEASURED, count -> run(count, start, outcome, delta, judged, next));
  }

  private static long run(
      int count, TableState start, RoundOutcome outcome, int[] delta, int[] judged, TableState next) {
    long checksum = 0;
    for (int i = 0; i < count; i++) {
      switch (i & 3) {
        case 0 -> outcome.setTsumo(ScoreMath.base(1 + (i >> 2) % 4, 30, 0), (i >> 2) % 4, NO_SEAT, 0b0001);
        case 1 -> {
          outcome.setRon(2_000, 1, 2, NO_SEAT, 0);
          outcome.addRonWinner(8_000, 3, 0);
        }
        case 2 -> outcome.setExhaustive((i >> 2) & 0b1111, 0, 0b0010);
        default -> outcome.setAbortive(AbortiveDraw.FOUR_WINDS, 0);
      }
      Settlement.apply(start.variant(), start, outcome, delta);
      Progression.next(start, outcome, delta, next);
      checksum += next.score(0) + next.round() + (Progression.finishes(next, DealerOutcome.PASSES) ? 1 : 0);
      for (int seat = 0; seat < judged.length; seat++) {
        judged[seat] = Progression.judgedScore(start, delta, seat);
      }
      checksum += Progression.finishes(start.variant(), start.round(), judged, DealerOutcome.CONTINUES) ? 1 : 0;
    }
    return checksum;
  }
}
