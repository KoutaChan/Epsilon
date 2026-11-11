package com.epsilon.mahjong.score;

import static com.epsilon.mahjong.score.WinCase.four;
import static com.epsilon.mahjong.score.WinCase.three;

import com.epsilon.mahjong.score.WinContext.Source;
import com.epsilon.mahjong.util.AllocationProbe;
import org.testng.annotations.Test;

/** ウォームアップの後、採点はメモリを確保しない。 */
public class ScoreAllocationTest {

  private static final int WARM_UP = 20_000;
  private static final int MEASURED = 100_000;

  @Test
  public void scoringAllocatesNothingOnceWarm() {
    AllocationProbe.assertNoAllocation(WARM_UP, MEASURED, new Workload()::run);
  }

  /** 四人打ちと三人打ちの、通常形・七対子・国士・副露・北抜き・役の無い手を順に採点する。 */
  private static final class Workload {

    private final HandScorer scorer = new HandScorer();
    private final WinScore score = new WinScore();
    private final WinContext[] contexts = {
      four("123m456p789p55s23s").riichi().ron("1s").dora("4s").ura("1z").red("5p").context(),
      four("1199m2255p3377s4z").tsumo("4z").dora("1m").context(),
      four("19m19p19s1234567z").dealer().firstDraw().tsumo("7z").context(),
      four("567m345p66p78s").chi("234m").ron("6s").context(),
      four("111m333p555s77z99s").ron("9s").context(),
      four("123m456m55p78s").ankan("9p").win("9s", Source.RINSHAN).dora("8p").context(),
      three("123p456p789s44z23s").riichi().ron("4s").nuki(1).dora("3z").ura("3z").context(),
      four("123m456p789p55s13s").ron("2s").context()
    };

    long run(int rounds) {
      long checksum = 0;
      for (int round = 0; round < rounds; round++) {
        if (scorer.score(contexts[round % contexts.length], score)) {
          checksum += score.base() + score.han() + score.fu() + score.yaku();
        }
      }
      return checksum;
    }
  }
}
