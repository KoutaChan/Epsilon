package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.util.AllocationProbe;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * ウォームアップの後は、エンジンがメモリを確保しないことを確かめる。半荘を始めて進め、手を確定する処理（局末と終局も含む）、
 * FullView から手牌をコピーする処理、ツモ和了したと仮定した点数の計算が対象。四人打ちと三人打ちの両方で確かめる。
 */
public class EngineAllocationTest {

  private static final int WARM_UP = 20_000;
  private static final int MEASURED = 100_000;

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test(dataProvider = "variants")
  public void stepsAllocateNothingOnceWarm(Variant variant) {
    RandomGames games = RandomGames.start(variant, 11);
    AllocationProbe.assertNoAllocation(
        WARM_UP,
        MEASURED,
        steps -> {
          long checksum = 0;
          for (int step = 0; step < steps; step++) {
            checksum += games.step().ordinal();
          }
          return checksum;
        });
  }

  /** 1 ステップごとに 1 席の手牌をコピーする。手番の判断の時点では、その席が最後に引いた牌でツモ和了したと仮定した点数も求める。 */
  @Test(dataProvider = "variants")
  public void fullViewReadsAllocateNothingOnceWarm(Variant variant) {
    RandomGames games = RandomGames.start(variant, 13);
    RuleWorkspace ws = new RuleWorkspace(variant);
    HandShape hand = new HandShape();
    WinScore score = new WinScore();
    AllocationProbe.assertNoAllocation(
        WARM_UP,
        MEASURED,
        steps -> {
          long checksum = 0;
          for (int step = 0; step < steps; step++) {
            FullView full = games.engine.full();
            full.copyHand(step % variant.seats(), hand);
            checksum += hand.size() + hand.melds();
            if (games.boundary() == Boundary.DECISIONS
                && games.engine.decisions().kind(0) == EngineDecisionKind.TURN) {
              int seat = games.engine.decisions().seat(0);
              checksum += full.tsumoScore(seat, full.seat(seat).last().tile(), ws, score) ? score.base() : 0;
            }
            checksum += games.step().ordinal();
          }
          return checksum;
        });
  }
}
