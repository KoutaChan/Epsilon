package com.epsilon.mahjong.shape;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.mahjong.util.AllocationProbe;
import org.testng.annotations.Test;

/** ウォームアップの後、牌の出し入れ・解析・分解はメモリを確保しない。 */
public class ShapeAllocationTest {

  private static final int WARM_UP = 20_000;
  private static final int MEASURED = 100_000;

  @Test
  public void handAnalysisAndDecompositionAllocateNothingOnceWarm() {
    AllocationProbe.assertNoAllocation(WARM_UP, MEASURED, new Workload()::run);
  }

  /** 四人打ちと三人打ちの聴牌・1 向聴の手で、各メソッドを一通り呼ぶ。 */
  private static final class Workload {

    private final ShapeAnalyzer four = new ShapeAnalyzer(Variant.TENHOU_4P);
    private final ShapeAnalyzer three = new ShapeAnalyzer(Variant.TENHOU_3P);
    private final HandShape[] fourHands = {
      ShapeFixtures.shape("1112345678999m", 0),
      ShapeFixtures.shape("123m456p789s11z23s", 0),
      ShapeFixtures.shape("13m456p789s11z237s", 0)
    };
    private final HandShape[] threeHands = {
      ShapeFixtures.shape("11m99m123p456p789s", 0), ShapeFixtures.shape("1111m9999m55p", 1)
    };
    private final WinningDecomposition decomposition = new WinningDecomposition();
    private final byte[] visible = new byte[Tiles.KINDS];

    long run(int rounds) {
      long checksum = 0;
      for (int round = 0; round < rounds; round++) {
        checksum += analyze(four, fourHands[round % fourHands.length], Tiles.S1);
        checksum += analyze(three, threeHands[round % threeHands.length], Tiles.M1);
      }
      return checksum;
    }

    private long analyze(ShapeAnalyzer analyzer, HandShape hand, int kind) {
      long checksum = analyzer.shanten(hand) + analyzer.standardShanten(hand) + analyzer.acceptance(hand, visible);
      long waits = analyzer.waits(hand);
      checksum += waits ^ analyzer.improving(hand);
      for (long rest = waits; rest != 0; rest &= rest - 1) {
        int winning = Long.numberOfTrailingZeros(rest);
        decomposition.bind(hand, winning);
        while (decomposition.next()) {
          checksum += decomposition.waits(winning) + decomposition.triplets() + decomposition.runs();
        }
      }
      if (hand.count(kind) < Tiles.COPIES) {
        hand.add(kind);
        checksum += analyzer.shanten(hand);
        hand.remove(kind);
      }
      return checksum;
    }
  }
}
