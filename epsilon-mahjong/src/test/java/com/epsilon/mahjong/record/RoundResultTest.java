package com.epsilon.mahjong.record;

import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.score.ScoringYaku;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 牌譜の局の結果が、放銃者を包の席にした記録を拒むことを確かめる。 */
public class RoundResultTest {

  private static final long DAISANGEN = ScoringYaku.DAISANGEN.bit();

  /**
   * 席 1 が席 0 から大三元をロンする。包の席が放銃者と同じなら包の無いロンとして書く決まりなので、放銃者を包の席にした記録は
   * IllegalArgumentException になる。ほかの席の包はそのまま残る。
   */
  @Test
  public void aRecordedRonCannotMakeTheDiscarderLiable() {
    int[] plainDeltas = {-32_000, 32_000, 0, 0};
    RoundResult.Win liable = new RoundResult.Win(1, 8_000, DAISANGEN, 0, 0, 0, 0, 0, 0);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> new RoundResult(EndKind.RON, null, List.of(liable), 0, 0, 0, plainDeltas, new int[0]));

    RoundResult.Win plain = new RoundResult.Win(1, 8_000, DAISANGEN, 0, 0, 0, 0, 0, RoundOutcome.NO_SEAT);
    RoundResult result = new RoundResult(EndKind.RON, null, List.of(plain), 0, 0, 0, plainDeltas, new int[0]);
    Assert.assertEquals(result.wins().get(0).pao(), RoundOutcome.NO_SEAT);

    int[] splitDeltas = {-16_000, 32_000, -16_000, 0};
    RoundResult.Win fed = new RoundResult.Win(1, 8_000, DAISANGEN, 0, 0, 0, 0, 0, 2);
    RoundResult split = new RoundResult(EndKind.RON, null, List.of(fed), 0, 0, 0, splitDeltas, new int[0]);
    Assert.assertEquals(split.wins().get(0).pao(), 2);
  }
}
