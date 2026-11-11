package com.epsilon.prism.luck;

import com.epsilon.grp.Phi;
import com.epsilon.mahjong.engine.ChanceKind;
import com.epsilon.mahjong.engine.ChanceLog;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.Scenes;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.noise.UraSettings;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 手で作った牌山の局で、立直の和了の局末の値を確かめる。槍槓の局と、槓ドラの表示が保留のまま残る局も含む。 */
public class UraExpectationSceneTest {

  private static final double EPSILON = 1e-12;

  private static final int WINNER = 2;

  private static final UraSettings REALIZED = new UraSettings.Realized(0.05), EXPECTED = new UraSettings.Expected();

  private final Phi phi = Phi.scoreSmooth(UraScenes.VARIANT, RankUtility.TENHOU);

  /** 槓の無い立直のロン: 裏ドラ表示牌は 1 枚で、期待値は数え上げの基準と一致する。基本点の組は 3 つ（裏が 0 枚・1 枚・2 枚以上で満貫）。 */
  @Test
  public void aPlainRiichiRonAveragesOverOneIndicator() {
    UraScenes.Played played = UraScenes.plainRon();
    UraExpectation values = evaluate(played);
    Assert.assertTrue(values.averaged());
    Assert.assertFalse(values.kan());
    Assert.assertEquals(played.engine().roundEnd().score(0).ura(), 1);
    Assert.assertEquals(played.engine().full().seat(0).indicatorCount(), 1);
    Assert.assertEquals(values.groups(), 3);
    assertMatchesReference(played, values, 1);
  }

  /** 槍槓の局: 槍槓された加槓は面子にならないが、局に槓があったと数える。裏ドラ表示牌は 1 枚のまま。 */
  @Test
  public void aChankanCountsAsARoundWithAKan() {
    UraScenes.Played played = UraScenes.chankan();
    RoundEnd end = played.engine().roundEnd();
    Assert.assertTrue(end.score(0).has(ScoringYaku.CHANKAN));
    TurnEvent last = played.engine().full().seat(0).last();
    Assert.assertEquals(last.kind(), TurnEvent.Kind.KAN);
    Assert.assertFalse(last.doraPending());
    UraExpectation values = evaluate(played);
    Assert.assertTrue(values.averaged());
    Assert.assertTrue(values.kan());
    assertMatchesReference(played, values, 1);
  }

  /**
   * 槓ドラの表示が保留のまま残る局: 1 つ目の加槓の槓ドラは、打牌の前に次の加槓が槍槓されて局が終わったので、めくられない。裏ドラ
   * 表示牌はめくった 1 枚だけで、保留の表示牌はプールに残る。
   */
  @Test
  public void aPendingKanIndicatorStaysInThePool() {
    UraScenes.Played played = UraScenes.chankanWithPendingIndicator();
    FullView full = played.engine().full();
    Assert.assertTrue(played.engine().roundEnd().score(0).has(ScoringYaku.CHANKAN));
    Assert.assertTrue(full.seat(0).last().doraPending());
    Assert.assertEquals(full.seat(0).indicatorCount(), 1);
    int pending = Scenes.faces(UraScenes.PENDING_INDICATOR)[0].kind();
    Assert.assertEquals(full.unseen(pending), Tiles.COPIES, "the pending indicator is seen by nobody");
    UraExpectation values = evaluate(played);
    Assert.assertTrue(values.averaged());
    Assert.assertTrue(values.kan());
    assertMatchesReference(played, values, 1);
  }

  /** 期待値を求めない立直の和了: 期待値は実際の値と同じで、分散は 0。期待値で学ぶときの値を求めると IllegalStateException を投げる。 */
  @Test
  public void aRiichiWinWithoutAveragingRefusesTheExpectedValue() {
    UraScenes.Played played = UraScenes.plainRon();
    UraExpectation values = new UraExpectation(phi);
    Assert.assertFalse(values.evaluate(played.engine().full(), played.engine().roundEnd(), played.start(), null, false));
    Assert.assertTrue(values.riichiWin());
    Assert.assertFalse(values.averaged());
    Assert.assertEquals(values.groups(), 0);
    double[] out = new double[UraScenes.VARIANT.seats()];
    values.values(REALIZED, out);
    for (int seat = 0; seat < out.length; seat++) {
      Assert.assertEquals(out[seat], values.actual(seat));
      Assert.assertEquals(values.expected(seat), values.actual(seat));
      Assert.assertEquals(values.variance(seat), 0.0);
    }
    Assert.assertThrows(IllegalStateException.class, () -> values.values(EXPECTED, out));
  }

  /** ラベルの作り方ごとの値: 実際の値で学ぶなら実際の値を、期待値で学ぶなら期待値を書く。 */
  @Test
  public void valuesFollowTheMode() {
    UraExpectation values = evaluate(UraScenes.plainRon());
    double[] out = new double[UraScenes.VARIANT.seats()];
    values.values(REALIZED, out);
    for (int seat = 0; seat < out.length; seat++) {
      Assert.assertEquals(out[seat], values.actual(seat));
    }
    values.values(EXPECTED, out);
    for (int seat = 0; seat < out.length; seat++) {
      Assert.assertEquals(out[seat], values.expected(seat));
    }
    Assert.assertNotEquals(values.expected(WINNER), values.actual(WINNER));
  }

  private UraExpectation evaluate(UraScenes.Played played) {
    UraExpectation values = new UraExpectation(phi);
    Assert.assertTrue(values.evaluate(played.engine().full(), played.engine().roundEnd(), played.start(), null, true));
    return values;
  }

  /** 期待値・分散・実際の値が数え上げの基準と一致し、裏ドラ表示牌の数がめくった表示牌の数と一致することを確かめる。 */
  private void assertMatchesReference(UraScenes.Played played, UraExpectation values, int draws) {
    FullView full = played.engine().full();
    RoundEnd end = played.engine().roundEnd();
    UraReference reference = new UraReference(phi);
    Assert.assertTrue(reference.compute(full, end, played.start()));
    Assert.assertEquals(reference.uraEvents, draws);
    Assert.assertEquals(reference.uraEvents, 1 + reference.indicatorEvents);
    Assert.assertEquals(reference.realizedUra(0), end.score(reference.varying(0)).ura());
    Assert.assertTrue(reference.normalized());
    Assert.assertEquals(values.groups(), reference.groups());
    double[] next = new double[full.variant().seats()];
    phi.values(null, end.next(), next);
    for (int seat = 0; seat < next.length; seat++) {
      Assert.assertEquals(values.actual(seat), next[seat]);
      Assert.assertEquals(reference.realized[seat], next[seat], EPSILON);
      Assert.assertEquals(values.expected(seat), reference.expected[seat], EPSILON);
      Assert.assertEquals(values.variance(seat), reference.variance[seat], EPSILON);
    }
    Assert.assertTrue(values.variance(WINNER) > 0);
    ChanceLog log = full.chance();
    Assert.assertEquals(log.kind(log.size() - 1), ChanceKind.URA);
  }
}
