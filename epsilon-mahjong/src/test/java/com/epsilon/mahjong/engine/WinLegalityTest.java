package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.Tiles;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 和了の役（嶺上開花・海底・天和のようにツモで決まる役も）、包の席、荒牌での聴牌、流し満貫、{@code FullView} が求めるロンの点のテスト。 */
public class WinLegalityTest {

  /** 1s・4s 待ち、一気通貫。 */
  private static final String WAITS_1S_4S = "123m456m789m23s11z";

  /** 1z 単騎、門前。 */
  private static final String TANKI_EAST = "123m456p789s234s1z";

  /** 和了の形でも役が無ければツモ和了は出ない。ドラと赤は役に数えない。 */
  @Test
  public void aWinNeedsAYakuAndDoraDoNotCount() {
    Scene none = Scene.first().pon(1, 0, "2m").indicator("1m").hand(1, "340p678s99s78p").draw(1, "6p").turn();
    Assert.assertFalse(none.legal(1).contains("TSUMO_AGARI"));
    Scene tanyao = Scene.first().pon(1, 0, "2m").indicator("1m").hand(1, "340p678s44s78p").draw(1, "6p").turn();
    Assert.assertEquals(tanyao.legal(1).get(0), "TSUMO_AGARI");
    WinScore score = tanyao.decisions.win(0);
    Assert.assertTrue(score.has(ScoringYaku.TANYAO));
    Assert.assertEquals(score.dora(), 3);
    Assert.assertEquals(score.aka(), 1);
  }

  /** 嶺上牌でのツモ和了は嶺上開花、山の最後のツモは海底、鳴きで途切れていない親の第一ツモは天和になる。 */
  @Test
  public void theDrawDecidesRinshanHaiteiAndTheFirstDraw() {
    WinScore rinshan = tsumo(Scene.first().river(0, "9m").ankan(0, "5z").hand(0, "123m456p789s1z").rinshan(0, "1z"));
    Assert.assertTrue(rinshan.has(ScoringYaku.RINSHAN));
    Assert.assertFalse(rinshan.has(ScoringYaku.HAITEI));
    WinScore haitei =
        tsumo(Scene.first().river(0, "9m").burn(Scene.VARIANT.liveWall() - 1).hand(0, TANKI_EAST).draw(0, "1z"));
    Assert.assertTrue(haitei.has(ScoringYaku.HAITEI));
    WinScore tenhou = tsumo(Scene.first().hand(0, TANKI_EAST).draw(0, "1z"));
    Assert.assertTrue(tenhou.has(ScoringYaku.TENHOU));
    WinScore later = tsumo(Scene.first().river(0, "9m").hand(0, TANKI_EAST).draw(0, "1z"));
    Assert.assertFalse(later.has(ScoringYaku.TENHOU));
    Assert.assertTrue(later.has(ScoringYaku.MENZEN_TSUMO));
  }

  /**
   * 包は、3 つ目の三元牌か 4 つ目の風牌の面子を鳴かせた席に付く。その面子が暗槓なら包は無い。ロンでその席自身が放銃したときは包は無く、
   * ほかの席の放銃なら包のまま。
   */
  @Test
  public void paoFallsOnTheSeatThatFedTheLastSet() {
    Scene dragons = Scene.first().pon(1, 0, "5z").pon(1, 2, "6z").pon(1, 3, "7z").hand(1, "123m4p").draw(1, "4p");
    WinScore score = tsumo(dragons);
    Assert.assertTrue(score.has(ScoringYaku.DAISANGEN));
    Assert.assertEquals(WinLegality.pao(dragons.full.seat(1), score, RoundOutcome.NO_SEAT), 3);
    Assert.assertEquals(WinLegality.pao(dragons.full.seat(1), score, 3), RoundOutcome.NO_SEAT);
    Assert.assertEquals(WinLegality.pao(dragons.full.seat(1), score, 0), 3);
    Assert.assertEquals(WinLegality.pao(dragons.full.seat(1), score, 2), 3);

    Scene closed = Scene.first().pon(1, 0, "5z").pon(1, 3, "7z").ankan(1, "6z").hand(1, "123m4p").draw(1, "4p");
    score = tsumo(closed);
    Assert.assertTrue(score.has(ScoringYaku.DAISANGEN));
    Assert.assertEquals(WinLegality.pao(closed.full.seat(1), score, RoundOutcome.NO_SEAT), RoundOutcome.NO_SEAT);
    Assert.assertEquals(WinLegality.pao(closed.full.seat(1), score, 3), RoundOutcome.NO_SEAT);

    Scene winds =
        Scene.first().pon(2, 0, "1z").pon(2, 1, "2z").pon(2, 3, "3z").pon(2, 1, "4z").hand(2, "5p").draw(2, "5p");
    score = tsumo(winds);
    Assert.assertTrue(score.has(ScoringYaku.DAISUUSHII));
    Assert.assertEquals(WinLegality.pao(winds.full.seat(2), score, RoundOutcome.NO_SEAT), 1);
    Assert.assertEquals(WinLegality.pao(winds.full.seat(2), score, 1), RoundOutcome.NO_SEAT);
    Assert.assertEquals(WinLegality.pao(winds.full.seat(2), score, 3), 1);

    Scene plain = Scene.first().river(0, "9m").hand(0, TANKI_EAST).draw(0, "1z");
    Assert.assertEquals(WinLegality.pao(plain.full.seat(0), tsumo(plain), RoundOutcome.NO_SEAT), RoundOutcome.NO_SEAT);
  }

  /** 荒牌での聴牌の判定。手の内に 4 枚持つ牌だけを待つ形は不聴になり、副露で 3 枚使った牌の単騎は聴牌になる。 */
  @Test
  public void exhaustiveTenpaiIgnoresAWaitHeldFourTimesInHand() {
    Assert.assertFalse(WinLegality.tenpai(Scene.first().hand(0, "1111m234p567p789s").hands[0]));
    Assert.assertTrue(WinLegality.tenpai(Scene.first().pon(0, 1, "1m").hand(0, "1m234p567p789s").hands[0]));
    Assert.assertTrue(WinLegality.tenpai(Scene.first().hand(0, "123m456p789s1122z").hands[0]));
    Assert.assertFalse(WinLegality.tenpai(Scene.first().hand(0, "13579m13579p135s").hands[0]));
  }

  /** 流し満貫は、河が空でなく、全てが么九牌で、どれも鳴かれていないときに成り立つ。自分が鳴いていてもよい。 */
  @Test
  public void nagashiNeedsAnUncalledRiverOfTerminalsAndHonors() {
    Scene scene = Scene.first().river(0, "19m1z9p").river(2, "1s5z2m");
    Assert.assertTrue(WinLegality.nagashi(scene.state.events(), 0));
    Assert.assertFalse(WinLegality.nagashi(scene.state.events(), 2));
    Assert.assertFalse(WinLegality.nagashi(scene.state.events(), 1));
    Scene called = Scene.first().river(0, "19m").pon(1, 0, "1z").river(1, "9s9p");
    Assert.assertFalse(WinLegality.nagashi(called.state.events(), 0));
    Assert.assertTrue(WinLegality.nagashi(called.state.events(), 1));
  }

  /** 今その牌が打たれたとしたときのロンの点は、待ち・振聴・役を他家の応答と同じに判定する。山の残りが 0 枚なら河底が付く。 */
  @Test
  public void ronScoreJudgesLikeTheWindow() {
    Scene scene = Scene.first().hand(3, WAITS_1S_4S);
    WinScore score = new WinScore();
    Assert.assertTrue(scene.full.ronScore(3, Tiles.physicalId(Tiles.S4, 1), 1, scene.ws, score));
    Assert.assertTrue(score.has(ScoringYaku.ITTSU));
    Assert.assertFalse(score.has(ScoringYaku.HOUTEI));
    Assert.assertFalse(scene.full.ronScore(3, Tiles.physicalId(Tiles.S7, 1), 1, scene.ws, score));
    scene.hands[3].setMissedRon(true);
    Assert.assertTrue(scene.full.furiten(3));
    Assert.assertFalse(scene.full.ronScore(3, Tiles.physicalId(Tiles.S4, 1), 1, scene.ws, score));

    Scene river = Scene.first().hand(3, WAITS_1S_4S).river(3, "1s");
    Assert.assertFalse(river.full.ronScore(3, Tiles.physicalId(Tiles.S4, 1), 1, river.ws, score));

    Scene last = Scene.first().burn(Scene.VARIANT.liveWall()).hand(3, WAITS_1S_4S);
    Assert.assertTrue(last.full.ronScore(3, Tiles.physicalId(Tiles.S1, 1), 2, last.ws, score));
    Assert.assertTrue(score.has(ScoringYaku.HOUTEI));
    Assert.assertEquals(last.full.waits(3), Tiles.bit(Tiles.S1) | Tiles.bit(Tiles.S4));

    Assert.assertThrows(
        IllegalArgumentException.class, () -> last.full.ronScore(3, Tiles.physicalId(Tiles.S1, 1), 3, last.ws, score));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> last.full.ronScore(3, Tiles.physicalId(Tiles.S1, 1), 2, new RuleWorkspace(Variant.TENHOU_3P), score));
  }

  private static WinScore tsumo(Scene scene) {
    WinScore score = new WinScore();
    Hand hand = scene.hands[scene.state.last().seat()];
    Assert.assertTrue(WinLegality.tsumo(scene.state, hand, scene.wall, scene.ws, score), "no win for " + hand.shape());
    return score;
  }
}
