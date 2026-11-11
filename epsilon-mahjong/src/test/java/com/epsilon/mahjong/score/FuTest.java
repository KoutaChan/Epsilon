package com.epsilon.mahjong.score;

import static com.epsilon.mahjong.score.WinCase.four;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 符の数え方で結果が分かれる手を、手で求めた値で確かめる（四人打ち、南家・東場）。 */
public class FuTest {

  private static final String PINFU = "123m456p789p55s23s";

  /** 平和のツモは、ツモの 2 符を付けずに 20 符（2 翻 20 符で基本点 320）。 */
  @Test
  public void pinfuTsumoIsTwentyFu() {
    WinScore score = four(PINFU).tsumo("1s").score();
    assertScore(score, 2, 20, 320);
    Assert.assertEquals(score.yaku(), YakuTest.bits(ScoringYaku.PINFU, ScoringYaku.MENZEN_TSUMO));
  }

  @Test
  public void pinfuRonIsThirtyFu() {
    assertScore(four(PINFU).ron("1s").score(), 1, 30, 240);
  }

  /** 七対子はツモでも 25 符。 */
  @Test
  public void chiitoitsuIsTwentyFiveFuEitherWay() {
    assertScore(four("1199m2255p3377s4z").ron("4z").score(), 2, 25, 400);
    assertScore(four("1199m2255p3377s4z").tsumo("4z").score(), 3, 25, 800);
  }

  /** 鳴いた平和の形は、ロンなら 20 符、ツモなら 22 符になるところを、どちらも 30 符にする。 */
  @Test
  public void anOpenPinfuShapeIsThirtyFu() {
    assertScore(four("567m345p66p78s").chi("234m").ron("6s").score(), 1, 30, 240);
    WinScore tsumo = four("567m345p66p78s").chi("234m").tsumo("6s").score();
    assertScore(tsumo, 1, 30, 240);
    Assert.assertFalse(tsumo.closed());
    Assert.assertEquals(tsumo.yaku(), ScoringYaku.TANYAO.bit());
  }

  /** 暗槓は么九牌なら 32 符、中張牌なら 16 符。明槓は么九牌なら 16 符、中張牌なら 8 符。暗槓は門前を崩さない。 */
  @Test
  public void kansCarryTheirFu() {
    WinScore terminal = four("234m567m55p78s").ankan("9p").riichi().ron("9s").score();
    assertScore(terminal, 1, 70, 560);
    Assert.assertTrue(terminal.closed());
    assertScore(four("234m567m55p78s").ankan("3p").riichi().ron("9s").score(), 1, 50, 400);
    assertScore(four("123m456p22s78s").ankan("5z").ron("9s").score(), 1, 70, 560);
    assertScore(four("345m678s9m").daiminkan("2p").pon("5z").ron("9m").score(), 1, 40, 320);
    assertScore(four("345m678s99m12p").kakan("1z").ron("3p").score(), 1, 40, 320);
  }

  /** 東場の親が東の雀頭を持てば 4 符（20 ＋ 10 ＋ 么九の暗刻 8 ＋ 4 = 42 → 50 符）。子なら場風の 2 符だけで 40 符。 */
  @Test
  public void aDoubleWindPairIsFourFu() {
    assertScore(four("111m456p789p11z23s").dealer().riichi().ron("1s").score(), 1, 50, 400);
    assertScore(four("111m456p789p11z23s").riichi().ron("1s").score(), 1, 40, 320);
  }

  /** 双碰待ちをロンで埋めた刻子は明刻になる（20 ＋ 10 ＋ 4 ＋ 4 ＋ 2 = 40 符）。ツモなら暗刻のままで三暗刻が付く。 */
  @Test
  public void aShanponRonOpensTheTriplet() {
    WinScore ron = four("222m444p456m77s88s").ron("8s").score();
    assertScore(ron, 1, 40, 320);
    Assert.assertEquals(ron.yaku(), ScoringYaku.TANYAO.bit());
    WinScore tsumo = four("222m444p456m77s88s").tsumo("8s").score();
    assertScore(tsumo, 4, 40, 2_000);
    Assert.assertEquals(
        tsumo.yaku(), YakuTest.bits(ScoringYaku.TANYAO, ScoringYaku.MENZEN_TSUMO, ScoringYaku.SANANKOU));
  }

  /** 4p は 345 の嵌張（32 → 40 符）とも 456 の両面（30 符）とも読める。翻が同じなので、符の高い嵌張の読みを採る。 */
  @Test
  public void theReadingWithMoreFuWins() {
    assertScore(four("111m34556p789s22s").riichi().tsumo("4p").score(), 2, 40, 640);
  }

  /** 3m は 123 の辺張とも 345 の両面とも読める。両面の読みなら平和が付いて基本点が高い（480 対 320）ので、こちらを採る。 */
  @Test
  public void pinfuBeatsTheEdgeWaitReading() {
    WinScore score = four("12345m678p234s99s").riichi().ron("3m").score();
    assertScore(score, 2, 30, 480);
    Assert.assertEquals(score.yaku(), YakuTest.bits(ScoringYaku.RIICHI, ScoringYaku.PINFU));
  }

  /** 平和の形でも雀頭が役牌なら平和は付かない。客風の雀頭は 0 符で平和になる。 */
  @Test
  public void aValuePairBlocksPinfu() {
    WinScore round = four("123m456p789p11z23s").riichi().ron("1s").score();
    assertScore(round, 1, 40, 320);
    Assert.assertFalse(round.has(ScoringYaku.PINFU));
    WinScore guest = four("123m456p789p33z23s").riichi().ron("1s").score();
    assertScore(guest, 2, 30, 480);
    Assert.assertTrue(guest.has(ScoringYaku.PINFU));
  }

  private static void assertScore(WinScore score, int han, int fu, int base) {
    Assert.assertEquals(score.han(), han, score.toString());
    Assert.assertEquals(score.fu(), fu, score.toString());
    Assert.assertEquals(score.base(), base, score.toString());
  }
}
