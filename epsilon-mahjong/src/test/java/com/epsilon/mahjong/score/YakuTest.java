package com.epsilon.mahjong.score;

import static com.epsilon.mahjong.score.WinCase.four;

import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.score.WinContext.Source;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 役ごとに、手で求めた翻・符と役満の倍率で確かめる（四人打ち、南家・東場、ドラなし）。42 の役のどれにも確かめる手がある。 */
public class YakuTest {

  private static final String KANCHAN = "123m456p789p55s13s";

  @DataProvider
  public Object[][] hands() {
    return new Object[][] {
      row(four(KANCHAN).riichi().ron("2s"), 1, 40, ScoringYaku.RIICHI),
      row(four(KANCHAN).doubleRiichi().ron("2s"), 2, 40, ScoringYaku.DOUBLE_RIICHI),
      row(four(KANCHAN).riichi().ippatsu().ron("2s"), 2, 40, ScoringYaku.RIICHI, ScoringYaku.IPPATSU),
      row(four(KANCHAN).tsumo("2s"), 1, 30, ScoringYaku.MENZEN_TSUMO),
      row(four("123m456p789p55s23s").ron("1s"), 1, 30, ScoringYaku.PINFU),
      row(four("567m345p66p78s").chi("234m").ron("6s"), 1, 30, ScoringYaku.TANYAO),
      row(four("123m456p22s78s").pon("5z").ron("9s"), 1, 30, ScoringYaku.YAKUHAI_HAKU),
      row(four("123m456p22s78s").pon("6z").ron("9s"), 1, 30, ScoringYaku.YAKUHAI_HATSU),
      row(four("123m456p22s78s").pon("7z").ron("9s"), 1, 30, ScoringYaku.YAKUHAI_CHUN),
      row(four("123m456p22s78s").pon("2z").ron("9s"), 1, 30, ScoringYaku.YAKUHAI_SEAT),
      row(four("123m456p22s78s").pon("1z").ron("9s"), 1, 30, ScoringYaku.YAKUHAI_ROUND),
      row(
          four("123m456p22s78s").dealer().pon("1z").ron("9s"),
          2,
          30,
          ScoringYaku.YAKUHAI_SEAT,
          ScoringYaku.YAKUHAI_ROUND),
      row(
          four("123m456m55p78s").ankan("9p").win("9s", Source.RINSHAN),
          2,
          60,
          ScoringYaku.MENZEN_TSUMO,
          ScoringYaku.RINSHAN),
      row(four(KANCHAN).win("2s", Source.CHANKAN), 1, 40, ScoringYaku.CHANKAN),
      row(four(KANCHAN).win("2s", Source.HAITEI), 2, 30, ScoringYaku.MENZEN_TSUMO, ScoringYaku.HAITEI),
      row(four(KANCHAN).win("2s", Source.HOUTEI), 1, 40, ScoringYaku.HOUTEI),
      row(four("112233m456p79s55s").ron("8s"), 1, 40, ScoringYaku.IPEIKOU),
      row(four("123m123p13s456s99m").ron("2s"), 2, 40, ScoringYaku.SANSHOKU),
      row(four("123p13s456s99m").chi("123m").ron("2s"), 1, 30, ScoringYaku.SANSHOKU),
      row(four("222s456m9p").pon("2m").pon("2p").ron("9p"), 2, 30, ScoringYaku.SANSHOKU_DOUKOU),
      row(four("123m456m789m55p13s").ron("2s"), 2, 40, ScoringYaku.ITTSU),
      row(four("456m789m55p13s").chi("123m").ron("2s"), 1, 30, ScoringYaku.ITTSU),
      row(four("444s99m77s").pon("2m").pon("8p").ron("7s"), 2, 30, ScoringYaku.TOITOI),
      row(four("111m333p555s789m9p").ron("9p"), 2, 50, ScoringYaku.SANANKOU),
      // 暗槓も暗刻に数える（暗刻 2 ＋ 暗槓 1）。符は 20 ＋ 10 ＋ 8 ＋ 4 ＋ 暗槓 16 ＋ 単騎 2。
      row(four("111m333p789s9p").ankan("4s").riichi().ron("9p"), 3, 60, ScoringYaku.RIICHI, ScoringYaku.SANANKOU),
      // 刻子 3 つの読み（2 翻 50 符）が、同じ順子 3 つの読み（一盃口で 1 翻 40 符）より高いので採られる。
      row(four("111222333m456p7s").ron("7s"), 2, 50, ScoringYaku.SANANKOU),
      row(four("567m9p").daiminkan("2m").daiminkan("3p").ankan("4s").ron("9p"), 2, 60, ScoringYaku.SANKANTSU),
      row(
          four("123m77z45p").pon("5z").pon("6z").ron("3p"),
          4,
          30,
          ScoringYaku.YAKUHAI_HAKU,
          ScoringYaku.YAKUHAI_HATSU,
          ScoringYaku.SHOUSANGEN),
      row(four("111s99s11z").pon("1m").pon("9p").ron("9s"), 4, 50, ScoringYaku.TOITOI, ScoringYaku.HONROUTOU),
      row(four("1199m1199p11s11z2z").ron("2z"), 4, 25, ScoringYaku.HONROUTOU, ScoringYaku.CHIITOITSU),
      row(four("123m789p999s11z12s").ron("3s"), 2, 50, ScoringYaku.CHANTA),
      row(four("789p999s11z12s").chi("123m").ron("3s"), 1, 40, ScoringYaku.CHANTA),
      row(four("223344m556677p8s").ron("8s"), 4, 40, ScoringYaku.TANYAO, ScoringYaku.RYANPEIKOU),
      row(four("123m789p999s11m12s").ron("3s"), 3, 40, ScoringYaku.JUNCHAN),
      row(four("789p999s11m12s").chi("123m").ron("3s"), 2, 30, ScoringYaku.JUNCHAN),
      row(four("123m345m666m11z79m").ron("8m"), 3, 40, ScoringYaku.HONITSU),
      row(four("345m11z79m").chi("123m").pon("6m").ron("8m"), 2, 30, ScoringYaku.HONITSU),
      row(four("111m234m567m99m68m").ron("7m"), 6, 40, ScoringYaku.CHINITSU),
      row(four("234m567m99m68m").pon("1m").ron("7m"), 5, 30, ScoringYaku.CHINITSU),
      row(four("1199m2255p3377s4z").ron("4z"), 2, 25, ScoringYaku.CHIITOITSU),
    };
  }

  @DataProvider
  public Object[][] yakuman() {
    return new Object[][] {
      row(four("19m19p19s1234567z").ron("1m"), ScoringYaku.KOKUSHI),
      row(four("111m333p555s999s7z").ron("7z"), ScoringYaku.SUUANKOU),
      row(four("111m333p555s77z99s").tsumo("9s"), ScoringYaku.SUUANKOU),
      // 暗刻 1 つと暗槓 3 つの単騎待ち。
      row(four("111m9p").ankan("3p").ankan("5s").ankan("7z").ron("9p"), ScoringYaku.SUUANKOU),
      // 先に読む順子 3 つの分け方（門前ツモ・一盃口）より、後で読む刻子の分け方の役満を採る。
      row(four("111222333m55p77s").tsumo("7s"), ScoringYaku.SUUANKOU),
      row(four("1112345678999m").ron("5m"), ScoringYaku.CHUUREN),
      row(four("666z777z123m9p").pon("5z").ron("9p"), ScoringYaku.DAISANGEN),
      row(four("333z555z6z").pon("1z").pon("2z").ron("6z"), ScoringYaku.TSUUIISOU),
      row(four("223344s666s88s66z").ron("8s"), ScoringYaku.RYUUIISOU),
      row(four("111p999s1s").pon("1m").pon("9m").ron("1s"), ScoringYaku.CHINROUTOU),
      row(four("333z444z9m").pon("1z").pon("2z").ron("9m"), ScoringYaku.DAISUUSHII),
      row(four("333z44z23m").pon("1z").pon("2z").ron("1m"), ScoringYaku.SHOUSUUSHII),
      row(four("9p").ankan("1m").daiminkan("2p").ankan("3s").kakan("4s").ron("9p"), ScoringYaku.SUUKANTSU),
      row(four("123m456p789s11z23s").dealer().firstDraw().tsumo("1s"), ScoringYaku.TENHOU),
      row(four("123m456p789s11z23s").firstDraw().tsumo("1s"), ScoringYaku.CHIIHOU),
      row(four("19m19p19s1234567z").dealer().firstDraw().tsumo("7z"), ScoringYaku.KOKUSHI, ScoringYaku.TENHOU),
      row(four("555z666z777z2z").pon("1z").ron("2z"), ScoringYaku.DAISANGEN, ScoringYaku.TSUUIISOU),
      row(
          four("111z222z333z444z5z").ron("5z"),
          ScoringYaku.SUUANKOU,
          ScoringYaku.TSUUIISOU,
          ScoringYaku.DAISUUSHII),
    };
  }

  @Test(dataProvider = "hands")
  public void handsScoreTheirYakuHanAndFu(WinCase hand, int han, int fu, ScoringYaku[] yaku) {
    WinScore score = hand.score();
    String message = hand + " " + score;
    Assert.assertEquals(score.yaku(), bits(yaku), message);
    Assert.assertEquals(score.han(), han, message);
    Assert.assertEquals(score.fu(), fu, message);
    Assert.assertEquals(score.yakuman(), 0, message);
    Assert.assertEquals(score.base(), ScoreMath.base(han, fu, 0), message);
    Assert.assertEquals(score.dora() + score.aka() + score.ura() + score.nuki(), 0, message);
  }

  /** 天鳳では、複合した役満を役ごとに 1 倍として足す。役満はドラ・翻・符を持たない。 */
  @Test(dataProvider = "yakuman")
  public void yakumanCountOncePerYaku(WinCase hand, ScoringYaku[] yaku) {
    WinScore score = hand.score();
    String message = hand + " " + score;
    Assert.assertEquals(score.yaku(), bits(yaku), message);
    Assert.assertEquals(score.yakuman(), yaku.length, message);
    Assert.assertEquals(score.base(), 8_000 * yaku.length, message);
    Assert.assertEquals(score.han(), 0, message);
    Assert.assertEquals(score.fu(), 0, message);
  }

  @Test
  public void everyYakuHasAScene() {
    long covered = 0;
    for (Object[] row : hands()) {
      covered |= bits((ScoringYaku[]) row[3]);
    }
    for (Object[] row : yakuman()) {
      covered |= bits((ScoringYaku[]) row[1]);
    }
    Assert.assertEquals(covered, (1L << ScoringYaku.values().length) - 1);
  }

  /** 四暗刻の形でも、双碰待ちをロンで埋めた刻子は明刻なので、三暗刻と対々和になる（4 翻 60 符）。 */
  @Test
  public void aShanponRonTurnsFourConcealedTripletsIntoThree() {
    WinScore score = four("111m333p555s77z99s").ron("9s").score();
    Assert.assertEquals(score.yaku(), bits(ScoringYaku.TOITOI, ScoringYaku.SANANKOU), score.toString());
    Assert.assertEquals(score.han(), 4);
    Assert.assertEquals(score.fu(), 60);
    Assert.assertEquals(score.base(), 2_000);
  }

  @Test
  public void handsWithoutYakuDoNotWin() {
    WinScore out = new WinScore();
    Assert.assertFalse(four(KANCHAN).ron("2s").scores(out));
    Assert.assertFalse(four("123m456p22s78s").pon("3z").ron("9s").scores(out), "a guest wind triplet");
    Assert.assertFalse(four("123m456p79s55s").chi("123m").ron("8s").scores(out), "an open ipeikou");
    Assert.assertFalse(four("456m789m55p13s").chi("123p").ron("2s").scores(out), "a broken straight");
  }

  /** 門前限定の役は 8 つ、喰い下がり（鳴くと 1 翻減る）の役は 6 つで、ほかの役は鳴いても翻が同じ。役満は 12 個で、翻を持たない。 */
  @Test
  public void openHandsLoseTheClosedOnlyYakuAndOneHanOfTheKuisagari() {
    long closedOnly =
        bits(
            ScoringYaku.RIICHI,
            ScoringYaku.DOUBLE_RIICHI,
            ScoringYaku.IPPATSU,
            ScoringYaku.MENZEN_TSUMO,
            ScoringYaku.PINFU,
            ScoringYaku.IPEIKOU,
            ScoringYaku.RYANPEIKOU,
            ScoringYaku.CHIITOITSU);
    long kuisagari =
        bits(
            ScoringYaku.SANSHOKU,
            ScoringYaku.ITTSU,
            ScoringYaku.CHANTA,
            ScoringYaku.JUNCHAN,
            ScoringYaku.HONITSU,
            ScoringYaku.CHINITSU);
    int yakuman = 0;
    for (ScoringYaku yaku : ScoringYaku.values()) {
      if (yaku.yakuman()) {
        yakuman++;
        Assert.assertEquals(yaku.han(true) + yaku.han(false), 0, yaku.name());
        continue;
      }
      int closed = yaku.han(true);
      Assert.assertTrue(closed > 0, yaku.name());
      int open = (closedOnly & yaku.bit()) != 0 ? 0 : (kuisagari & yaku.bit()) != 0 ? closed - 1 : closed;
      Assert.assertEquals(yaku.han(false), open, yaku.name());
    }
    Assert.assertEquals(yakuman, 12);
  }

  private static Object[] row(WinCase hand, int han, int fu, ScoringYaku... yaku) {
    return new Object[] {hand, han, fu, yaku};
  }

  private static Object[] row(WinCase hand, ScoringYaku... yaku) {
    return new Object[] {hand, yaku};
  }

  static long bits(ScoringYaku... yaku) {
    long bits = 0;
    for (ScoringYaku each : yaku) {
      bits |= each.bit();
    }
    return bits;
  }
}
