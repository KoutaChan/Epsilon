package com.epsilon.mahjong.score;

import static com.epsilon.mahjong.score.WinCase.four;
import static com.epsilon.mahjong.score.WinCase.three;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.action.Meld.AkaSource;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinContext.Source;
import com.epsilon.mahjong.tile.Tiles;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 表ドラ・赤・裏ドラと、三人打ちの抜きドラを確かめる。どれも翻に足すが、1 翻縛りの役には数えず、役満では数えない。 */
public class DoraTest {

  private static final String KANCHAN = "123m456p789p55s13s";

  /** 三人打ちで平和になる手（雀頭は 1m、2s・3s の両面待ち）。 */
  private static final String SANMA_PINFU = "123p456p789s11m23s";

  @Test
  public void doraCountsEveryTileOfTheHandAndTheMelds() {
    assertDora(four(KANCHAN).riichi().ron("2s").dora("4s").score(), 3, 2, 0, 0, 0);
    assertDora(four("123m456p22s78s").pon("5z").ron("9s").dora("7z").score(), 4, 3, 0, 0, 0);
    assertDora(four("123m456m55p78s").ankan("9p").win("9s", Source.RINSHAN).dora("8p").score(), 6, 4, 0, 0, 0);
    assertDora(four("567m345p66p78s").chi("234m").ron("6s").dora("2m").score(), 2, 1, 0, 0, 0);
  }

  @Test
  public void everyIndicatorCountsOnItsOwn() {
    assertDora(four(KANCHAN).riichi().ron("2s").dora("4s4s").score(), 5, 4, 0, 0, 0);
    assertDora(four(KANCHAN).riichi().ron("2s").dora("4s3z").score(), 3, 2, 0, 0, 0);
  }

  /** 赤は、純手牌にある赤と、面子に含まれる赤を数える。 */
  @Test
  public void redFivesCountInTheHandAndTheMelds() {
    Meld redChi = Meld.chi(Tiles.M3, Tiles.M3, AkaSource.CONSUMED_HAND_TILE);
    assertDora(four("678m345p66p78s").meld(redChi).red("5p").ron("6s").score(), 3, 0, 2, 0, 0);
    assertDora(four("678m345p66p78s").meld(redChi).ron("6s").score(), 2, 0, 1, 0, 0);
  }

  @Test
  public void uraCountsOnlyForARiichi() {
    assertDora(four(KANCHAN).riichi().ron("2s").dora("1z").ura("4s").score(), 3, 0, 0, 2, 0);
    assertDora(four(KANCHAN).win("2s", Source.HOUTEI).dora("1z").ura("4s").score(), 1, 0, 0, 0, 0);
  }

  @Test
  public void yakumanCountNoDora() {
    WinScore score = four("19m19p19s1234567z").ron("1m").dora("9m").score();
    Assert.assertEquals(score.yakuman(), 1);
    Assert.assertEquals(score.base(), 8_000);
    assertDora(score, 0, 0, 0, 0, 0);
  }

  @Test
  public void doraIsNotAYaku() {
    Assert.assertFalse(four(KANCHAN).ron("2s").dora("4s").red("5p").scores(new WinScore()));
    Assert.assertFalse(three("123p456p789s11m13s").ron("2s").nuki(3).dora("3z").scores(new WinScore()));
  }

  /** 三人打ちの萬子は 1m と 9m だけなので、表示牌が 9m ならドラは 1m、1m ならドラは 9m。四人打ちでは表示牌 1m のドラは 2m。 */
  @Test
  public void sanmaIndicatorsWrapOverTheMissingMan() {
    assertDora(three(SANMA_PINFU).ron("4s").dora("9m").score(), 3, 2, 0, 0, 0);
    assertDora(three("123p456p789s99m23s").ron("4s").dora("1m").score(), 3, 2, 0, 0, 0);
    assertDora(four("123p456p789s22m23s").ron("4s").dora("1m").score(), 3, 2, 0, 0, 0);
  }

  /** 抜いた北は 1 枚につき 1 翻の抜きドラになり、北をドラにする表示牌（西）1 枚ごとに表ドラとしてさらに 1 翻増える。 */
  @Test
  public void nukiCountsEachNorthOnceAndAgainPerWestIndicator() {
    WinScore plain = three(SANMA_PINFU).ron("4s").nuki(2).dora("5s").score();
    assertDora(plain, 3, 0, 0, 0, 2);
    Assert.assertEquals(plain.base(), 960);
    WinScore west = three(SANMA_PINFU).ron("4s").nuki(2).dora("3z").score();
    assertDora(west, 5, 2, 0, 0, 2);
    Assert.assertEquals(west.base(), 2_000);
    assertDora(three(SANMA_PINFU).ron("4s").nuki(1).dora("3z3z").score(), 4, 2, 0, 0, 1);
  }

  /**
   * 北抜きの補充で和了すれば、槓が無くても嶺上開花が付く。符は 20 ＋ ツモ 2 ＋ 1p の暗刻 8。抜いた北は、抜きドラで 1 翻、西の表示牌に
   * よる表ドラで 1 翻になる。
   */
  @Test
  public void aWinOnTheNukiReplacementIsRinshan() {
    WinScore score = three("111p456p789s11m23s").nuki(1).win("4s", Source.RINSHAN).dora("3z").score();
    assertDora(score, 4, 1, 0, 0, 1);
    Assert.assertEquals(score.yaku(), YakuTest.bits(ScoringYaku.MENZEN_TSUMO, ScoringYaku.RINSHAN));
    Assert.assertEquals(score.fu(), 30);
    Assert.assertEquals(score.base(), 1_920);
  }

  /** 裏ドラ表示牌が西でも、抜いた北 1 枚ごとに裏ドラとして 1 翻を足す。ただし立直の和了のときだけ。 */
  @Test
  public void aWestUraIndicatorWeightsTheNorthOnlyForARiichi() {
    WinScore riichi = three(SANMA_PINFU).riichi().ron("4s").nuki(1).dora("3z").ura("3z").score();
    assertDora(riichi, 5, 1, 0, 1, 1);
    Assert.assertEquals(riichi.yaku(), YakuTest.bits(ScoringYaku.RIICHI, ScoringYaku.PINFU));
    assertDora(three(SANMA_PINFU).ron("4s").nuki(1).dora("3z").ura("3z").score(), 3, 1, 0, 0, 1);
  }

  /** 三人打ちの北は客風なので、北の雀頭でも平和が付く。手の北も抜いた北も表ドラに数え、抜いた北はさらに抜きドラにも数える。 */
  @Test
  public void northTilesInTheHandAreOrdinaryDora() {
    WinScore score = three("123p456p789s44z23s").ron("4s").nuki(1).dora("3z").score();
    assertDora(score, 5, 3, 0, 0, 1);
    Assert.assertTrue(score.has(ScoringYaku.PINFU), score.toString());
  }

  /**
   * 北の暗槓は北抜きではないので、抜きドラは付かない。表示牌が西なら、槓子の 4 枚はほかの牌と同じく表ドラになって 4 翻。立直で裏ドラ
   * 表示牌が西でも 4 翻。暗槓の代わりに 9s の暗刻を持って北を 4 枚抜いた手は、表ドラが同じ 4 翻で、抜きドラが 4 翻増える。
   */
  @Test
  public void northsInAnAnkanAreOrdinaryDoraWithoutNuki() {
    String concealed = "123p456p11m23s";
    assertDora(three(concealed).ankan("4z").riichi().ron("4s").dora("1z").score(), 1, 0, 0, 0, 0);
    assertDora(three(concealed).ankan("4z").riichi().ron("4s").dora("3z").score(), 5, 4, 0, 0, 0);
    assertDora(three(concealed).ankan("4z").riichi().ron("4s").dora("1z").ura("3z").score(), 5, 0, 0, 4, 0);
    assertDora(three("123p456p999s11m23s").riichi().ron("4s").nuki(4).dora("3z").score(), 9, 4, 0, 0, 4);
  }

  /**
   * 北のポンも北抜きではないので、抜きドラは付かない（役は白のポン）。表示牌が西なら、ポンの 3 枚が表ドラで 3 翻。4 枚目の北を抜けば、
   * 抜きドラ 1 翻と表ドラ 1 翻が増える。
   */
  @Test
  public void northsInAPonAreOrdinaryDoraWithoutNuki() {
    String concealed = "123p11m23s";
    assertDora(three(concealed).pon("4z").pon("5z").ron("4s").dora("1z").score(), 1, 0, 0, 0, 0);
    assertDora(three(concealed).pon("4z").pon("5z").ron("4s").dora("3z").score(), 4, 3, 0, 0, 0);
    assertDora(three(concealed).pon("4z").pon("5z").ron("4s").nuki(1).dora("3z").score(), 6, 4, 0, 0, 1);
  }

  @Test
  public void nukiNeedsTheNukiRuleAndAFreeNorth() {
    Assert.assertThrows(IllegalArgumentException.class, () -> new WinContext(Variant.TENHOU_4P).setNuki(1));
    Assert.assertThrows(IllegalArgumentException.class, () -> new WinContext(Variant.TENHOU_3P).setNuki(5));
    WinCase fiveNorths = three("123p456p789s44z23s").ron("4s").nuki(3);
    Assert.assertThrows(IllegalArgumentException.class, () -> fiveNorths.scores(new WinScore()));
  }

  private static void assertDora(WinScore score, int han, int dora, int aka, int ura, int nuki) {
    String message = score.toString();
    Assert.assertEquals(score.han(), han, message);
    Assert.assertEquals(score.dora(), dora, message);
    Assert.assertEquals(score.aka(), aka, message);
    Assert.assertEquals(score.ura(), ura, message);
    Assert.assertEquals(score.nuki(), nuki, message);
  }
}
