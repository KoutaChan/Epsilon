package com.epsilon.mahjong.score;

import static com.epsilon.mahjong.score.WinCase.four;
import static com.epsilon.mahjong.score.WinCase.three;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.action.Meld.AkaSource;
import com.epsilon.mahjong.action.Meld.RelativeSource;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinContext.Source;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeFixtures;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** ランダムな和了の形で採点器の性質を確かめる。食い違った入力を拒むことも確かめる。 */
public class HandScorerTest {

  private static final int SAMPLES = 2_000;

  private static final String KANCHAN = "123m456p789p55s13s";

  @DataProvider
  public Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  /**
   * 門前で和了の形になった手の立直ツモは必ず和了でき、結果は式と合う。基本点は {@code ScoreMath.base} の値、翻は役の翻の合計（ドラは
   * 無い）、符は 20（平和ツモ）か 25（七対子）か、30 以上の 10 の倍数。13 枚と和了牌に分けて渡しても 14 枚で渡しても結果は同じで、
   * 同じ採点器で何度採点しても変わらない。
   */
  @Test(dataProvider = "variants")
  public void closedRiichiTsumoAlwaysWinsAndFollowsTheFormulas(Variant variant) {
    Random random = new Random(71 + variant.ordinal());
    HandScorer scorer = new HandScorer();
    WinScore first = new WinScore();
    WinScore second = new WinScore();
    for (int sample = 0; sample < SAMPLES; sample++) {
      int[] counts = randomWin(random, variant.tiles(), sample);
      int winning = ShapeFixtures.randomHeld(random, counts);
      WinContext context = new WinContext(variant);
      counts[winning]--;
      context.hand().copyFrom(ShapeFixtures.shape(counts, 0));
      context.setWin(winning, Source.TSUMO);
      context.setRiichi(RiichiState.RIICHI, false);
      context.setWinds(Tiles.TON + random.nextInt(variant.seats()), Tiles.TON + random.nextInt(2));
      Assert.assertTrue(scorer.score(context, first), context.toString());
      assertFormulas(first, context);
      Assert.assertTrue(first.yakuman() > 0 || first.has(ScoringYaku.RIICHI), context + " " + first);
      context.hand().add(winning);
      Assert.assertTrue(scorer.score(context, second), context.toString());
      Assert.assertEquals(second.toString(), first.toString(), context.toString());
      Assert.assertTrue(scorer.score(context, second), context.toString());
      Assert.assertEquals(second.toString(), first.toString(), context.toString());
    }
  }

  /** ロンでも、和了できるなら同じ式に合う。役が無ければ結果は空になる。 */
  @Test(dataProvider = "variants")
  public void ronsFollowTheFormulasOrScoreNothing(Variant variant) {
    Random random = new Random(83 + variant.ordinal());
    HandScorer scorer = new HandScorer();
    WinScore score = new WinScore();
    for (int sample = 0; sample < SAMPLES; sample++) {
      int[] counts = randomWin(random, variant.tiles(), sample);
      int winning = ShapeFixtures.randomHeld(random, counts);
      WinContext context = new WinContext(variant);
      counts[winning]--;
      context.hand().copyFrom(ShapeFixtures.shape(counts, 0));
      context.setWin(winning, Source.RON);
      if (scorer.score(context, score)) {
        assertFormulas(score, context);
      } else {
        Assert.assertEquals(score.base() + score.han() + score.fu() + score.yakuman(), 0, context.toString());
        Assert.assertEquals(score.yaku(), 0L, context.toString());
      }
    }
  }

  @Test
  public void aHandWithoutYakuClearsTheLastScore() {
    HandScorer scorer = new HandScorer();
    WinScore score = new WinScore();
    Assert.assertTrue(scorer.score(four(KANCHAN).riichi().ron("2s").dora("4s").context(), score));
    Assert.assertFalse(scorer.score(four(KANCHAN).ron("2s").dora("4s").context(), score));
    Assert.assertEquals(score.toString(), new WinScore().toString());
    Assert.assertEquals(score.dora(), 0);
  }

  @Test
  public void clearForgetsEveryInput() {
    WinContext context =
        three("123p456p789s44z23s").riichi().ippatsu().ron("4s").nuki(1).dora("3z").ura("3z").red("5p").context();
    context.clear();
    Assert.assertEquals(context.toString(), new WinContext(Variant.TENHOU_3P).toString());
    Assert.assertEquals(context.hand().size(), 0);
    Assert.assertEquals(context.meldCount() + context.indicatorCount() + context.uraIndicatorCount(), 0);
    Assert.assertThrows(IllegalArgumentException.class, () -> new HandScorer().score(context, new WinScore()));
  }

  @Test
  public void inconsistentContextsAreRejected() {
    assertRejected(four("567m345p66p78s").chi("234m").riichi().ron("6s"), "riichi on an open hand");
    assertRejected(four("123m456p789p55s1s").ron("2s"), "12 tiles");
    assertRejected(four("123m456p789p55s123s").ron("4s"), "14 tiles without the winning tile");
    assertRejected(four("1111m456p789p5s13s").ron("1m"), "a fifth tile");
    assertRejected(four("11m789p55s13s9p").pon("1m").ron("2s"), "a fifth tile across a meld");
    assertRejected(four(KANCHAN).red("5m").ron("2s"), "a red five outside the hand");
    assertRejected(four(KANCHAN).firstDraw().ron("2s"), "a first draw on a ron");
    assertRejected(
        four("9p").ankan("1m").daiminkan("2p").ankan("3s").kakan("4s").dealer().firstDraw().tsumo("9p"),
        "a first draw after melds");
    assertRejected(four(KANCHAN).riichi().firstDraw().tsumo("2s"), "a first draw after a riichi");
    assertRejected(three("123p456p789s11m23s").nuki(1).firstDraw().tsumo("1s"), "a first draw after a nuki");
    assertRejected(
        four("123m456m55p78s").ankan("9p").riichi().ippatsu().win("9s", Source.RINSHAN), "ippatsu on a rinshan draw");
    assertRejected(four(KANCHAN).win("2s", Source.RINSHAN), "a rinshan draw without a kan or a nuki");
    assertRejected(four(KANCHAN).riichi().ron("2s").dora("1z1z").ura("4s"), "fewer ura than dora indicators");
    assertRejected(three(KANCHAN).ron("2s"), "two to eight man in three-player");
    WinCase uncounted = four(KANCHAN).ron("2s");
    uncounted.context().hand().remove(Tiles.M1);
    uncounted.context().hand().remove(Tiles.M2);
    uncounted.context().hand().remove(Tiles.M3);
    uncounted.context().addMeld(Meld.chi(Tiles.M1, Tiles.M1, AkaSource.NONE));
    assertRejected(uncounted, "a meld the hand does not count");
    assertRejected(four(KANCHAN), "no winning tile");
  }

  @Test
  public void settersCheckTheirRange() {
    WinContext three = new WinContext(Variant.TENHOU_3P);
    Assert.assertThrows(IllegalArgumentException.class, () -> three.setWin(Tiles.M2, Source.RON));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.addIndicator(Tiles.M5));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.setWinds(Tiles.PEI, Tiles.TON));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.setWinds(Tiles.TON, Tiles.PEI));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.setRedFives(Tiles.bit(Tiles.M5)));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> three.addMeld(Meld.chi(Tiles.P1, Tiles.P1, AkaSource.NONE)));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> three.addMeld(Meld.pon(Tiles.M5, RelativeSource.KAMICHA, AkaSource.NONE)));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.addMeld(Meld.ankan(Tiles.M2, false)));
    new WinContext(Variant.TENHOU_3P).addMeld(Meld.pon(Tiles.PEI, RelativeSource.KAMICHA, AkaSource.NONE));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.setRiichi(RiichiState.NONE, true));
    WinContext four = new WinContext(Variant.TENHOU_4P);
    four.setWinds(Tiles.PEI, Tiles.SHA);
    four.setRedFives(Tiles.FIVES);
    for (int i = 0; i < WinContext.MAX_INDICATORS; i++) {
      four.addIndicator(Tiles.TON);
    }
    Assert.assertThrows(IllegalStateException.class, () -> four.addIndicator(Tiles.TON));
    for (int i = 0; i < Variant.MAX_MELDS; i++) {
      four.addMeld(Meld.pon(Tiles.TON + i, RelativeSource.KAMICHA, AkaSource.NONE));
    }
    Meld fifth = Meld.pon(Tiles.HAKU, RelativeSource.KAMICHA, AkaSource.NONE);
    Assert.assertThrows(IllegalStateException.class, () -> four.addMeld(fifth));
  }

  private static void assertRejected(WinCase hand, String reason) {
    boolean rejected = false;
    try {
      hand.scores(new WinScore());
    } catch (IllegalArgumentException expected) {
      rejected = true;
    }
    Assert.assertTrue(rejected, "accepted " + reason + ": " + hand);
  }

  private static void assertFormulas(WinScore score, WinContext context) {
    String message = context + " " + score;
    if (score.yakuman() > 0) {
      Assert.assertEquals(Long.bitCount(score.yaku()), score.yakuman(), message);
      Assert.assertEquals(score.base(), ScoreMath.base(0, 0, score.yakuman()), message);
      Assert.assertEquals(score.han() + score.fu(), 0, message);
      return;
    }
    int han = 0;
    for (ScoringYaku yaku : ScoringYaku.values()) {
      if (score.has(yaku)) {
        Assert.assertFalse(yaku.yakuman(), message);
        Assert.assertTrue(yaku.han(score.closed()) > 0, message);
        han += yaku.han(score.closed());
      }
    }
    Assert.assertEquals(score.han(), han + score.dora() + score.aka() + score.ura() + score.nuki(), message);
    Assert.assertEquals(score.base(), ScoreMath.base(score.han(), score.fu(), 0), message);
    int fu = score.fu();
    if (fu == 20) {
      Assert.assertTrue(score.has(ScoringYaku.PINFU) && context.source().tsumo(), message);
    } else if (fu == 25) {
      Assert.assertTrue(score.has(ScoringYaku.CHIITOITSU), message);
    } else {
      Assert.assertTrue(fu >= 30 && fu % 10 == 0, message);
    }
  }

  /** ランダムな和了の形の 14 枚（種類ごとの枚数）。たいていは面子 4 つと雀頭で、10 回に 1 回は七対子、50 回に 1 回は国士。 */
  private static int[] randomWin(Random random, TileSet set, int sample) {
    if (sample % 50 == 1) {
      int[] counts = new int[Tiles.KINDS];
      for (long rest = Tiles.TERMINALS_AND_HONORS; rest != 0; rest &= rest - 1) {
        counts[Long.numberOfTrailingZeros(rest)] = 1;
      }
      counts[ShapeFixtures.randomHeld(random, counts)]++;
      return counts;
    }
    if (sample % 10 == 0) {
      int[] counts = new int[Tiles.KINDS];
      for (int pairs = 0; pairs < HandShape.SEVEN_PAIRS; ) {
        int kind = ShapeFixtures.randomKind(random, set);
        if (counts[kind] == 0) {
          counts[kind] = 2;
          pairs++;
        }
      }
      return counts;
    }
    return ShapeFixtures.randomComplete(random, set, Variant.MAX_MELDS);
  }
}
