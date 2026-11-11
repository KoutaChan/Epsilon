package com.epsilon.agent.facts;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.facts.WaitScores.Method;
import com.epsilon.agent.facts.WaitScores.WinningTile;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Locale;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 待ちごとの点数を、乱数で作った聴牌の手で総当たり（{@link DirectScoring}）と照らし合わせる。手は四人打ちと三人打ちの
 * 両方で作り、副露・立直・抜いた北・赤も含む。その種類の牌を純手牌・面子・抜いた北で全て持っている待ちは、採点しない。
 */
public class WaitScoresTest {

  private static final Logger log = LoggerFactory.getLogger(WaitScoresTest.class);
  private static final int HANDS = 3_000;
  private static final RiichiState[] RIICHI = RiichiState.values();

  @DataProvider
  public Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test(dataProvider = "variants")
  public void scoresMatchDirectScoringOnRandomTenpaiHands(Variant variant) {
    Random random = new Random(71 + variant.ordinal());
    WaitScores scores = new WaitScores(variant);
    ShapeAnalyzer analyzer = new ShapeAnalyzer(variant);
    DirectScoring brute = new DirectScoring(variant);
    for (int sample = 0; sample < HANDS; sample++) {
      long red = randomTenpai(random, variant, scores.context());
      long waits = analyzer.waits(scores.context().hand());
      scores.load(waits, red);
      brute.check(scores, waits, red);
    }
    log.info(
        String.format(
            Locale.ROOT,
            "WAIT SCORES %s: hands %d scoredWins %d redWins %d",
            variant,
            HANDS,
            brute.scoredWins(),
            brute.redWins()));
    assertTrue(brute.scoredWins() > HANDS, "scored wins " + brute.scoredWins());
    assertTrue(brute.redWins() > 0, "no red winning tile was scored");
  }

  /** 5p をポンし、4 枚目の 5p（赤）を持って単騎で待つ手。形の上では待ちだが、和了牌が 5 枚目の 5p になるので採点しない。 */
  @Test
  public void aWaitWhoseCopiesAreAllHeldIsNotScored() {
    WaitScores scores = new WaitScores(Variant.TENHOU_4P);
    WinContext context = scores.context();
    context.clear();
    concealed(
        context.hand(),
        Tiles.P5, Tiles.M1, Tiles.M2, Tiles.M3, Tiles.M4, Tiles.M5, Tiles.M6, Tiles.M7, Tiles.M8, Tiles.M9);
    meld(context, Meld.pon(Tiles.P5, Meld.RelativeSource.KAMICHA, Meld.AkaSource.NONE));
    context.setRedFives(Tiles.bit(Tiles.P5));
    expectExhausted(scores, Tiles.P5);
  }

  /** 三人打ちで北を 3 枚抜いた後の、北の単騎待ち。抜いた北と合わせて 4 枚を使い切っているので採点しない。 */
  @Test
  public void aNorthWaitAfterThreeNukiIsNotScored() {
    WaitScores scores = new WaitScores(Variant.TENHOU_3P);
    WinContext context = scores.context();
    context.clear();
    concealed(
        context.hand(),
        Tiles.PEI, Tiles.P1, Tiles.P2, Tiles.P3, Tiles.P4, Tiles.P5, Tiles.P6, Tiles.S7, Tiles.S8, Tiles.S9);
    meld(context, Meld.pon(Tiles.HAKU, Meld.RelativeSource.KAMICHA, Meld.AkaSource.NONE));
    context.setNuki(3);
    expectExhausted(scores, Tiles.PEI);
  }

  private static void expectExhausted(WaitScores scores, int kind) {
    long waits = new ShapeAnalyzer(scores.context().variant()).waits(scores.context().hand());
    assertEquals(waits, Tiles.bit(kind));
    scores.load(waits, 0);
    assertEquals(scores.waits(), 1);
    assertEquals(scores.wait(0), kind);
    for (Method method : Method.values()) {
      for (WinningTile tile : WinningTile.values()) {
        assertEquals(scores.score(0, method, tile).base(), 0, method + " " + tile);
      }
    }
    DirectScoring brute = new DirectScoring(scores.context().variant());
    brute.check(scores, waits, 0);
    assertEquals(brute.exhaustedWaits(), 1);
  }

  /**
   * 和了の形（4 面子 1 雀頭）を乱数で作り、純手牌から 1 枚除いて聴牌にした手と、和了の条件を書く。3 回に 1 回は、面子のいくつかを
   * ポンかチー（チーのあるルールだけ）にする。北抜きのあるルールでは北も抜く。副露は赤を含まない形だけにするので、5 を含む面子は
   * 副露にしない。北抜きのあるルールでは北を鳴けないので、北の刻子も副露にしない。
   *
   * @param context 手と条件を書く先
   * @return 赤の和了牌でも採点する 5 の種類のビット集合。純手牌に赤を入れなかった種類のうち、4 枚を使い切っていない種類から乱数で選ぶ
   */
  private static long randomTenpai(Random random, Variant variant, WinContext context) {
    TileSet set = variant.tiles();
    int[] used = new int[Tiles.KINDS];
    context.clear();
    HandShape hand = context.hand();
    int pair = kind(random, set, used, 2);
    concealed(hand, pair, pair);
    used[pair] += 2;
    int opened = random.nextInt(3) == 0 ? 1 + random.nextInt(Variant.MAX_MELDS) : 0;
    for (int index = 0; index < Variant.MAX_MELDS; index++) {
      boolean open = index < opened;
      int start = random.nextBoolean() ? runStart(random, set, used) : -1;
      if (start >= 0) {
        used[start]++;
        used[start + 1]++;
        used[start + 2]++;
        boolean five = (Tiles.range(start, start + 2) & Tiles.FIVES) != 0;
        if (open && !five && variant.has(RuleFlag.CHI)) {
          meld(context, Meld.chi(start, start, Meld.AkaSource.NONE));
        } else {
          concealed(hand, start, start + 1, start + 2);
        }
      } else {
        int triplet = kind(random, set, used, 3);
        used[triplet] += 3;
        boolean five = Tiles.contains(Tiles.FIVES, triplet);
        if (open && !five && (triplet != Tiles.PEI || !variant.has(RuleFlag.NUKI))) {
          meld(context, Meld.pon(triplet, Meld.RelativeSource.KAMICHA, Meld.AkaSource.NONE));
        } else {
          concealed(hand, triplet, triplet, triplet);
        }
      }
    }
    int removed;
    do {
      removed = random.nextInt(Tiles.KINDS);
    } while (hand.count(removed) == 0);
    hand.remove(removed);
    used[removed]--;
    long handRed = 0;
    long red = 0;
    for (long fives = Tiles.FIVES & set.kinds(); fives != 0; fives &= fives - 1) {
      int five = Long.numberOfTrailingZeros(fives);
      if (!set.red(Tiles.physicalId(five, Tiles.RED_COPY))) {
        continue;
      }
      if (hand.count(five) == Tiles.COPIES || hand.count(five) > 0 && random.nextBoolean()) {
        handRed |= Tiles.bit(five);
      } else if (used[five] < Tiles.COPIES && random.nextBoolean()) {
        red |= Tiles.bit(five);
      }
    }
    context.setRedFives(handRed);
    int indicators = 1 + random.nextInt(WinContext.MAX_INDICATORS);
    for (int index = 0; index < indicators; index++) {
      context.addIndicator(kind(random, set, new int[Tiles.KINDS], 1));
    }
    int roundWinds = variant.roundWind(variant.finalRound()) - Tiles.TON + 1;
    context.setWinds(Tiles.TON + random.nextInt(variant.seats()), Tiles.TON + random.nextInt(roundWinds));
    RiichiState riichi = context.meldCount() == 0 ? RIICHI[random.nextInt(RIICHI.length)] : RiichiState.NONE;
    context.setRiichi(riichi, riichi != RiichiState.NONE && random.nextBoolean());
    if (variant.has(RuleFlag.NUKI)) {
      context.setNuki(random.nextInt(Tiles.COPIES - used[Tiles.PEI] + 1));
    }
    return red;
  }

  /** 牌の種類を乱数で選ぶ。選ぶのは、指定の枚数を足しても、使った枚数がその種類の枚数（ルールで決まる）を超えない種類。 */
  private static int kind(Random random, TileSet set, int[] used, int extra) {
    while (true) {
      int kind = random.nextInt(Tiles.KINDS);
      if (used[kind] + extra <= set.copies(kind)) {
        return kind;
      }
    }
  }

  /** 3 種類とも 1 枚ずつ足せる順子を乱数で選び、その一番小さい牌の種類を返す。そのルールで使わない牌は足せない。見つからなければ −1 を返す。 */
  private static int runStart(Random random, TileSet set, int[] used) {
    for (int attempt = 0; attempt < Tiles.KINDS; attempt++) {
      Tiles.Group group = Tiles.Group.values()[random.nextInt(Tiles.NUMBER_SUITS)];
      int start = group.first() + random.nextInt(group.size() - 2);
      boolean fits = true;
      for (int kind = start; kind < start + 3; kind++) {
        fits &= used[kind] < set.copies(kind);
      }
      if (fits) {
        return start;
      }
    }
    return -1;
  }

  private static void concealed(HandShape hand, int... kinds) {
    for (int kind : kinds) {
      hand.add(kind);
    }
  }

  private static void meld(WinContext context, Meld meld) {
    context.addMeld(meld);
    context.hand().addMeld();
  }
}
