package com.epsilon.classic.rulebot;

import com.epsilon.agent.facts.ShapeFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.facts.WaitScores;
import com.epsilon.agent.facts.WaitScores.Method;
import com.epsilon.agent.facts.WaitScores.WinningTile;
import com.epsilon.agent.observe.Observation;
import com.epsilon.classic.input.OwnHand;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 合法手を選んだ後の自分の手（{@link OwnHand}）について、形・受け入れ・待ちの点数を求めて {@link Quality} に書く。待ちの点数は、
 * 聴牌の手を和了の条件（{@link WinContext}）に入れて採点する。判断のたびに {@link #load} で今の手を読み込む。解析ではメモリを確保
 * しない。1 つのインスタンスを同時に使えるのは 1 スレッドだけ。
 *
 * <p>待ちの点数では、和了牌を赤でない牌として、一発・嶺上・海底・河底・槍槓・天和・地和と裏ドラは数えない。打牌・槓の後の手は、渡された
 * 立直の状態で採点する。鳴いた後の手と応答の時点の手は、立直なしで採点する。
 */
final class Projection {

  /** 2 枚目の受け入れを数える向聴の範囲。 */
  private static final int SECOND_ORDER_MIN_SHANTEN = 1;

  private static final int SECOND_ORDER_MAX_SHANTEN = 2;

  /** 順子の枚数。 */
  private static final int RUN = 3;

  /** 一気通貫の順子の始まり（色の中の 1・4・7）。 */
  private static final int[] STRAIGHT = {0, 3, 6};

  private static final Tiles.Group[] SUITS = {Tiles.Group.MAN, Tiles.Group.PIN, Tiles.Group.SOU};

  private final ShapeFacts shape;
  private final ShapeAnalyzer analyzer;
  private final WaitScores waits;
  private final WinContext context;
  private final HandShape probe = new HandShape();
  private final int[] remaining = new int[Tiles.KINDS];

  /** 判断の時点の自分の手と、いま調べている合法手を選んだ後の手。 */
  private final OwnHand current;

  private final OwnHand after;

  private Observation o;
  private Situation situation;
  private TileCounts tiles;

  Projection(Variant variant) {
    shape = new ShapeFacts(variant);
    analyzer = new ShapeAnalyzer(variant);
    waits = new WaitScores(variant);
    context = waits.context();
    current = new OwnHand(variant);
    after = new OwnHand(variant);
  }

  /**
   * 判断の時点の自分の手を読み込む。
   *
   * @param situation 同じ判断の記録から読んだ場の状況
   */
  void load(Observation o, Situation situation) {
    this.o = o;
    this.situation = situation;
    tiles = situation.tiles();
    current.load(tiles, situation.facts());
  }

  /** 応答の時点の今の手（13 枚相当）を、立直なしで解析する。 */
  Quality current(Quality out) {
    return analyze(current, RiichiState.NONE, out);
  }

  /**
   * 打牌の後の手を解析する。
   *
   * @param discard 打牌か、立直を宣言する打牌
   * @param riichi 打牌の後の立直の状態
   */
  Quality discard(Action discard, RiichiState riichi, Quality out) {
    after.play(current, discard, null, null);
    return analyze(after, riichi, out);
  }

  /**
   * チーかポンをして打牌した後の手を、立直なしで解析する。
   *
   * @param call チーかポン
   * @param discard 鳴いた後に打つ牌
   */
  Quality call(Action call, Action discard, Quality out) {
    after.play(current, call, current.meldOf(o, call), discard);
    return analyze(after, RiichiState.NONE, out);
  }

  /**
   * 大明槓・暗槓・加槓の後（嶺上を引く前の 13 枚相当）の手を解析する。
   *
   * @param kan 大明槓・暗槓・加槓
   * @param riichi 槓の後の立直の状態
   */
  Quality kan(Action kan, RiichiState riichi, Quality out) {
    Meld meld = current.meldOf(o, kan);
    if (meld == null || !meld.type().kan()) {
      throw new IllegalArgumentException("Not a kan: " + kan);
    }
    after.play(current, kan, meld, null);
    return analyze(after, riichi, out);
  }

  /**
   * 今の手（14 枚相当）の向聴が、通常形・七対子・国士のどれでも上限を超えるかを返す。面子のある手では、七対子と国士は数えない。
   *
   * @param limit 向聴の上限
   */
  boolean far(int limit) {
    shape.analyze(current.shape(), tiles);
    if (shape.standardShanten() <= limit) {
      return false;
    }
    return !shape.special() || shape.chiitoitsuShanten() > limit && shape.kokushiShanten() > limit;
  }

  /** 手を解析して、結果を書く。 */
  private Quality analyze(OwnHand hand, RiichiState riichi, Quality out) {
    shape.analyze(hand.shape(), tiles);
    int shanten = shape.shanten();
    long improving = shape.improving();
    boolean special = shape.special();
    int secondOrder =
        shanten >= SECOND_ORDER_MIN_SHANTEN && shanten <= SECOND_ORDER_MAX_SHANTEN
            ? secondOrder(hand.shape(), improving)
            : 0;
    out.setShape(
        shanten,
        special,
        special ? shape.chiitoitsuShanten() : 0,
        shape.acceptance(),
        OwnHand.liveKinds(tiles, improving),
        secondOrder,
        shape.waits());
    describeHand(hand, out);
    scoreWaits(hand, riichi, out);
    return out;
  }

  /** 手の牌と面子から決まる値を書く。 */
  private void describeHand(OwnHand hand, Quality out) {
    long kinds = hand.shape().kinds();
    boolean simple = (kinds & Tiles.TERMINALS_AND_HONORS) == 0;
    int suits = suits(kinds);
    long runs = 0;
    int confirmed = 0;
    for (int index = 0; index < hand.meldCount(); index++) {
      Meld meld = hand.meld(index);
      simple &= (meld.kinds() & Tiles.TERMINALS_AND_HONORS) == 0;
      suits |= suits(meld.kinds());
      if (meld.type() == Meld.Type.CHI) {
        runs |= Tiles.bit(meld.base());
      } else {
        confirmed += yakuhai(meld.base());
      }
    }
    if (hand.meldCount() == Variant.MAX_MELDS && runs == 0) {
      confirmed++;
    }
    confirmed += straights(runs) + (threeColors(runs) ? 1 : 0);
    out.setHand(simple, Integer.bitCount(suits) <= 1, hand.dora(tiles), hand.aka(), confirmed);
  }

  /** 聴牌の手について、役のある待ちの枚数とロンの点数を書く。聴牌でなければ全て 0 を書く。 */
  private void scoreWaits(OwnHand hand, RiichiState riichi, Quality out) {
    long shapeWaits = shape.waits();
    if (shapeWaits == 0) {
      out.setWaits(0, 0, 0, 0, 0);
      return;
    }
    context.clear();
    context.hand().copyFrom(hand.shape());
    for (int index = 0; index < hand.meldCount(); index++) {
      context.addMeld(hand.meld(index));
    }
    context.setRedFives(hand.reds());
    context.setWinds(situation.seatWind(), situation.roundWind());
    for (int index = 0; index < o.indicators(); index++) {
      context.addIndicator(o.indicatorTile(index));
    }
    context.setRiichi(riichi, false);
    context.setNuki(o.nuki(0));
    waits.load(shapeWaits, 0);
    int liveWaits = 0;
    int liveWaitKinds = 0;
    int liveRon = 0;
    int minimumRon = Integer.MAX_VALUE;
    long ronTotal = 0;
    for (int index = 0; index < waits.waits(); index++) {
      int left = tiles.unseen(waits.wait(index));
      if (left <= 0) {
        continue;
      }
      WinScore ron = waits.score(index, Method.RON, WinningTile.PLAIN);
      WinScore tsumo = waits.score(index, Method.TSUMO, WinningTile.PLAIN);
      if (ron.base() > 0 || tsumo.base() > 0) {
        liveWaits += left;
        liveWaitKinds++;
      }
      if (ron.base() > 0) {
        int points = ScoreMath.ron(ron.base(), situation.dealer());
        liveRon += left;
        minimumRon = Math.min(minimumRon, points);
        ronTotal += (long) left * points;
      }
    }
    if (liveRon == 0) {
      out.setWaits(liveWaits, liveWaitKinds, 0, 0, 0);
    } else {
      out.setWaits(liveWaits, liveWaitKinds, liveRon, minimumRon, (int) Math.round(ronTotal / (double) liveRon));
    }
  }

  /**
   * 2 枚目の受け入れを返す。受け入れの牌を 1 枚引き、いちばん良い牌（向聴が小さく、同じなら受け入れが多い）を打った後の受け入れの
   * 枚数を求める。これを、引く牌の生きている枚数で重みを付けて平均する。引いた後の受け入れは、引いた 1 枚を除いて数える。
   *
   * @param hand 解析する手の純手牌と面子の数
   */
  private int secondOrder(HandShape hand, long improving) {
    probe.copyFrom(hand);
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      remaining[kind] = tiles.unseen(kind);
    }
    long weighted = 0;
    int weights = 0;
    for (long draws = improving; draws != 0; draws &= draws - 1) {
      int draw = Long.numberOfTrailingZeros(draws);
      int weight = remaining[draw];
      if (weight <= 0) {
        continue;
      }
      probe.add(draw);
      remaining[draw]--;
      int bestShanten = Integer.MAX_VALUE;
      int bestUkeire = 0;
      for (long discards = probe.kinds(); discards != 0; discards &= discards - 1) {
        int discard = Long.numberOfTrailingZeros(discards);
        probe.remove(discard);
        int shanten = analyzer.shanten(probe);
        int ukeire = remaining(analyzer.improving(probe));
        if (shanten < bestShanten || shanten == bestShanten && ukeire > bestUkeire) {
          bestShanten = shanten;
          bestUkeire = ukeire;
        }
        probe.add(discard);
      }
      remaining[draw]++;
      probe.remove(draw);
      weighted += (long) weight * bestUkeire;
      weights += weight;
    }
    return weights == 0 ? 0 : (int) Math.round(weighted / (double) weights);
  }

  /** 牌の種類の集合について、2 枚目の受け入れを数えるときの残りの枚数を足し合わせる。 */
  private int remaining(long kinds) {
    int sum = 0;
    for (long rest = kinds; rest != 0; rest &= rest - 1) {
      sum += remaining[Long.numberOfTrailingZeros(rest)];
    }
    return sum;
  }

  /** この種類の刻子・槓子で付く役牌の数（三元牌・自風・場風。連風牌は 2）。 */
  private int yakuhai(int kind) {
    int count = Tiles.Group.of(kind) == Tiles.Group.DRAGON ? 1 : 0;
    count += kind == situation.seatWind() ? 1 : 0;
    count += kind == situation.roundWind() ? 1 : 0;
    return count;
  }

  /** 含まれる数牌の色のビット（萬・筒・索の順）。 */
  private static int suits(long kinds) {
    int suits = 0;
    for (int suit = 0; suit < SUITS.length; suit++) {
      if ((kinds & Tiles.range(SUITS[suit].first(), SUITS[suit].last())) != 0) {
        suits |= 1 << suit;
      }
    }
    return suits;
  }

  /** 鳴いた順子だけで一気通貫になる色の数。 */
  private static int straights(long runs) {
    int count = 0;
    for (Tiles.Group suit : SUITS) {
      boolean all = true;
      for (int start : STRAIGHT) {
        all &= Tiles.contains(runs, suit.first() + start);
      }
      count += all ? 1 : 0;
    }
    return count;
  }

  /** 鳴いた順子だけで三色同順になるか。 */
  private static boolean threeColors(long runs) {
    for (int start = 0; start <= Tiles.SUIT_SIZE - RUN; start++) {
      boolean all = true;
      for (Tiles.Group suit : SUITS) {
        all &= Tiles.contains(runs, suit.first() + start);
      }
      if (all) {
        return true;
      }
    }
    return false;
  }
}
