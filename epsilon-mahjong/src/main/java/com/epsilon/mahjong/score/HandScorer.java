package com.epsilon.mahjong.score;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinContext.Source;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.WinningDecomposition;
import com.epsilon.mahjong.shape.WinningDecomposition.Wait;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;

/**
 * 天鳳のルールで和了を採点する。役・符・ドラを数えて基本点を求める。入力は {@link WinContext} だけで、採点のたびにオブジェクトを
 * 作らない。1 つのインスタンスを同時に使えるのは 1 スレッドだけ。
 *
 * <p>手の形は国士無双・七対子・通常の形の 3 つ。通常の形は、面子への分け方と待ちの形（和了牌がどこに入ったか）の組み合わせごとに
 * 採点する。ロンの双碰で完成した刻子は明刻に数える。どの組み合わせを取るかは高点法で決める。役満があれば役満を取り、無ければ
 * 基本点、翻、符の順に比べて高いものを取る。
 *
 * <p>役満でなければ、翻は役の翻にドラ・赤ドラ・裏ドラ・抜きドラを足したもの。ドラと裏ドラは、純手牌・面子・抜いた北から数える。
 * 抜きドラは抜いた北 1 枚につき 1 翻（天鳳の牌譜は、抜きドラをドラに含めて書き、裏ドラは別に書く）。ドラ・赤ドラ・裏ドラ・抜きドラ
 * だけでは 1 翻縛りを満たさない。裏ドラは立直した和了でだけ数える。役満ではドラを数えない。
 */
public final class HandScorer {

  private static final ScoringYaku[] YAKU = ScoringYaku.values();
  private static final Wait[] WAITS = Wait.values();

  /** 翻が付く役の集合。門前の手の分と、副露した手の分を別に持つ。 */
  private static final long CLOSED_YAKU = hanBits(true);

  private static final long OPEN_YAKU = hanBits(false);

  /** 三暗刻・三槓子に要る暗刻・槓子の数。 */
  private static final int THREE_SETS = 3;

  private static final long DRAGONS = Tiles.range(Tiles.HAKU, Tiles.CHUN);
  private static final long WINDS = Tiles.range(Tiles.TON, Tiles.PEI);
  private static final long GREEN =
      Tiles.bit(Tiles.S2)
          | Tiles.bit(Tiles.S3)
          | Tiles.bit(Tiles.S4)
          | Tiles.bit(Tiles.S6)
          | Tiles.bit(Tiles.S8)
          | Tiles.bit(Tiles.HATSU);

  /** 数牌の 3 色。{@code Group} で宣言した順に並ぶ。 */
  private static final Tiles.Group[] SUITS =
      Arrays.stream(Tiles.Group.values()).filter(Tiles.Group::suited).toArray(Tiles.Group[]::new);

  /** 色ごとに、その色の牌の種類の集合。 */
  private static final long[] SUIT_KINDS =
      Arrays.stream(SUITS).mapToLong(suit -> Tiles.range(suit.first(), suit.last())).toArray();

  /** 一気通貫になる 3 つの順子（123・456・789）の先頭の牌を、色ごとに集合で持つ。 */
  private static final long[] STRAIGHTS =
      Arrays.stream(SUITS)
          .mapToLong(
              suit ->
                  Tiles.bit(suit.first())
                      | Tiles.bit(suit.first() + HandShape.MELD_SIZE)
                      | Tiles.bit(suit.first() + 2 * HandShape.MELD_SIZE))
          .toArray();

  /** 么九牌を含む順子（123・789）の先頭の牌を、3 色まとめて集合で持つ。 */
  private static final long OUTSIDE_RUNS =
      Arrays.stream(SUITS)
          .mapToLong(suit -> Tiles.bit(suit.first()) | Tiles.bit(suit.last() - HandShape.MELD_SIZE + 1))
          .reduce(0, (a, b) -> a | b);

  /** 1 色の 9 つの数（0 始まり）に全てビットを立てたマスク。 */
  private static final long NUMBERS = (1L << Tiles.SUIT_SIZE) - 1;

  private final HandShape hand = new HandShape();
  private final WinningDecomposition decomposition = new WinningDecomposition();

  private WinContext context;
  private int winning;
  private boolean tsumo;
  private boolean closed;

  /** 純手牌（和了牌を含む）と面子を合わせた牌の種類と、面子だけの牌の種類。 */
  private long kinds;

  private long meldKinds;

  /** 面子のうち、刻子と槓子の牌の種類。 */
  private long meldTriplets;

  /** チーした順子の先頭の牌の種類と、チーの数。 */
  private long meldRuns;

  private int meldRunCount;

  /** 面子に入っている赤 5 の種類の集合。 */
  private long meldReds;

  private int ankans;
  private int kans;
  private int meldFu;

  /** 面子への分け方で変わらない役と役満。 */
  private long handYaku;

  private long handYakuman;

  /** ドラ・赤ドラ・裏ドラ・抜きドラの翻。役満でないときに、役の翻へ足す。 */
  private int dora;

  private int aka;
  private int ura;
  private int nuki;

  /** これまでに見た中で最も高い組み合わせ。基本点が 0 なら、役のある組み合わせはまだ見つかっていない。 */
  private long bestYaku;

  private int bestHan;
  private int bestFu;
  private int bestYakuman;
  private int bestBase;

  /**
   * 和了を採点して結果に書く。役が無ければ結果を空にする。
   *
   * <p>入力のどこかがおかしいと IllegalArgumentException を投げる。調べるのは、手の枚数、和了牌、面子の数、同じ種類の 5 枚目、
   * そのルールで使わない牌、赤 5 と抜いた北の枚数、副露した手の立直、第一ツモ（面子・立直・抜きの無い手のツモに限る）、嶺上開花
   * （和了した人の槓か抜きの後に限り、一発は付かない）、裏ドラ表示牌の数（0 か、ドラ表示牌と同じ数）。
   *
   * @param context 採点する和了
   * @param out 結果を書く先
   * @return 役があれば true
   */
  public boolean score(WinContext context, WinScore out) {
    load(context);
    bestYaku = 0;
    bestHan = 0;
    bestFu = 0;
    bestYakuman = 0;
    bestBase = 0;
    boolean concealedOnly = context.meldCount() == 0;
    if (concealedOnly && hand.kinds() == Tiles.TERMINALS_AND_HONORS) {
      consider(0, handYakuman | ScoringYaku.KOKUSHI.bit(), 0);
    } else {
      if (concealedOnly && Long.bitCount(hand.pairs()) == HandShape.SEVEN_PAIRS) {
        consider(handYaku | ScoringYaku.CHIITOITSU.bit(), handYakuman, Fu.CHIITOITSU);
      }
      scoreDecompositions();
    }
    if (bestBase == 0) {
      out.clear();
      return false;
    }
    if (bestYakuman > 0) {
      out.set(bestYaku, closed, 0, 0, 0, 0, 0, 0, bestYakuman, bestBase);
    } else {
      out.set(bestYaku, closed, bestHan, bestFu, dora, aka, ura, nuki, 0, bestBase);
    }
    return true;
  }

  private void scoreDecompositions() {
    decomposition.bind(hand);
    while (decomposition.next()) {
      long concealedTriplets = decomposition.triplets();
      long triplets = concealedTriplets | meldTriplets;
      long runs = decomposition.runs() | meldRuns;
      int runCount = decomposition.meldCount() - Long.bitCount(concealedTriplets) + meldRunCount;
      int pair = decomposition.pair();
      long yaku = handYaku | setYaku(triplets, runs, runCount, pair);
      long yakuman = handYakuman | setYakuman(triplets, pair);
      int pairFu = Fu.pair(pair, context.seatWind(), context.roundWind());
      int concealedCount = Long.bitCount(concealedTriplets) + ankans;
      for (int waits = decomposition.waits(winning); waits != 0; waits &= waits - 1) {
        Wait wait = WAITS[Integer.numberOfTrailingZeros(waits)];
        boolean opened = !tsumo && wait == Wait.SHANPON;
        int concealed = concealedCount - (opened ? 1 : 0);
        boolean pinfu = closed && runCount == Variant.MAX_MELDS && pairFu == 0 && wait == Wait.RYANMEN;
        int tripletFu =
            opened
                ? Fu.triplets(concealedTriplets & ~Tiles.bit(winning), true) + Fu.triplet(winning, false, false)
                : Fu.triplets(concealedTriplets, true);
        consider(
            yaku | when(ScoringYaku.SANANKOU, concealed >= THREE_SETS) | when(ScoringYaku.PINFU, pinfu),
            yakuman | when(ScoringYaku.SUUANKOU, concealed == Variant.MAX_MELDS),
            Fu.total(pairFu + meldFu + tripletFu + Fu.wait(wait), tsumo, closed, pinfu));
      }
    }
  }

  /**
   * 面子への分け方で決まる役。刻子の種類、順子の先頭の牌、順子の数、雀頭から求める。刻子と順子には副露した面子も含める。
   */
  private long setYaku(long triplets, long runs, int runCount, int pair) {
    long yaku =
        when(ScoringYaku.YAKUHAI_HAKU, Tiles.contains(triplets, Tiles.HAKU))
            | when(ScoringYaku.YAKUHAI_HATSU, Tiles.contains(triplets, Tiles.HATSU))
            | when(ScoringYaku.YAKUHAI_CHUN, Tiles.contains(triplets, Tiles.CHUN))
            | when(ScoringYaku.YAKUHAI_SEAT, Tiles.contains(triplets, context.seatWind()))
            | when(ScoringYaku.YAKUHAI_ROUND, Tiles.contains(triplets, context.roundWind()))
            | when(ScoringYaku.SANSHOKU, inEverySuit(runs))
            | when(ScoringYaku.SANSHOKU_DOUKOU, inEverySuit(triplets))
            | when(ScoringYaku.ITTSU, straight(runs))
            | when(ScoringYaku.TOITOI, runCount == 0)
            | when(ScoringYaku.SHOUSANGEN, completes(DRAGONS, triplets, pair));
    if (closed) {
      int peikou = peikou();
      yaku |= when(ScoringYaku.IPEIKOU, peikou == 1) | when(ScoringYaku.RYANPEIKOU, peikou > 1);
    }
    boolean outside =
        runCount > 0
            && Tiles.contains(Tiles.TERMINALS_AND_HONORS, pair)
            && (triplets & ~Tiles.TERMINALS_AND_HONORS) == 0
            && (runs & ~OUTSIDE_RUNS) == 0;
    if (outside) {
      yaku |= ((kinds & Tiles.HONORS) == 0 ? ScoringYaku.JUNCHAN : ScoringYaku.CHANTA).bit();
    }
    return yaku;
  }

  private static long setYakuman(long triplets, int pair) {
    return when(ScoringYaku.DAISANGEN, (triplets & DRAGONS) == DRAGONS)
        | when(ScoringYaku.DAISUUSHII, (triplets & WINDS) == WINDS)
        | when(ScoringYaku.SHOUSUUSHII, completes(WINDS, triplets, pair));
  }

  /** 雀頭が集合の牌で、刻子と雀頭を合わせると集合の種類が全てそろうか（小三元・小四喜）。 */
  private static boolean completes(long group, long triplets, int pair) {
    return Tiles.contains(group, pair) && ((triplets | Tiles.bit(pair)) & group) == group;
  }

  /**
   * 同じ数の牌が 3 色とも集合にあるか（三色同順・三色同刻）。各色をその色の 1 が最下位のビットに来るまでずらして重ね、3 色に
   * 共通の数があるかを見る。
   */
  private static boolean inEverySuit(long kinds) {
    long numbers = NUMBERS;
    for (Tiles.Group suit : SUITS) {
      numbers &= kinds >>> suit.first();
    }
    return numbers != 0;
  }

  private static boolean straight(long runs) {
    for (long straight : STRAIGHTS) {
      if ((runs & straight) == straight) {
        return true;
      }
    }
    return false;
  }

  /**
   * 同じ順子 2 つの組がいくつあるか（一盃口なら 1、二盃口なら 2）。分けた面子は種類の順に並び、同じ種類では刻子が先に来る。
   * そのため同じ順子は隣り合う。
   */
  private int peikou() {
    int pairs = 0;
    for (int i = 1; i < decomposition.meldCount(); i++) {
      if (decomposition.run(i)
          && decomposition.run(i - 1)
          && decomposition.meld(i) == decomposition.meld(i - 1)) {
        pairs++;
        i++;
      }
    }
    return pairs;
  }

  /**
   * この組み合わせを、これまでで最も高いものと高点法で比べ、高ければ残す。
   *
   * @param yaku 役の集合
   * @param yakuman 役満の集合。0 でなければ役満として比べる
   * @param fu 符
   */
  private void consider(long yaku, long yakuman, int fu) {
    if (yakuman != 0) {
      int multiple = Long.bitCount(yakuman);
      int base = ScoreMath.base(0, 0, multiple);
      if (bestYakuman == 0 || base > bestBase) {
        keep(yakuman, 0, 0, multiple, base);
      }
      return;
    }
    long counted = yaku & (closed ? CLOSED_YAKU : OPEN_YAKU);
    if (bestYakuman > 0 || counted == 0) {
      return;
    }
    int han = han(counted) + dora + aka + ura + nuki;
    int base = ScoreMath.base(han, fu, 0);
    if (base > bestBase || base == bestBase && (han > bestHan || han == bestHan && fu > bestFu)) {
      keep(counted, han, fu, 0, base);
    }
  }

  private void keep(long yaku, int han, int fu, int yakuman, int base) {
    bestYaku = yaku;
    bestHan = han;
    bestFu = fu;
    bestYakuman = yakuman;
    bestBase = base;
  }

  private int han(long yaku) {
    int han = 0;
    for (long rest = yaku; rest != 0; rest &= rest - 1) {
      han += YAKU[Long.numberOfTrailingZeros(rest)].han(closed);
    }
    return han;
  }

  private void load(WinContext context) {
    Source source = context.source();
    if (source == null) {
      throw new IllegalArgumentException("No winning tile is set: " + context);
    }
    HandShape given = context.hand();
    if (given.melds() != context.meldCount()) {
      throw new IllegalArgumentException(
          "The hand counts " + given.melds() + " melds but " + context.meldCount() + " are given: " + context);
    }
    int kind = context.winningKind();
    hand.copyFrom(given);
    if (given.size() == HandShape.MAX_SIZE - 1 && given.count(kind) < Tiles.COPIES) {
      hand.add(kind);
    } else if (given.size() != HandShape.MAX_SIZE || given.count(kind) == 0) {
      throw new IllegalArgumentException(
          "A win needs 13 tiles and the winning tile, or 14 tiles with it: " + context);
    }
    this.context = context;
    winning = kind;
    tsumo = source.tsumo();
    loadMelds();
    kinds = hand.kinds() | meldKinds;
    validate(source);
    loadYaku();
    loadDora();
  }

  private void loadMelds() {
    closed = true;
    meldKinds = 0;
    meldTriplets = 0;
    meldRuns = 0;
    meldRunCount = 0;
    meldReds = 0;
    ankans = 0;
    kans = 0;
    meldFu = 0;
    for (int i = 0; i < context.meldCount(); i++) {
      Meld meld = context.meld(i);
      Meld.Type type = meld.type();
      closed &= type.concealed();
      meldKinds |= meld.kinds();
      meldFu += Fu.meld(meld);
      if (type == Meld.Type.CHI) {
        meldRuns |= Tiles.bit(meld.base());
        meldRunCount++;
      } else {
        meldTriplets |= Tiles.bit(meld.base());
      }
      kans += type.kan() ? 1 : 0;
      ankans += type == Meld.Type.ANKAN ? 1 : 0;
      if (meld.red()) {
        long red = meld.kinds() & Tiles.FIVES;
        if ((meldReds & red) != 0) {
          throw new IllegalArgumentException("Two melds hold the same red five: " + context);
        }
        meldReds |= red;
      }
    }
  }

  private void validate(Source source) {
    if ((hand.kinds() & ~context.variant().tiles().kinds()) != 0) {
      throw new IllegalArgumentException(context.variant() + " does not use some tiles of " + context);
    }
    for (long rest = meldKinds; rest != 0; rest &= rest - 1) {
      if (count(Long.numberOfTrailingZeros(rest)) > Tiles.COPIES) {
        throw new IllegalArgumentException("A kind has more than " + Tiles.COPIES + " tiles in " + context);
      }
    }
    if (context.nuki() > 0 && count(Tiles.PEI) + context.nuki() > Tiles.COPIES) {
      throw new IllegalArgumentException("The hand and the nuki hold more than four north tiles: " + context);
    }
    long reds = context.redFives();
    if ((reds & ~hand.kinds()) != 0 || (reds & meldReds) != 0) {
      throw new IllegalArgumentException("The red fives are not among the concealed tiles: " + context);
    }
    if (context.riichi() != RiichiState.NONE && !closed) {
      throw new IllegalArgumentException("Riichi needs a closed hand: " + context);
    }
    if (context.firstDraw() && source != Source.TSUMO) {
      throw new IllegalArgumentException("Only a plain tsumo can be the first draw: " + context);
    }
    if (context.firstDraw()
        && (context.meldCount() > 0 || context.riichi() != RiichiState.NONE || context.nuki() > 0)) {
      throw new IllegalArgumentException("A first draw cannot follow a meld, a riichi or a nuki: " + context);
    }
    if (source == Source.RINSHAN && kans == 0 && context.nuki() == 0) {
      throw new IllegalArgumentException("A rinshan draw needs a kan or a nuki of the winner: " + context);
    }
    if (source == Source.RINSHAN && context.ippatsu()) {
      throw new IllegalArgumentException("The kan or nuki before a rinshan draw ends the ippatsu: " + context);
    }
    int uraIndicators = context.uraIndicatorCount();
    if (uraIndicators != 0 && uraIndicators != context.indicatorCount()) {
      throw new IllegalArgumentException("Ura indicators must match the dora indicators: " + context);
    }
  }

  private void loadYaku() {
    long yaku =
        switch (context.riichi()) {
          case NONE -> 0;
          case RIICHI -> ScoringYaku.RIICHI.bit();
          case DOUBLE_RIICHI -> ScoringYaku.DOUBLE_RIICHI.bit();
        };
    yaku |= when(ScoringYaku.IPPATSU, context.ippatsu()) | when(ScoringYaku.MENZEN_TSUMO, closed && tsumo);
    ScoringYaku sourceYaku = context.source().yaku();
    if (sourceYaku != null) {
      yaku |= sourceYaku.bit();
    }
    int suits = suitCount(kinds);
    boolean honors = (kinds & Tiles.HONORS) != 0;
    yaku |=
        when(ScoringYaku.TANYAO, (kinds & Tiles.TERMINALS_AND_HONORS) == 0)
            | when(ScoringYaku.HONROUTOU, (kinds & ~Tiles.TERMINALS_AND_HONORS) == 0)
            | when(ScoringYaku.HONITSU, suits == 1 && honors)
            | when(ScoringYaku.CHINITSU, suits == 1 && !honors)
            | when(ScoringYaku.SANKANTSU, kans == THREE_SETS);
    handYaku = yaku;
    long yakuman =
        when(ScoringYaku.TSUUIISOU, suits == 0)
            | when(ScoringYaku.RYUUIISOU, (kinds & ~GREEN) == 0)
            | when(ScoringYaku.CHINROUTOU, (kinds & ~Tiles.TERMINALS) == 0)
            | when(ScoringYaku.SUUKANTSU, kans == Variant.MAX_MELDS)
            | when(ScoringYaku.CHUUREN, suits == 1 && !honors && chuuren());
    if (context.firstDraw()) {
      yakuman |= (context.seatWind() == Tiles.TON ? ScoringYaku.TENHOU : ScoringYaku.CHIIHOU).bit();
    }
    handYakuman = yakuman;
  }

  /** 九蓮宝燈の形か。面子が無く、1 と 9 が 3 枚以上、2〜8 が 1 枚以上ある。1 色の手であることは呼び出し元が確かめる。 */
  private boolean chuuren() {
    if (context.meldCount() > 0) {
      return false;
    }
    Tiles.Group suit = Tiles.Group.of(Long.numberOfTrailingZeros(kinds));
    for (int kind = suit.first(); kind <= suit.last(); kind++) {
      boolean terminal = kind == suit.first() || kind == suit.last();
      if (hand.count(kind) < (terminal ? HandShape.MELD_SIZE : 1)) {
        return false;
      }
    }
    return true;
  }

  private void loadDora() {
    TileSet tiles = context.variant().tiles();
    dora = 0;
    for (int i = 0; i < context.indicatorCount(); i++) {
      dora += doraCount(tiles.doraOf(context.indicator(i)));
    }
    ura = 0;
    if (context.riichi() != RiichiState.NONE) {
      for (int i = 0; i < context.uraIndicatorCount(); i++) {
        ura += doraCount(tiles.doraOf(context.uraIndicator(i)));
      }
    }
    aka = Long.bitCount(context.redFives()) + Long.bitCount(meldReds);
    nuki = context.nuki();
  }

  /**
   * ドラ 1 種類について、純手牌と面子にある枚数を数える。北なら抜いた北の数も足す。
   *
   * @param kind ドラの種類（表示牌の次の牌）
   */
  private int doraCount(int kind) {
    return count(kind) + (kind == Tiles.PEI ? context.nuki() : 0);
  }

  /** その種類の牌が、純手牌（和了牌を含む）と面子に何枚あるか。チーの面子に同じ種類は 1 枚しか無いので、1 枚と数える。 */
  private int count(int kind) {
    int count = hand.count(kind);
    for (int i = 0; i < context.meldCount(); i++) {
      Meld meld = context.meld(i);
      if (Tiles.contains(meld.kinds(), kind)) {
        count += meld.type() == Meld.Type.CHI ? 1 : meld.size();
      }
    }
    return count;
  }

  private static int suitCount(long kinds) {
    int suits = 0;
    for (long suit : SUIT_KINDS) {
      if ((kinds & suit) != 0) {
        suits++;
      }
    }
    return suits;
  }

  private static long when(ScoringYaku yaku, boolean holds) {
    return holds ? yaku.bit() : 0;
  }

  private static long hanBits(boolean closed) {
    long bits = 0;
    for (ScoringYaku yaku : YAKU) {
      if (yaku.han(closed) > 0) {
        bits |= yaku.bit();
      }
    }
    return bits;
  }
}
