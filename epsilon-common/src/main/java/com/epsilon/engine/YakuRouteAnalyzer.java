package com.epsilon.engine;

import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;

/**
 * 公開情報と行動適用後の手牌から、確定面子と両立する役の種類を求める。
 *
 * <p>確定面子の構成、残りの面子を作れる数、役牌刻子に必要な牌の未確認枚数を調べる。現在の手牌に不要な牌があっても、後で捨てられるため対象の役を除外しない。
 *
 * <p>結果は成立の可能性を構造上残している役の一覧であり、和了確率や最低打点を保証するものではない。
 */
public final class YakuRouteAnalyzer {

  private static final int SEQUENCE_START_MASK = 0x7f;
  private static final int ITTSU_START_MASK = 0b1001001;

  /** 方策入力へ独立に渡す役の候補。染め手と一気通貫は対象色まで保持する。 */
  public enum Route {
    /** 白の刻子による役牌。 */
    YAKUHAI_HAKU,
    /** 發の刻子による役牌。 */
    YAKUHAI_HATSU,
    /** 中の刻子による役牌。 */
    YAKUHAI_CHUN,
    /** 自風の刻子による役牌。 */
    YAKUHAI_SEAT_WIND,
    /** 場風の刻子による役牌。 */
    YAKUHAI_ROUND_WIND,
    /** 断么九。 */
    TANYAO,
    /** 対々和。 */
    TOITOI,
    /** 萬子の混一色。 */
    HONITSU_MAN,
    /** 筒子の混一色。 */
    HONITSU_PIN,
    /** 索子の混一色。 */
    HONITSU_SOU,
    /** 萬子の清一色。 */
    CHINITSU_MAN,
    /** 筒子の清一色。 */
    CHINITSU_PIN,
    /** 索子の清一色。 */
    CHINITSU_SOU,
    /** 混全帯么九。 */
    CHANTA,
    /** 純全帯么九。 */
    JUNCHAN,
    /** 萬子の一気通貫。 */
    ITTSU_MAN,
    /** 筒子の一気通貫。 */
    ITTSU_PIN,
    /** 索子の一気通貫。 */
    ITTSU_SOU,
    /** 三色同順。 */
    SANSHOKU,
    /** 混老頭。 */
    HONROUTOU
  }

  private static final Route[] ROUTES = Route.values();
  private static final long ROUTE_MASK = (1L << ROUTES.length) - 1L;

  /** 成立する可能性のある役と、確定面子だけで成立条件を満たす役を区別して保持する。 */
  public record Analysis(long viableRouteBits, long confirmedRouteBits) {

    /** 指定した役が成立する可能性を残していればtrue。 */
    public boolean isViable(Route route) {
      return (viableRouteBits & routeBit(route)) != 0L;
    }

    /** 指定した役の成立条件を確定面子だけで満たしていればtrue。 */
    public boolean isConfirmed(Route route) {
      return (confirmedRouteBits & routeBit(route)) != 0L;
    }
  }

  private YakuRouteAnalyzer() {}

  /**
   * 高頻度の入力生成で一時集合を作らず、方策スキーマと同じビット配置を返す。
   *
   * @param seatWind 評価席の自風牌種
   * @param roundWind 場風牌種
   * @param hand 起点行動適用後の手牌
   * @param visibleTileCounts 手牌を含まない、意思決定時点の公開牌枚数
   * @return {@link Route#ordinal()} をビット位置とするマスク
   */
  public static long viableRouteBits(
      int seatWind, int roundWind, HandView hand, int[] visibleTileCounts) {
    return analyzePacked(seatWind, roundWind, hand, visibleTileCounts) & ROUTE_MASK;
  }

  /** 固定面子だけで確定する役を返す。公開牌や未確定の役は計算しない。 */
  public static long confirmedRouteBits(int seatWind, int roundWind, HandView hand) {
    long confirmed = 0L;
    long sequenceBases = 0L;
    for (int index = 0; index < hand.meldCount(); index++) {
      Meld meld = hand.meld(index);
      if (meld.type() == Meld.Type.CHI) sequenceBases |= 1L << meld.baseTileType();
      else confirmed |= confirmedYakuhaiBits(meld.baseTileType(), seatWind, roundWind);
    }
    if (hand.meldCount() == 4 && sequenceBases == 0L) confirmed |= routeBit(Route.TOITOI);
    return confirmed
        | confirmedSequenceRouteBits(
            sequenceStarts(sequenceBases, Tile.SUIT_MAN),
            sequenceStarts(sequenceBases, Tile.SUIT_PIN),
            sequenceStarts(sequenceBases, Tile.SUIT_SOU));
  }

  /**
   * 成立する可能性のある役と、固定済み面子だけで保証された役を一度の面子走査で求める。
   *
   * <p>確定扱いするのは、今後作る雀頭や暗刻・順子によらず消えない役牌・一気通貫・三色同順と、4面子がすべて刻子系の対々和だけである。
   */
  public static Analysis analyze(
      int seatWind, int roundWind, HandView hand, int[] visibleTileCounts) {
    long packed = analyzePacked(seatWind, roundWind, hand, visibleTileCounts);
    return new Analysis(packed & ROUTE_MASK, packed >>> 32);
  }

  private static long analyzePacked(
      int seatWind, int roundWind, HandView hand, int[] visibleTileCounts) {
    long tripletTileMask = 0L;
    long chiBaseTileMask = 0L;
    long confirmedRoutes = 0L;
    int honitsuSuitMask = 0b111;
    int chinitsuSuitMask = 0b111;
    boolean allFixedTilesAreTanyao = true;
    boolean allMeldsAreChanta = true;
    boolean allMeldsAreJunchan = true;
    boolean allMeldsAreHonroutou = true;
    for (int meldIndex = 0; meldIndex < hand.meldCount(); meldIndex++) {
      Meld meld = hand.meld(meldIndex);
      if (meld.type() == Meld.Type.CHI) {
        chiBaseTileMask |= 1L << meld.baseTileType();
      } else {
        int tripletTileType = meld.baseTileType();
        tripletTileMask |= 1L << tripletTileType;
        confirmedRoutes |= confirmedYakuhaiBits(tripletTileType, seatWind, roundWind);
      }
      int base = meld.baseTileType();
      boolean honor = Tile.isHonor(base);
      boolean sequence = meld.type() == Meld.Type.CHI;
      boolean outside = sequence ? base % 9 == 0 || base % 9 == 6 : Tile.isTerminalOrHonor(base);
      allMeldsAreChanta &= outside;
      allMeldsAreJunchan &= outside && !honor;
      allMeldsAreHonroutou &= outside && !sequence;
      allFixedTilesAreTanyao &= !outside;
      // 一面子の全牌は同じ色。刻子/槓子を3～4回数える必要はない。
      if (honor) chinitsuSuitMask = 0;
      else {
        int suit = 1 << Tile.suitOf(base);
        honitsuSuitMask &= suit;
        chinitsuSuitMask &= suit;
      }
    }

    long routes = 0L;
    routes |=
        yakuhaiRouteViable(Tile.HAKU, tripletTileMask, hand, visibleTileCounts)
            ? routeBit(Route.YAKUHAI_HAKU)
            : 0L;
    routes |=
        yakuhaiRouteViable(Tile.HATSU, tripletTileMask, hand, visibleTileCounts)
            ? routeBit(Route.YAKUHAI_HATSU)
            : 0L;
    routes |=
        yakuhaiRouteViable(Tile.CHUN, tripletTileMask, hand, visibleTileCounts)
            ? routeBit(Route.YAKUHAI_CHUN)
            : 0L;
    routes |=
        yakuhaiRouteViable(seatWind, tripletTileMask, hand, visibleTileCounts)
            ? routeBit(Route.YAKUHAI_SEAT_WIND)
            : 0L;
    routes |=
        yakuhaiRouteViable(roundWind, tripletTileMask, hand, visibleTileCounts)
            ? routeBit(Route.YAKUHAI_ROUND_WIND)
            : 0L;
    if (allFixedTilesAreTanyao) {
      routes |= routeBit(Route.TANYAO);
    }
    if (chiBaseTileMask == 0L) {
      routes |= routeBit(Route.TOITOI);
      if (hand.meldCount() == 4) {
        confirmedRoutes |= routeBit(Route.TOITOI);
      }
    }
    routes |= flushRouteBits(honitsuSuitMask, chinitsuSuitMask);
    boolean sequenceCanFit = chiBaseTileMask != 0L || hand.meldCount() < 4;
    if (allMeldsAreChanta && sequenceCanFit) {
      routes |= routeBit(Route.CHANTA);
    }
    if (allMeldsAreJunchan && sequenceCanFit) {
      routes |= routeBit(Route.JUNCHAN);
    }
    int remainingMeldSlots = 4 - hand.meldCount();
    int manSequenceStarts = sequenceStarts(chiBaseTileMask, Tile.SUIT_MAN);
    int pinSequenceStarts = sequenceStarts(chiBaseTileMask, Tile.SUIT_PIN);
    int souSequenceStarts = sequenceStarts(chiBaseTileMask, Tile.SUIT_SOU);
    routes |=
        ittsuRouteBits(manSequenceStarts, pinSequenceStarts, souSequenceStarts, remainingMeldSlots);
    confirmedRoutes |=
        confirmedSequenceRouteBits(manSequenceStarts, pinSequenceStarts, souSequenceStarts);
    if (sanshokuCanFit(
        manSequenceStarts, pinSequenceStarts, souSequenceStarts, remainingMeldSlots)) {
      routes |= routeBit(Route.SANSHOKU);
    }
    if (allMeldsAreHonroutou) {
      routes |= routeBit(Route.HONROUTOU);
    }
    return routes | confirmedRoutes << 32;
  }

  private static long confirmedYakuhaiBits(int tileType, int seatWind, int roundWind) {
    long routes = 0L;
    routes |= tileType == Tile.HAKU ? routeBit(Route.YAKUHAI_HAKU) : 0L;
    routes |= tileType == Tile.HATSU ? routeBit(Route.YAKUHAI_HATSU) : 0L;
    routes |= tileType == Tile.CHUN ? routeBit(Route.YAKUHAI_CHUN) : 0L;
    routes |= tileType == seatWind ? routeBit(Route.YAKUHAI_SEAT_WIND) : 0L;
    routes |= tileType == roundWind ? routeBit(Route.YAKUHAI_ROUND_WIND) : 0L;
    return routes;
  }

  private static boolean yakuhaiRouteViable(
      int tileType, long tripletTileMask, HandView hand, int[] visibleTileCounts) {
    boolean completed = (tripletTileMask & (1L << tileType)) != 0L;
    int concealed = hand.count(tileType);
    int unseen = Math.max(0, Tile.TILES_PER_TYPE - visibleTileCounts[tileType] - concealed);
    return completed || hand.meldCount() < 4 && concealed + unseen >= 3;
  }

  private static long flushRouteBits(int honitsuSuitMask, int chinitsuSuitMask) {
    long routes = 0L;
    routes |= (honitsuSuitMask & 0b001) != 0 ? routeBit(Route.HONITSU_MAN) : 0L;
    routes |= (honitsuSuitMask & 0b010) != 0 ? routeBit(Route.HONITSU_PIN) : 0L;
    routes |= (honitsuSuitMask & 0b100) != 0 ? routeBit(Route.HONITSU_SOU) : 0L;
    routes |= (chinitsuSuitMask & 0b001) != 0 ? routeBit(Route.CHINITSU_MAN) : 0L;
    routes |= (chinitsuSuitMask & 0b010) != 0 ? routeBit(Route.CHINITSU_PIN) : 0L;
    routes |= (chinitsuSuitMask & 0b100) != 0 ? routeBit(Route.CHINITSU_SOU) : 0L;
    return routes;
  }

  private static long ittsuRouteBits(
      int manSequenceStarts, int pinSequenceStarts, int souSequenceStarts, int remainingMeldSlots) {
    long routes = 0L;
    routes |= ittsuCanFit(manSequenceStarts, remainingMeldSlots) ? routeBit(Route.ITTSU_MAN) : 0L;
    routes |= ittsuCanFit(pinSequenceStarts, remainingMeldSlots) ? routeBit(Route.ITTSU_PIN) : 0L;
    routes |= ittsuCanFit(souSequenceStarts, remainingMeldSlots) ? routeBit(Route.ITTSU_SOU) : 0L;
    return routes;
  }

  private static long confirmedSequenceRouteBits(
      int manSequenceStarts, int pinSequenceStarts, int souSequenceStarts) {
    long routes = 0L;
    routes |=
        (manSequenceStarts & ITTSU_START_MASK) == ITTSU_START_MASK ? routeBit(Route.ITTSU_MAN) : 0L;
    routes |=
        (pinSequenceStarts & ITTSU_START_MASK) == ITTSU_START_MASK ? routeBit(Route.ITTSU_PIN) : 0L;
    routes |=
        (souSequenceStarts & ITTSU_START_MASK) == ITTSU_START_MASK ? routeBit(Route.ITTSU_SOU) : 0L;
    routes |=
        (manSequenceStarts & pinSequenceStarts & souSequenceStarts) != 0
            ? routeBit(Route.SANSHOKU)
            : 0L;
    return routes;
  }

  private static long routeBit(Route route) {
    return 1L << route.ordinal();
  }

  private static boolean ittsuCanFit(int sequenceStarts, int remainingMeldSlots) {
    return 3 - Integer.bitCount(sequenceStarts & ITTSU_START_MASK) <= remainingMeldSlots;
  }

  private static boolean sanshokuCanFit(
      int manSequenceStarts, int pinSequenceStarts, int souSequenceStarts, int remainingMeldSlots) {
    return switch (remainingMeldSlots) {
      case 0 -> (manSequenceStarts & pinSequenceStarts & souSequenceStarts) != 0;
      case 1 ->
          ((manSequenceStarts & pinSequenceStarts)
                  | (manSequenceStarts & souSequenceStarts)
                  | (pinSequenceStarts & souSequenceStarts))
              != 0;
      case 2 -> (manSequenceStarts | pinSequenceStarts | souSequenceStarts) != 0;
      default -> true;
    };
  }

  private static int sequenceStarts(long chiBaseTileMask, int suit) {
    return (int) (chiBaseTileMask >>> (suit * 9)) & SEQUENCE_START_MASK;
  }
}
