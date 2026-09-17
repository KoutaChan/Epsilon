package com.epsilon.calculate.shape;

import com.epsilon.calculate.table.ShantenTable;
import com.epsilon.core.Tile;

/**
 * 通常形・七対子・国士の向聴、完成形、待ち、受け入れを一つの入口で解析する。
 *
 * <p>ワーカーごとに所有する。再利用領域は外へ公開せず、入力を変更しない。牌種の形状と公開残枚数は独立して扱い、役・赤牌・局況には依存しない。各色の参照は候補間で再利用する。
 */
public final class HandShapeAnalyzer {
  private static final long ALL_TILES = (1L << Tile.NUM_TILE_TYPES) - 1;
  private static final long ORPHANS = Tile.TERMINAL_OR_HONOR_TYPE_MASK;
  private static final long SEQUENCE_STARTS = 0x7fL | (0x7fL << 9) | (0x7fL << 18);
  private final HandShapeState arrayInput = new HandShapeState();
  private final HandShapeState lookupShape = new HandShapeState();
  private final HandShapeState removal = new HandShapeState();
  private final DrawAnalysis draw = new DrawAnalysis();

  public HandShapeAnalyzer() {}

  /** 0～14枚相当の形状を解析する。完成手では次のツモによる受け入れを返さない。 */
  public void analyzeInto(HandShapeView hand, HandShapeAnalysisBuffer destination) {
    validateShape(hand);
    draw.load(prepare(hand));
    int standard = draw.standardShanten();
    boolean special = draw.special;
    int pairs = special ? sevenPairs(draw.distinct, draw.pairs) : 0;
    int orphans = special ? thirteenOrphans(draw.orphans, draw.orphanPair) : 0;
    int minimum = special ? Math.min(standard, Math.min(pairs, orphans)) : standard;
    draw.prepareImprovements(minimum, standard, pairs, orphans);
    long improving = hand.effectiveTileCount() == 14 ? 0L : collectImprovingTiles(hand);
    destination.minimumShanten = minimum;
    destination.standardShanten = standard;
    destination.chiitoitsuShanten = pairs;
    destination.kokushiShanten = orphans;
    destination.specialHandsAvailable = special;
    destination.ukeireTileTypeMask = improving;
    destination.agariTileTypeMask =
        minimum == 0 && hand.effectiveTileCount() == 13 ? improving : 0L;
  }

  /** 確定面子数から残りの面子数を決める。短い手牌に架空の副露を補わない。 */
  public int calculateMinimum(HandShapeView hand) {
    validateShape(hand);
    int standard = standardShanten(hand);
    return hand.meldCount() == 0
        ? Math.min(
            standard,
            Math.min(
                sevenPairs(
                    hand.distinctConcealedTileTypeCount(), hand.concealedPairTileTypeCount()),
                thirteenOrphans(hand.kokushiTileTypeCount(), hand.kokushiPairTileTypeCount() > 0)))
        : standard;
  }

  public int calculateMinimum(int[] counts, int meldCount) {
    arrayInput.load(counts, meldCount);
    return calculateMinimum(arrayInput);
  }

  public int calculateStandard(HandShapeView hand) {
    validateShape(hand);
    return standardShanten(hand);
  }

  /** 第2引数は確定面子数。残りの面子数ではない。 */
  public int calculateStandard(int[] counts, int meldCount) {
    arrayInput.load(counts, meldCount);
    return standardShanten(arrayInput);
  }

  public int calculateChiitoitsu(HandShapeView hand) {
    validateShape(hand);
    return hand.meldCount() == 0
        ? sevenPairs(hand.distinctConcealedTileTypeCount(), hand.concealedPairTileTypeCount())
        : Integer.MAX_VALUE;
  }

  public int calculateChiitoitsu(int[] counts) {
    arrayInput.load(counts, 0);
    return calculateChiitoitsu(arrayInput);
  }

  public int calculateKokushi(HandShapeView hand) {
    validateShape(hand);
    return hand.meldCount() == 0
        ? thirteenOrphans(hand.kokushiTileTypeCount(), hand.kokushiPairTileTypeCount() > 0)
        : Integer.MAX_VALUE;
  }

  public int calculateKokushi(int[] counts) {
    arrayInput.load(counts, 0);
    return calculateKokushi(arrayInput);
  }

  public int calculateAfterAdding(HandShapeView hand, int tileType) {
    validateShape(hand);
    Tile.requireValidType(tileType, "tileType");
    if (hand.effectiveTileCount() >= 14 || hand.count(tileType) >= Tile.TILES_PER_TYPE) {
      throw new IllegalArgumentException("draw exceeds the hand or tile capacity");
    }
    draw.load(prepare(hand));
    return draw.afterAdding(tileType, hand.count(tileType));
  }

  public int calculateMinimumAfterRemoving(HandShapeView hand, int tileType) {
    validateShape(hand);
    removal.load(hand);
    removal.remove(tileType);
    return calculateMinimum(removal);
  }

  /** 和了形かを判定する。和了形でなければfalseを返し、牌の枚数が不正なら例外を送出する。 */
  public boolean isAgari(HandShapeView hand) {
    requireEffectiveTiles(hand, 14);
    return calculateMinimum(hand) == -1;
  }

  public boolean isAgari(int[] counts) {
    arrayInput.load(counts, 0);
    return isAgari(arrayInput);
  }

  public boolean isStandardAgari(HandShapeView hand) {
    requireEffectiveTiles(hand, 14);
    return standardShanten(hand) == -1;
  }

  public boolean isStandardAgari(int[] counts) {
    arrayInput.load(counts, 0);
    return isStandardAgari(arrayInput);
  }

  public boolean isChiitoitsu(int[] counts) {
    arrayInput.load(counts, 0);
    requireEffectiveTiles(arrayInput, 14);
    return calculateChiitoitsu(arrayInput) == -1;
  }

  public boolean isKokushi(int[] counts) {
    arrayInput.load(counts, 0);
    requireEffectiveTiles(arrayInput, 14);
    return calculateKokushi(arrayInput) == -1;
  }

  /** 役・フリテン・残枚数によらない、13枚相当の形上の待ち。 */
  public long agariTileTypeMask(HandShapeView hand) {
    requireEffectiveTiles(hand, 13);
    draw.load(prepare(hand));
    draw.prepareImprovements(0);
    return collectImprovingTiles(hand);
  }

  public long agariTileTypeMask(int[] counts, int meldCount) {
    arrayInput.load(counts, meldCount);
    return agariTileTypeMask(arrayInput);
  }

  /** 暗槓前後の待ち比較などに使う。元の手牌もその物理牌情報も変更しない。 */
  public long agariTileTypeMaskAfterRemoving(
      HandShapeView hand, int tileType, int removeCount, int meldCountAfterRemoval) {
    validateShape(hand);
    Tile.requireValidType(tileType, "tileType");
    if (removeCount < 1 || removeCount > hand.count(tileType)) {
      throw new IllegalArgumentException("invalid removal count: " + removeCount);
    }
    removal.load(hand);
    removal.setMeldCount(meldCountAfterRemoval);
    for (int n = 0; n < removeCount; n++) removal.remove(tileType);
    return agariTileTypeMask(removal);
  }

  /** 残枚数を考慮しない改善牌のマスク。14枚相当ではゼロ。 */
  public long improvingTileTypeMask(HandShapeView hand, int currentShanten) {
    validateShape(hand);
    if (hand.effectiveTileCount() == 14) return 0L;
    draw.load(prepare(hand));
    draw.prepareImprovements(currentShanten);
    return collectImprovingTiles(hand);
  }

  /** 基準形と受け入れを同時解析するときは、既に読み込んだ色キーを使い回す。 */
  private long collectImprovingTiles(HandShapeView hand) {
    if (draw.allDrawsImprove) {
      long possible = ALL_TILES;
      for (long pairs = hand.concealedPairTileTypeMask(); pairs != 0L; pairs &= pairs - 1) {
        int tile = Long.numberOfTrailingZeros(pairs);
        if (hand.count(tile) == Tile.TILES_PER_TYPE) possible &= ~(1L << tile);
      }
      return possible;
    }
    long mask = draw.specialImprovingTiles;
    if (!draw.standardCompetitive) return mask;
    long candidates = ALL_TILES & ~mask;
    if (draw.improvementThreshold == 0) {
      candidates &= completionCandidates(hand.concealedTileTypeMask());
    }
    draw.prepareStandardDraws();
    for (; candidates != 0L; candidates &= candidates - 1) {
      int tile = Long.numberOfTrailingZeros(candidates);
      if (hand.count(tile) < Tile.TILES_PER_TYPE && draw.improvesStandard(tile)) mask |= 1L << tile;
    }
    return mask;
  }

  /** 和了牌は既存牌との対子・刻子か、同色の二牌との順子に入る。 */
  private static long completionCandidates(long tileTypes) {
    long low = (tileTypes >>> 1) & (tileTypes >>> 2) & SEQUENCE_STARTS;
    long middle = (tileTypes & (tileTypes >>> 2) & SEQUENCE_STARTS) << 1;
    long high = (tileTypes & (tileTypes >>> 1) & SEQUENCE_STARTS) << 2;
    return tileTypes | low | middle | high;
  }

  /** 公開済み牌を除いた改善牌種を昇順で書く。visibleCounts は自分の手牌（副露・暗槓を除く）を含めない。 */
  public int writeImprovingTileTypes(
      HandShapeView hand, int[] visibleCounts, int currentShanten, int[] destination) {
    requireCountArray(visibleCounts, true);
    long mask = improvingTileTypeMask(hand, currentShanten);
    int written = 0;
    for (; mask != 0L; mask &= mask - 1) {
      int tile = Long.numberOfTrailingZeros(mask);
      if (Tile.TILES_PER_TYPE - hand.count(tile) - (visibleCounts == null ? 0 : visibleCounts[tile])
          > 0) {
        destination[written++] = tile;
      }
    }
    return written;
  }

  /** 呼び出し側で集計済みの未確認枚数から改善牌を集計する。 */
  public void summarizeImprovementsInto(
      HandShapeView hand,
      int[] remainingCounts,
      int currentShanten,
      ImprovingTilesBuffer destination) {
    requireCountArray(remainingCounts, false);
    long mask = improvingTileTypeMask(hand, currentShanten);
    long live = 0L;
    int count = 0;
    for (; mask != 0L; mask &= mask - 1) {
      int tile = Long.numberOfTrailingZeros(mask);
      if (remainingCounts[tile] > 0) {
        live |= 1L << tile;
        count += remainingCounts[tile];
      }
    }
    destination.tileTypeMask = live;
    destination.tileTypeCount = Long.bitCount(live);
    destination.remainingTileCount = count;
  }

  public int remainingImprovementTileCount(
      HandShapeView hand, int[] remainingCounts, int currentShanten) {
    requireCountArray(remainingCounts, false);
    long mask = improvingTileTypeMask(hand, currentShanten);
    int count = 0;
    for (; mask != 0L; mask &= mask - 1) {
      int tile = Long.numberOfTrailingZeros(mask);
      count += Math.max(0, remainingCounts[tile]);
    }
    return count;
  }

  public static int countTiles(int[] counts) {
    if (counts.length != Tile.NUM_TILE_TYPES)
      throw new IllegalArgumentException("counts must contain 34 entries");
    int total = 0;
    for (int count : counts) {
      if (count < 0 || count > Tile.TILES_PER_TYPE)
        throw new IllegalArgumentException("invalid tile count: " + count);
      total += count;
    }
    return total;
  }

  private static void validateShape(HandShapeView hand) {
    if (hand.meldCount() < 0
        || hand.meldCount() > 4
        || hand.concealedTileCount() < 0
        || hand.effectiveTileCount() > 14) {
      throw new IllegalArgumentException("invalid hand shape size");
    }
  }

  private static void requireEffectiveTiles(HandShapeView hand, int expected) {
    validateShape(hand);
    if (hand.effectiveTileCount() != expected) {
      throw new IllegalArgumentException(
          "expected " + expected + " effective tiles, got " + hand.effectiveTileCount());
    }
  }

  private static void requireCountArray(int[] counts, boolean nullable) {
    if (nullable && counts == null) return;
    if (counts == null || counts.length != Tile.NUM_TILE_TYPES)
      throw new IllegalArgumentException("counts must contain 34 entries");
  }

  private HandShapeState prepare(HandShapeView hand) {
    hand.copyShapeInto(lookupShape);
    return lookupShape;
  }

  private int standardShanten(HandShapeView hand) {
    HandShapeState state = prepare(hand);
    return ShantenTable.standardShanten(
        ShantenTable.suitKeyUnchecked(state.manzuCountCode()),
        ShantenTable.suitKeyUnchecked(state.pinzuCountCode()),
        ShantenTable.suitKeyUnchecked(state.souzuCountCode()),
        ShantenTable.honorKeyUnchecked(state.honorCountCode()),
        4 - state.meldCount());
  }

  private static int sevenPairs(int distinct, int pairs) {
    return 6 - pairs + Math.max(0, 7 - distinct);
  }

  private static int thirteenOrphans(int distinct, boolean pair) {
    return 13 - distinct - (pair ? 1 : 0);
  }

  /** 一ツモ候補の基準キーを保持し、変更色だけを引き直す。 */
  private static final class DrawAnalysis {
    private int target;
    private boolean special;
    private int manCode, pinCode, souCode, honorCode;
    private int manKey, pinKey, souKey, honorKey;
    private int pinSouKey, manSouKey, manPinKey, mergedSuitKey;
    private int distinct, pairs, orphans;
    private boolean orphanPair;
    private int improvementThreshold;
    private boolean standardCompetitive;
    private boolean allDrawsImprove;
    private long tileTypes, pairTypes;
    private long specialImprovingTiles;

    void load(HandShapeState hand) {
      target = 4 - hand.meldCount();
      special = hand.meldCount() == 0;
      manCode = hand.manzuCountCode();
      pinCode = hand.pinzuCountCode();
      souCode = hand.souzuCountCode();
      honorCode = hand.honorCountCode();
      manKey = ShantenTable.suitKeyUnchecked(manCode);
      pinKey = ShantenTable.suitKeyUnchecked(pinCode);
      souKey = ShantenTable.suitKeyUnchecked(souCode);
      honorKey = ShantenTable.honorKeyUnchecked(honorCode);
      if (special) {
        tileTypes = hand.concealedTileTypeMask();
        pairTypes = hand.concealedPairTileTypeMask();
        distinct = hand.distinctConcealedTileTypeCount();
        pairs = hand.concealedPairTileTypeCount();
        orphans = hand.kokushiTileTypeCount();
        orphanPair = hand.kokushiPairTileTypeCount() != 0;
      }
    }

    int standardShanten() {
      return ShantenTable.standardShanten(manKey, pinKey, souKey, honorKey, target);
    }

    /** 一牌追加で各形の不足枚数が減るのは高々一枚。最小値より遠い形は探索しない。 */
    void prepareImprovements(int threshold) {
      prepareImprovements(
          threshold,
          standardShanten(),
          special ? sevenPairs(distinct, pairs) : 0,
          special ? thirteenOrphans(orphans, orphanPair) : 0);
    }

    void prepareImprovements(int threshold, int standard, int sevenPairs, int thirteenOrphans) {
      improvementThreshold = threshold;
      standardCompetitive = standard == threshold;
      allDrawsImprove =
          standard < threshold
              || special && (sevenPairs < threshold || thirteenOrphans < threshold);
      specialImprovingTiles = 0L;
      if (!special || allDrawsImprove) return;
      if (sevenPairs == threshold) {
        // 単独牌の対子化と、七種類未満のときの新しい牌種が一向聴だけ改善する。
        specialImprovingTiles = tileTypes & ~pairTypes;
        if (distinct < 7) specialImprovingTiles |= ALL_TILES & ~tileTypes;
      }
      if (thirteenOrphans == threshold) {
        // 雀頭があれば欠けた么九牌、なければ么九牌の追加が改善する。
        specialImprovingTiles |= orphanPair ? ORPHANS & ~tileTypes : ORPHANS;
      }
    }

    void prepareStandardDraws() {
      pinSouKey = ShantenTable.mergeTwoSuitKeys(pinKey, souKey);
      manSouKey = ShantenTable.mergeTwoSuitKeys(manKey, souKey);
      manPinKey = ShantenTable.mergeTwoSuitKeys(manKey, pinKey);
      mergedSuitKey = ShantenTable.mergeThirdSuitKey(manPinKey, souKey);
    }

    boolean improvesStandard(int tile) {
      int suits;
      int honor = honorKey;
      if (tile < Tile.P1) {
        suits =
            ShantenTable.mergeThirdSuitKey(
                pinSouKey, ShantenTable.suitKeyAfterAdding(manCode, tile));
      } else if (tile < Tile.S1) {
        suits =
            ShantenTable.mergeThirdSuitKey(
                manSouKey, ShantenTable.suitKeyAfterAdding(pinCode, tile - Tile.P1));
      } else if (tile < Tile.TON) {
        suits =
            ShantenTable.mergeThirdSuitKey(
                manPinKey, ShantenTable.suitKeyAfterAdding(souCode, tile - Tile.S1));
      } else {
        suits = mergedSuitKey;
        honor = ShantenTable.honorKeyAfterAdding(honorCode, tile - Tile.TON);
      }
      return ShantenTable.standardShantenFromMergedSuits(suits, honor, target)
          < improvementThreshold;
    }

    private int standardAfterAdding(int tile) {
      int man = manKey, pin = pinKey, sou = souKey, honor = honorKey;
      if (tile < Tile.P1) man = ShantenTable.suitKeyAfterAdding(manCode, tile);
      else if (tile < Tile.S1) pin = ShantenTable.suitKeyAfterAdding(pinCode, tile - Tile.P1);
      else if (tile < Tile.TON) sou = ShantenTable.suitKeyAfterAdding(souCode, tile - Tile.S1);
      else honor = ShantenTable.honorKeyAfterAdding(honorCode, tile - Tile.TON);
      return ShantenTable.standardShanten(man, pin, sou, honor, target);
    }

    int afterAdding(int tile, int previous) {
      int normal = standardAfterAdding(tile);
      if (!special) return normal;
      int chiitoitsu =
          sevenPairs(distinct + (previous == 0 ? 1 : 0), pairs + (previous == 1 ? 1 : 0));
      boolean orphan = (ORPHANS & 1L << tile) != 0;
      int kokushi =
          thirteenOrphans(
              orphans + (orphan && previous == 0 ? 1 : 0), orphanPair || orphan && previous == 1);
      return Math.min(normal, Math.min(chiitoitsu, kokushi));
    }
  }
}
