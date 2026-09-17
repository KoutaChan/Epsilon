package com.epsilon.calculate.scoring;

import com.epsilon.calculate.shape.HandShapeState;
import com.epsilon.calculate.shape.HandShapeView;
import com.epsilon.core.DoraState;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;

/** 一回の和了評価で一度だけ構築する、手牌・副露・局況の内部処理用特徴。 */
final class WinningHandFacts {

  private static final long GREEN_TILES =
      1L << Tile.S2
          | 1L << Tile.S3
          | 1L << Tile.S4
          | 1L << Tile.S6
          | 1L << Tile.S8
          | 1L << Tile.HATSU;
  private static final long SHUNTSU_START_MASK = 0x7fL | 0x7fL << Tile.P1 | 0x7fL << Tile.S1;

  private final HandShapeState.TileCountBuffer fullCounts = new HandShapeState.TileCountBuffer();
  private long fullTileTypeMask;
  private long fixedTileTypeMask;
  private long fixedMeldSignature;
  private int fixedMeldCount = -1;
  int omoteDoraCount;
  int uraDoraCount;
  int fixedAkaMask;

  WinConditions context;
  boolean menzen;
  boolean sevenPairsComplete;
  boolean thirteenOrphansComplete;
  boolean standardPatternMayExist;
  int suitMask;
  boolean hasHonor;
  boolean tanyao;
  boolean honroutou;
  boolean ryuuiisou;
  int kanCount;
  int sequenceMask;
  long tripletMask;
  int ittsuMask;
  int declaredSequenceCount;
  int ankoCount;
  int declaredMeldFu;
  boolean everyMentsuContainsTerminalOrHonor;
  boolean everyMentsuContainsTerminal;

  WinningHandFacts load(
      HandShapeView shape,
      HandView hand,
      int addedAgariTile,
      WinConditions context,
      long doraTileTypesPacked,
      int doraTileTypeCount,
      long uraDoraTileTypesPacked,
      int uraKnownMask) {
    shape.copyTileCountsInto(fullCounts);
    long concealedMask = shape.concealedTileTypeMask();
    long pairMask = shape.concealedPairTileTypeMask();
    int concealedCount = shape.concealedTileCount();
    int kokushiTypes = shape.kokushiTileTypeCount();
    int kokushiPairs = shape.kokushiPairTileTypeCount();
    if (addedAgariTile >= 0) {
      int previous = shape.count(addedAgariTile);
      fullCounts.add(addedAgariTile);
      concealedCount++;
      long bit = 1L << addedAgariTile;
      if (previous == 0) {
        concealedMask |= bit;
        if (Tile.isTerminalOrHonor(addedAgariTile)) kokushiTypes++;
      } else if (previous == 1) {
        pairMask |= bit;
        if (Tile.isTerminalOrHonor(addedAgariTile)) kokushiPairs++;
      }
    }

    this.context = context;
    menzen = hand.isMenzen();
    sevenPairsComplete =
        menzen && hand.meldCount() == 0 && concealedCount == 14 && Long.bitCount(pairMask) == 7;
    thirteenOrphansComplete =
        menzen
            && hand.meldCount() == 0
            && concealedCount == 14
            && kokushiTypes == 13
            && kokushiPairs == 1
            && (concealedMask & ~Tile.TERMINAL_OR_HONOR_TYPE_MASK) == 0L;
    standardPatternMayExist = true;
    if (sevenPairsComplete) {
      // 七対子と通常形を兼ねるには、4順子を作る少なくとも2種類の順子開始位置が必要。
      long shuntsuStarts = pairMask & pairMask >>> 1 & pairMask >>> 2 & SHUNTSU_START_MASK;
      standardPatternMayExist = Long.bitCount(shuntsuStarts) >= 2;
    }

    int meldCount = hand.meldCount();
    long meldSignature = hand.canonicalMeldSignature();
    boolean fixedMeldsChanged = fixedMeldCount != meldCount || fixedMeldSignature != meldSignature;
    if (fixedMeldsChanged) {
      // 再構築の途中で例外になっても、不完全な固定特徴を次回に使わない。
      fixedMeldCount = -1;
      prepareFixedMeldFacts(hand, meldCount);
    }
    fullTileTypeMask = concealedMask | fixedTileTypeMask;
    // 固定特徴が同じでも、手牌（副露・暗槓を除く）との合計が4枚を超えないか毎回検証する。
    for (int meldIndex = 0; meldIndex < meldCount; meldIndex++) {
      Meld meld = hand.meld(meldIndex);
      for (int tileIndex = 0; tileIndex < meld.size(); tileIndex++) {
        fullCounts.add(meld.tileAt(tileIndex));
      }
    }

    suitMask = (fullTileTypeMask & 0x1ffL) != 0L ? 1 : 0;
    if ((fullTileTypeMask & (0x1ffL << 9)) != 0L) suitMask |= 2;
    if ((fullTileTypeMask & (0x1ffL << 18)) != 0L) suitMask |= 4;
    hasHonor = (fullTileTypeMask & (0x7fL << 27)) != 0L;
    tanyao = (fullTileTypeMask & Tile.TERMINAL_OR_HONOR_TYPE_MASK) == 0L;
    honroutou = (fullTileTypeMask & ~Tile.TERMINAL_OR_HONOR_TYPE_MASK) == 0L;
    ryuuiisou = (fullTileTypeMask & ~GREEN_TILES) == 0L;
    omoteDoraCount = countDora(doraTileTypesPacked, doraTileTypeCount, -1);
    uraDoraCount = countDora(uraDoraTileTypesPacked, doraTileTypeCount, uraKnownMask);
    if (fixedMeldsChanged) {
      fixedMeldSignature = meldSignature;
      fixedMeldCount = meldCount;
    }
    return this;
  }

  /** 確定済みの面子の構成だけに依存する特徴。局況・手牌（副露・暗槓を除く）・ドラは含めない。 */
  private void prepareFixedMeldFacts(HandView hand, int meldCount) {
    kanCount = hand.kanCount();
    fixedTileTypeMask = 0L;
    fixedAkaMask = 0;
    sequenceMask = 0;
    tripletMask = 0L;
    ittsuMask = 0;
    declaredSequenceCount = 0;
    ankoCount = 0;
    declaredMeldFu = 0;
    everyMentsuContainsTerminalOrHonor = true;
    everyMentsuContainsTerminal = true;

    for (int meldIndex = 0; meldIndex < meldCount; meldIndex++) {
      Meld meld = hand.meld(meldIndex);
      int tile = meld.baseTileType();
      fixedTileTypeMask |= meld.tileTypeMask();
      if (meld.akaSource() != Meld.AkaSource.NONE) fixedAkaMask |= 1 << (tile / 9);
      boolean terminalOrHonor = Tile.isTerminalOrHonor(tile);
      if (meld.type() == Meld.Type.CHI) {
        declaredSequenceCount++;
        sequenceMask |= 1 << sequenceMaskIndex(tile);
        ittsuMask |= ittsuBit(tile);
        int number = tile % 9;
        if (number != 0 && number != 6) {
          everyMentsuContainsTerminalOrHonor = false;
          everyMentsuContainsTerminal = false;
        }
      } else {
        tripletMask |= 1L << tile;
        if (meld.preservesMenzen()) ankoCount++;
        if (!terminalOrHonor) {
          everyMentsuContainsTerminalOrHonor = false;
          everyMentsuContainsTerminal = false;
        } else if (!Tile.isTerminal(tile)) {
          everyMentsuContainsTerminal = false;
        }
      }
      declaredMeldFu +=
          switch (meld.type()) {
            case CHI -> 0;
            case PON -> terminalOrHonor ? 4 : 2;
            case DAIMINKAN, KAKAN -> terminalOrHonor ? 16 : 8;
            case ANKAN -> terminalOrHonor ? 32 : 16;
          };
    }
  }

  int fullTileCount(int tileType) {
    return fullCounts.count(tileType);
  }

  boolean isHonitsu() {
    return Integer.bitCount(suitMask) == 1 && hasHonor;
  }

  boolean isChinitsu() {
    return Integer.bitCount(suitMask) == 1 && !hasHonor;
  }

  boolean isTsuuiisou() {
    return suitMask == 0;
  }

  boolean hasDaisangen() {
    return fullTileCount(Tile.HAKU) >= 3
        && fullTileCount(Tile.HATSU) >= 3
        && fullTileCount(Tile.CHUN) >= 3;
  }

  int windTriplets() {
    int count = 0;
    for (int tile = Tile.TON; tile <= Tile.PEI; tile++) if (fullTileCount(tile) >= 3) count++;
    return count;
  }

  boolean hasWindPair() {
    for (int tile = Tile.TON; tile <= Tile.PEI; tile++) if (fullTileCount(tile) == 2) return true;
    return false;
  }

  static int sequenceMaskIndex(int tile) {
    return (tile / 9) * 7 + tile % 9;
  }

  static int ittsuBit(int tile) {
    int number = tile % 9;
    int suit = tile / 9;
    return switch (number) {
      case 0 -> 1 << (suit * 3);
      case 3 -> 1 << (suit * 3 + 1);
      case 6 -> 1 << (suit * 3 + 2);
      default -> 0;
    };
  }

  private int countDora(long tileTypesPacked, int count, int knownMask) {
    int total = 0;
    for (int index = 0; index < count; index++) {
      if (knownMask < 0 || (knownMask & 1 << index) != 0) {
        total += fullTileCount(DoraState.packedTile(tileTypesPacked, index));
      }
    }
    return total;
  }
}
