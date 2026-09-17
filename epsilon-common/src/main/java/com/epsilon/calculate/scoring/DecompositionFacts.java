package com.epsilon.calculate.scoring;

import com.epsilon.calculate.shape.WinningDecompositionCursor;
import com.epsilon.core.Tile;

/** 一つの和了形を一度だけ展開して得る、役・符・待ち形の計算用特徴。 */
final class DecompositionFacts {

  int jantouTile;
  int pairFu;
  boolean pairDragon;
  int sequenceMask;
  long tripletMask;
  int ittsuMask;
  int totalSequenceCount;
  int ankoCount;
  int peikouCount;
  boolean everyMentsuContainsTerminalOrHonor;
  boolean everyMentsuContainsTerminal;
  int machiTypeMask;
  int ankoFu;

  int load(WinningHandFacts hand, WinningDecompositionCursor pattern, int agariTile) {
    jantouTile = pattern.jantouTile();
    pairDragon = Tile.isDragon(jantouTile);
    pairFu =
        (jantouTile == hand.context.jikaze() ? 2 : 0)
            + (jantouTile == hand.context.bakaze() ? 2 : 0)
            + (pairDragon ? 2 : 0);
    sequenceMask = hand.sequenceMask | pattern.concealedShuntsuMask();
    tripletMask = hand.tripletMask | pattern.ankoTileMask();
    ittsuMask = hand.ittsuMask | pattern.concealedIttsuMask();
    totalSequenceCount = hand.declaredSequenceCount + pattern.concealedShuntsuCount();
    ankoCount = hand.ankoCount + pattern.ankoCount();
    peikouCount = pattern.peikouCount();
    everyMentsuContainsTerminalOrHonor =
        hand.everyMentsuContainsTerminalOrHonor && pattern.everyMentsuContainsTerminalOrHonor();
    everyMentsuContainsTerminal =
        hand.everyMentsuContainsTerminal && pattern.everyMentsuContainsTerminal();
    machiTypeMask = pattern.machiTypeMask(agariTile);
    ankoFu = pattern.ankoFu();
    return machiTypeMask;
  }

  boolean pairIsValueless() {
    return pairFu == 0;
  }

  boolean hasTriplet(int tile) {
    return (tripletMask & 1L << tile) != 0L;
  }

  int dragonTripletCount() {
    long dragons = 1L << Tile.HAKU | 1L << Tile.HATSU | 1L << Tile.CHUN;
    return Long.bitCount(tripletMask & dragons);
  }

  boolean hasSanshokuDoujun() {
    return (sequenceMask & sequenceMask >>> 7 & sequenceMask >>> 14 & 0x7f) != 0;
  }

  boolean hasSanshokuDoukou() {
    return (tripletMask & tripletMask >>> 9 & tripletMask >>> 18 & 0x1ffL) != 0L;
  }

  boolean hasIttsu() {
    for (int suit = 0; suit < 3; suit++) {
      int bits = 7 << (suit * 3);
      if ((ittsuMask & bits) == bits) return true;
    }
    return false;
  }
}
