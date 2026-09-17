package com.epsilon.calculate.scoring;

import com.epsilon.core.Tile;

/** 手牌共通特徴と和了形候補から、成立する役のビットマスクを組み立てる内部評価器。 */
final class YakuRules {

  private YakuRules() {}

  static long evaluateHandYakuMask(WinningHandFacts hand) {
    long bits = YakuBits.when(ScoringYaku.TANYAO, hand.tanyao);
    bits |= YakuBits.when(ScoringYaku.HONROUTOU, hand.honroutou);
    if (hand.isChinitsu()) bits |= ScoringYaku.CHINITSU.bit();
    else if (hand.isHonitsu()) bits |= ScoringYaku.HONITSU.bit();
    bits |= YakuBits.when(ScoringYaku.SANKANTSU, hand.kanCount == 3);
    return bits;
  }

  static long evaluateContextYakuMask(boolean menzen, WinMethod method, WinConditions conditions) {
    long bits = 0L;
    if (menzen) {
      bits |=
          switch (conditions.riichiStatus()) {
            case NONE -> 0L;
            case RIICHI -> ScoringYaku.RIICHI.bit();
            case DOUBLE_RIICHI -> ScoringYaku.DOUBLE_RIICHI.bit();
          };
    }
    bits |=
        YakuBits.when(
            ScoringYaku.IPPATSU,
            menzen && conditions.riichiStatus() != RiichiState.NONE && conditions.ippatsu());
    bits |= YakuBits.when(ScoringYaku.MENZEN_TSUMO, menzen && method == WinMethod.TSUMO);
    bits |=
        YakuBits.when(
            ScoringYaku.RINSHAN,
            method == WinMethod.TSUMO && conditions.source() == WinSource.RINSHAN);
    bits |=
        YakuBits.when(
            ScoringYaku.CHANKAN,
            method == WinMethod.RON && conditions.source() == WinSource.CHANKAN);
    if (conditions.source() == WinSource.LAST_TILE) {
      bits |= method == WinMethod.TSUMO ? ScoringYaku.HAITEI.bit() : ScoringYaku.HOUTEI.bit();
    }
    return bits;
  }

  static long evaluateHandYakumanMask(WinningHandFacts hand) {
    // 国士の13種類を持つ完成形は、他の牌種・面子役満とは複合しない。
    // 天和・地和は局況側で判定し、呼び出し元がこのマスクと合成する。
    if (hand.thirteenOrphansComplete) return ScoringYaku.KOKUSHI.bit();
    long bits = YakuBits.when(ScoringYaku.CHUUREN, hand.menzen && isChuuren(hand));
    bits |= YakuBits.when(ScoringYaku.DAISANGEN, hand.hasDaisangen());
    bits |= YakuBits.when(ScoringYaku.TSUUIISOU, hand.isTsuuiisou());
    bits |= YakuBits.when(ScoringYaku.RYUUIISOU, hand.ryuuiisou);
    bits |= YakuBits.when(ScoringYaku.CHINROUTOU, hand.honroutou && !hand.hasHonor);
    int windTriplets = hand.windTriplets();
    if (windTriplets == 4) bits |= ScoringYaku.DAISUUSHII.bit();
    else if (windTriplets == 3 && hand.hasWindPair()) bits |= ScoringYaku.SHOUSUUSHII.bit();
    bits |= YakuBits.when(ScoringYaku.SUUKANTSU, hand.kanCount == 4);
    return bits;
  }

  static long evaluateContextYakumanMask(WinMethod method, WinConditions conditions) {
    if (method == WinMethod.TSUMO
        && conditions.uninterruptedFirstDraw()
        && conditions.source() != WinSource.RINSHAN) {
      return conditions.oya() ? ScoringYaku.TENHOU.bit() : ScoringYaku.CHIIHOU.bit();
    }
    return 0L;
  }

  static long evaluatePatternYakuMask(WinningHandFacts hand, DecompositionFacts candidate) {
    long bits =
        YakuBits.when(ScoringYaku.YAKUHAI_HAKU, candidate.hasTriplet(Tile.HAKU))
            | YakuBits.when(ScoringYaku.YAKUHAI_HATSU, candidate.hasTriplet(Tile.HATSU))
            | YakuBits.when(ScoringYaku.YAKUHAI_CHUN, candidate.hasTriplet(Tile.CHUN))
            | YakuBits.when(ScoringYaku.YAKUHAI_SEAT, candidate.hasTriplet(hand.context.jikaze()))
            | YakuBits.when(ScoringYaku.YAKUHAI_ROUND, candidate.hasTriplet(hand.context.bakaze()));
    if (hand.menzen) {
      if (candidate.peikouCount >= 2) bits |= ScoringYaku.RYANPEIKOU.bit();
      else if (candidate.peikouCount == 1) bits |= ScoringYaku.IPEIKOU.bit();
    }
    bits |= YakuBits.when(ScoringYaku.SANSHOKU, candidate.hasSanshokuDoujun());
    bits |= YakuBits.when(ScoringYaku.SANSHOKU_DOUKOU, candidate.hasSanshokuDoukou());
    bits |= YakuBits.when(ScoringYaku.ITTSU, candidate.hasIttsu());
    bits |= YakuBits.when(ScoringYaku.TOITOI, candidate.totalSequenceCount == 0);
    bits |=
        YakuBits.when(
            ScoringYaku.SHOUSANGEN, candidate.pairDragon && candidate.dragonTripletCount() == 2);
    if (candidate.totalSequenceCount != 0
        && candidate.everyMentsuContainsTerminalOrHonor
        && Tile.isTerminalOrHonor(candidate.jantouTile)) {
      bits |=
          candidate.everyMentsuContainsTerminal && Tile.isTerminal(candidate.jantouTile)
              ? ScoringYaku.JUNCHAN.bit()
              : ScoringYaku.CHANTA.bit();
    }
    return bits;
  }

  static long evaluateCandidateYakuMask(
      WinningHandFacts hand,
      DecompositionFacts candidate,
      WinMethod method,
      WaitShape shape,
      long handYakuMask,
      long handYakumanMask,
      long patternYakuMask) {
    long yakuman = handYakumanMask;
    boolean opensTriplet = method == WinMethod.RON && shape.opensTripletOnRon();
    yakuman |=
        YakuBits.when(
            ScoringYaku.SUUANKOU,
            hand.menzen
                && candidate.totalSequenceCount == 0
                && candidate.ankoCount == 4
                && !opensTriplet);
    if (yakuman != 0L) return yakuman;

    long placement =
        YakuBits.when(ScoringYaku.SANANKOU, candidate.ankoCount - (opensTriplet ? 1 : 0) >= 3)
            | YakuBits.when(
                ScoringYaku.PINFU,
                hand.menzen
                    && candidate.ankoCount == 0
                    && candidate.pairIsValueless()
                    && shape == WaitShape.RYANMEN);
    return YakuBits.effective(handYakuMask | patternYakuMask | placement, hand.menzen);
  }

  private static boolean isChuuren(WinningHandFacts hand) {
    if (!hand.isChinitsu()) return false;
    for (int suit = 0; suit < 3; suit++) {
      int base = suit * 9;
      if (hand.fullTileCount(base) == 0) continue;
      int total = 0;
      boolean valid = true;
      for (int number = 0; number < 9; number++) {
        int count = hand.fullTileCount(base + number);
        total += count;
        if ((number == 0 || number == 8) ? count < 3 : count < 1) {
          valid = false;
          break;
        }
      }
      if (valid && total == 14) return true;
    }
    return false;
  }
}
