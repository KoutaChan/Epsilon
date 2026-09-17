package com.epsilon.client.tenhou;

import com.epsilon.core.Meld;
import com.epsilon.core.Tile;

/**
 * 天鳳プロトコルの鳴きコード (m パラメータ) デコーダー。
 *
 * <p>ビットレイアウト（tenhou-python-bot 準拠）:
 *
 * <ul>
 *   <li>2 ビット目が立っている → チー
 *   <li>2 ビット目が立っておらず 3 ビット目が立っている → ポン
 *   <li>2,3 ビット目が立っておらず 4 ビット目が立っている → 加槓
 *   <li>2,3,4 ビット目が立っていない → 暗槓または大明槓
 * </ul>
 */
public final class TenhouMeldDecoder {

  private TenhouMeldDecoder() {}

  /**
   * デコード結果。
   *
   * @param meld epsilon Meld オブジェクト
   * @param consumedPhysicalTileIds 自手牌から消費された物理牌ID
   * @param calledPhysicalTileId 副露に含まれる元の他家打牌の物理牌ID。暗槓は -1
   */
  public record DecodedMeld(Meld meld, int[] consumedPhysicalTileIds, int calledPhysicalTileId) {}

  /**
   * m パラメータをデコードして Meld を生成。
   *
   * @param meldCode m 値
   * @return デコード結果
   */
  public static DecodedMeld decode(int meldCode) {
    if (meldCode < 0) {
      throw new IllegalArgumentException("meldCode must be non-negative");
    }
    if ((meldCode & (1 << 2)) != 0) {
      return decodeChi(meldCode);
    } else if ((meldCode & (1 << 3)) != 0) {
      return decodePon(meldCode);
    } else if ((meldCode & (1 << 4)) != 0) {
      return decodeKakan(meldCode);
    } else {
      return decodeKan(meldCode);
    }
  }

  // チー
  private static DecodedMeld decodeChi(int meldCode) {
    if ((meldCode & 0x3) != 3) {
      throw new IllegalArgumentException("chi must be called from kamicha");
    }
    // チーは常に上家 (fromPlayer=3)
    int firstTileCopyIndex = (meldCode >> 3) & 0x3;
    int secondTileCopyIndex = (meldCode >> 5) & 0x3;
    int thirdTileCopyIndex = (meldCode >> 7) & 0x3;
    int encodedSequenceAndCalledPosition = (meldCode >> 10) & 0x3F;
    int encodedSequenceIndex = encodedSequenceAndCalledPosition / 3;
    int calledPosition = encodedSequenceAndCalledPosition % 3;

    int suitIndex = encodedSequenceIndex / 7;
    int sequenceStartNumber = encodedSequenceIndex % 7;
    int firstTileType = suitIndex * 9 + sequenceStartNumber;
    int secondTileType = firstTileType + 1;
    int thirdTileType = firstTileType + 2;

    int firstPhysicalTileId = firstTileType * 4 + firstTileCopyIndex;
    int secondPhysicalTileId = secondTileType * 4 + secondTileCopyIndex;
    int thirdPhysicalTileId = thirdTileType * 4 + thirdTileCopyIndex;

    int[] consumedPhysicalTileIds;
    int calledTileType;
    int calledPhysicalTileId;
    if (calledPosition == 0) {
      calledTileType = firstTileType;
      calledPhysicalTileId = firstPhysicalTileId;
      consumedPhysicalTileIds = new int[] {secondPhysicalTileId, thirdPhysicalTileId};
    } else if (calledPosition == 1) {
      calledTileType = secondTileType;
      calledPhysicalTileId = secondPhysicalTileId;
      consumedPhysicalTileIds = new int[] {firstPhysicalTileId, thirdPhysicalTileId};
    } else {
      calledTileType = thirdTileType;
      calledPhysicalTileId = thirdPhysicalTileId;
      consumedPhysicalTileIds = new int[] {firstPhysicalTileId, secondPhysicalTileId};
    }

    Meld meld =
        Meld.chi(
            firstTileType,
            calledTileType,
            Meld.AkaSource.forCall(
                Tile.isAka(calledPhysicalTileId), containsAkaTile(consumedPhysicalTileIds)));

    return new DecodedMeld(meld, consumedPhysicalTileIds, calledPhysicalTileId);
  }

  // ポン
  private static DecodedMeld decodePon(int meldCode) {
    int unusedCopyIndex = (meldCode >> 5) & 0x3;
    int encodedTileAndCalledIndex = (meldCode >> 9) & 0x7F;
    int tileType = encodedTileAndCalledIndex / 3;
    int calledMeldTileIndex = encodedTileAndCalledIndex % 3;

    int sourcePlayerOffset = meldCode & 0x3;

    // 4枚の実牌IDのうち unusedCopyIndex を除外した3枚がポン
    int[] allPhysicalTileIds = new int[4];
    for (int copyIndex = 0; copyIndex < 4; copyIndex++) {
      allPhysicalTileIds[copyIndex] = tileType * 4 + copyIndex;
    }

    // unusedCopyIndex は鳴いていない牌を示す
    int[] ponPhysicalTileIds = new int[3];
    int ponTileIndex = 0;
    for (int copyIndex = 0; copyIndex < 4; copyIndex++) {
      if (copyIndex == unusedCopyIndex) {
        continue;
      }
      ponPhysicalTileIds[ponTileIndex++] = allPhysicalTileIds[copyIndex];
    }

    // 鳴いた牌 (from のプレイヤーが切った牌) を特定
    // calledMeldTileIndex で鳴いた牌のインデックスを特定
    int calledPhysicalTileId = ponPhysicalTileIds[calledMeldTileIndex];
    int[] consumedPhysicalTileIds = new int[2];
    int consumedTileIndex = 0;
    for (int meldTileIndex = 0; meldTileIndex < 3; meldTileIndex++) {
      if (meldTileIndex != calledMeldTileIndex) {
        consumedPhysicalTileIds[consumedTileIndex++] = ponPhysicalTileIds[meldTileIndex];
      }
    }

    Meld meld =
        Meld.pon(
            tileType,
            Meld.RelativeSource.fromPlayerOffset(sourcePlayerOffset),
            Meld.AkaSource.forCall(
                Tile.isAka(calledPhysicalTileId), containsAkaTile(consumedPhysicalTileIds)));
    return new DecodedMeld(meld, consumedPhysicalTileIds, calledPhysicalTileId);
  }

  // 加槓
  private static DecodedMeld decodeKakan(int meldCode) {
    int addedTileCopyIndex = (meldCode >> 5) & 0x3;
    int encodedTileAndCalledIndex = (meldCode >> 9) & 0x7F;
    int tileType = encodedTileAndCalledIndex / 3;
    int calledMeldTileIndex = encodedTileAndCalledIndex % 3;
    int sourcePlayerOffset = meldCode & 0x3;

    // 加槓で追加された牌の実牌ID
    int addedPhysicalTileId = tileType * 4 + addedTileCopyIndex;
    int calledPhysicalTileId =
        calledPhysicalTileIdExcluding(tileType, addedTileCopyIndex, calledMeldTileIndex);
    boolean ponContainsConsumedHandAkaTile = false;
    for (int copyIndex = 0; copyIndex < Tile.TILES_PER_TYPE; copyIndex++) {
      int physicalTileId = tileType * Tile.TILES_PER_TYPE + copyIndex;
      if (physicalTileId != addedPhysicalTileId
          && physicalTileId != calledPhysicalTileId
          && Tile.isAka(physicalTileId)) {
        ponContainsConsumedHandAkaTile = true;
      }
    }

    Meld basePon =
        Meld.pon(
            tileType,
            Meld.RelativeSource.fromPlayerOffset(sourcePlayerOffset),
            Meld.AkaSource.forCall(
                Tile.isAka(calledPhysicalTileId), ponContainsConsumedHandAkaTile));
    Meld meld = Meld.kakan(basePon, Tile.isAka(addedPhysicalTileId));
    return new DecodedMeld(meld, new int[] {addedPhysicalTileId}, calledPhysicalTileId);
  }

  // 暗槓 / 大明槓
  private static DecodedMeld decodeKan(int meldCode) {
    int encodedPhysicalTileId = (meldCode >> 8) & 0xFF;
    int tileType = encodedPhysicalTileId / Tile.TILES_PER_TYPE;
    int sourcePlayerOffset = meldCode & 0x3;

    if (sourcePlayerOffset == 0) {
      // 暗槓
      int[] consumedPhysicalTileIds =
          new int[] {tileType * 4, tileType * 4 + 1, tileType * 4 + 2, tileType * 4 + 3};
      Meld meld = Meld.ankan(tileType, containsAkaTile(consumedPhysicalTileIds));
      return new DecodedMeld(meld, consumedPhysicalTileIds, -1);
    } else {
      // consumedPhysicalTileIds は自手牌からの3枚 (鳴き牌を除外)
      int calledPhysicalTileId = encodedPhysicalTileId;
      int[] consumedPhysicalTileIds = new int[3];
      int consumedTileIndex = 0;
      for (int copyIndex = 0; copyIndex < 4; copyIndex++) {
        int physicalTileId = tileType * 4 + copyIndex;
        if (physicalTileId != calledPhysicalTileId) {
          consumedPhysicalTileIds[consumedTileIndex++] = physicalTileId;
        }
      }
      Meld meld =
          Meld.daiminkan(
              tileType,
              Meld.RelativeSource.fromPlayerOffset(sourcePlayerOffset),
              Meld.AkaSource.forCall(
                  Tile.isAka(calledPhysicalTileId), containsAkaTile(consumedPhysicalTileIds)));
      return new DecodedMeld(meld, consumedPhysicalTileIds, calledPhysicalTileId);
    }
  }

  private static boolean containsAkaTile(int[] physicalTileIds) {
    for (int physicalTileId : physicalTileIds) {
      if (Tile.isAka(physicalTileId)) {
        return true;
      }
    }
    return false;
  }

  private static int calledPhysicalTileIdExcluding(
      int tileType, int excludedCopyIndex, int calledMeldTileIndex) {
    int calledCopyIndex =
        calledMeldTileIndex < excludedCopyIndex ? calledMeldTileIndex : calledMeldTileIndex + 1;
    return tileType * Tile.TILES_PER_TYPE + calledCopyIndex;
  }
}
