package com.epsilon.calculate.shape;

import com.epsilon.calculate.table.DecompositionTable;
import com.epsilon.core.Tile;
import java.util.Objects;

/**
 * 通常和了の面子分解を、参照テーブル上の直積として列挙する再利用可能カーソル。
 *
 * <p>入力手牌を変更せず、コールバック・分解オブジェクト・コレクションを生成しない。インスタンスは変更可能で複数スレッドから同時に利用できないため、 エンジン
 * ワーカーごとに一つ所有して繰り返し利用する。
 *
 * <p>萬子・筒子・索子・字牌それぞれの局部分解集合を {@code Dm, Dp, Ds, Dz} とすると、列挙対象は {@code Dm x Dp x Ds x Dz}
 * である。通常和了として有効なのは、4成分のうち雀頭を持つ成分がちょうど一つの組合せだけである。
 */
public final class WinningDecompositionCursor {

  private static final int GROUP_BITS = 7;
  private static final int GROUP_MASK = 0x7f;
  private static final int TILE_MASK = 0x3f;
  private static final int SEQUENCE_FLAG = 1 << 6;
  private static final int GROUPS_SHIFT = 9;
  private static final int[] BASE5_POWERS = {1, 5, 25, 125, 625, 3_125, 15_625, 78_125, 390_625};

  private final HandShapeState shape = new HandShapeState();
  private int manIndex;
  private int manEnd;
  private int pinIndex;
  private int pinStart;
  private int pinEnd;
  private int souIndex;
  private int souStart;
  private int souEnd;
  private int honorIndex;
  private int honorStart;
  private int honorEnd;
  private int currentManIndex;
  private int currentPinIndex;
  private int currentSouIndex;
  private int currentHonorIndex;
  private int currentMan;
  private int currentPin;
  private int currentSou;
  private int currentHonor;
  private long manFeatures;
  private long pinFeatures;
  private long souFeatures;
  private long honorFeatures;
  private int jantouTile;
  private boolean active;
  private boolean packedPatternReady;
  private long packedPattern;

  /** 入力を変更せず、現在の完成手にカーソルを結び付ける。 */
  public WinningDecompositionCursor bindCompleted(HandShapeView hand) {
    Objects.requireNonNull(hand, "hand").copyShapeInto(shape);
    validateSize(hand.concealedTileCount(), hand.meldCount(), 14);
    return bind(
        shape.manzuCountCode(),
        shape.pinzuCountCode(),
        shape.souzuCountCode(),
        shape.honorCountCode());
  }

  /** 入力を変更せず、指定した和了牌を仮想的に一枚加えた手牌へカーソルを結び付ける。 */
  public WinningDecompositionCursor bindAfterAdding(HandShapeView hand, int tileType) {
    Objects.requireNonNull(hand, "hand").copyShapeInto(shape);
    if (tileType < 0 || tileType >= Tile.NUM_TILE_TYPES) {
      throw new IllegalArgumentException("invalid tile type: " + tileType);
    }
    if (hand.count(tileType) >= Tile.TILES_PER_TYPE) {
      throw new IllegalArgumentException(
          "cannot add a fifth tile to completed-hand analysis: " + Tile.name(tileType));
    }
    validateSize(hand.concealedTileCount(), hand.meldCount(), 13);
    int man = shape.manzuCountCode();
    int pin = shape.pinzuCountCode();
    int sou = shape.souzuCountCode();
    int honor = shape.honorCountCode();
    int power = BASE5_POWERS[tileType % 9];
    switch (tileType / 9) {
      case 0 -> man += power;
      case 1 -> pin += power;
      case 2 -> sou += power;
      case 3 -> honor += power;
      default -> throw new IllegalArgumentException("invalid tile type: " + tileType);
    }
    return bind(man, pin, sou, honor);
  }

  /** テストや外部からの検証で、34牌種ごとの枚数を配列として受け取る。 */
  public WinningDecompositionCursor bind(int[] tileCounts) {
    return bind(tileCounts, 0);
  }

  /** 確定面子を除いた手牌（副露・暗槓を除く）を、明示的な確定面子数とともに受け取る。 */
  public WinningDecompositionCursor bind(int[] tileCounts, int fixedMeldCount) {
    Objects.requireNonNull(tileCounts, "tileCounts");
    if (tileCounts.length != Tile.NUM_TILE_TYPES) {
      throw new IllegalArgumentException(
          "tileCounts length must be " + Tile.NUM_TILE_TYPES + ": " + tileCounts.length);
    }
    int total = 0;
    for (int count : tileCounts) total += count;
    validateSize(total, fixedMeldCount, 14);
    return bind(
        DecompositionTable.suitCountCode(tileCounts, Tile.M1),
        DecompositionTable.suitCountCode(tileCounts, Tile.P1),
        DecompositionTable.suitCountCode(tileCounts, Tile.S1),
        DecompositionTable.honorCountCode(tileCounts));
  }

  private static void validateSize(int concealed, int fixedMelds, int expected) {
    if (fixedMelds < 0 || fixedMelds > 4 || concealed + fixedMelds * 3 != expected)
      throw new IllegalArgumentException("Expected " + expected + " tiles including fixed melds");
  }

  private WinningDecompositionCursor bind(int manCode, int pinCode, int souCode, int honorCode) {
    int manRange = DecompositionTable.suitPatternRange(manCode);
    int pinRange = DecompositionTable.suitPatternRange(pinCode);
    int souRange = DecompositionTable.suitPatternRange(souCode);
    int honorRange = DecompositionTable.honorPatternRange(honorCode);
    int manCount = DecompositionTable.patternCount(manRange);
    int pinCount = DecompositionTable.patternCount(pinRange);
    int souCount = DecompositionTable.patternCount(souRange);
    int honorCount = DecompositionTable.patternCount(honorRange);
    manIndex = DecompositionTable.patternStart(manRange);
    manEnd = manIndex + manCount;
    pinStart = DecompositionTable.patternStart(pinRange);
    pinIndex = pinStart;
    pinEnd = pinStart + pinCount;
    souStart = DecompositionTable.patternStart(souRange);
    souIndex = souStart;
    souEnd = souStart + souCount;
    honorStart = DecompositionTable.patternStart(honorRange);
    honorIndex = honorStart;
    honorEnd = honorStart + honorCount;
    // 各領域の最上位ビットを別位置へずらし、雀頭を持つ成分数を一度のbitCountで得る。
    int pairComponents = manRange >>> 31 | pinRange >>> 30 | souRange >>> 29 | honorRange >>> 28;
    active =
        manCount != 0
            && pinCount != 0
            && souCount != 0
            && honorCount != 0
            && Integer.bitCount(pairComponents) == 1;
    // 全入力で14枚相当を検証済み。各局部分解の牌数と雀頭1個から面子数も一意に決まる。
    jantouTile = 0;
    packedPattern = 0L;
    packedPatternReady = true;
    return this;
  }

  public boolean next() {
    if (!active) return false;
    currentManIndex = manIndex;
    currentPinIndex = pinIndex;
    currentSouIndex = souIndex;
    currentHonorIndex = honorIndex;
    currentMan = DecompositionTable.suitPattern(manIndex);
    currentPin = DecompositionTable.suitPattern(pinIndex);
    currentSou = DecompositionTable.suitPattern(souIndex);
    currentHonor = DecompositionTable.honorPattern(honorIndex);
    manFeatures = DecompositionTable.suitFeaturePacket(manIndex);
    pinFeatures = DecompositionTable.suitFeaturePacket(pinIndex);
    souFeatures = DecompositionTable.suitFeaturePacket(souIndex);
    honorFeatures = DecompositionTable.honorFeaturePacket(honorIndex);
    jantouTile =
        WinningDecompositionCursor.localJantouTile(
            currentMan, currentPin, currentSou, currentHonor);
    packedPatternReady = false;
    advanceCombination();
    return true;
  }

  public long packedPattern() {
    if (!packedPatternReady) {
      packedPattern = pack(jantouTile, currentMan, currentPin, currentSou, currentHonor);
      packedPatternReady = true;
    }
    return packedPattern;
  }

  public int jantouTile() {
    return jantouTile;
  }

  public int mentsuCount() {
    return WinningDecompositionCursor.mentsuCount(packedPattern());
  }

  public int mentsuTile(int mentsuIndex) {
    return WinningDecompositionCursor.mentsuTile(packedPattern(), mentsuIndex);
  }

  public boolean isShuntsu(int mentsuIndex) {
    return WinningDecompositionCursor.isShuntsu(packedPattern(), mentsuIndex);
  }

  /** 現在の分解で和了牌が取り得る待ち形を、{@code WaitShape} 序数互換のビット集合で返す。 */
  public int machiTypeMask(int agariTile) {
    long matrix =
        switch (agariTile / 9) {
          case 0 -> DecompositionTable.suitMachiTypeMatrix(currentManIndex);
          case 1 -> DecompositionTable.suitMachiTypeMatrix(currentPinIndex);
          case 2 -> DecompositionTable.suitMachiTypeMatrix(currentSouIndex);
          case 3 -> DecompositionTable.honorMachiTypeMatrix(currentHonorIndex);
          default -> throw new IllegalArgumentException("invalid tile type: " + agariTile);
        };
    // 45ビット = 9種類の数字 × 5種類の待ち形。各数字に対応する5ビットを取り出す。
    return (int) (matrix >>> (agariTile % 9 * 5)) & 0x1f;
  }

  public int concealedShuntsuMask() {
    return (int) (manFeatures & 0x7f)
        | (int) (pinFeatures & 0x7f) << 7
        | (int) (souFeatures & 0x7f) << 14;
  }

  public long ankoTileMask() {
    return manFeatures >>> 7 & 0x1ffL
        | (pinFeatures >>> 7 & 0x1ffL) << 9
        | (souFeatures >>> 7 & 0x1ffL) << 18
        | (honorFeatures >>> 7 & 0x7fL) << 27;
  }

  public int concealedIttsuMask() {
    return (int) (manFeatures >>> 16 & 7)
        | (int) (pinFeatures >>> 16 & 7) << 3
        | (int) (souFeatures >>> 16 & 7) << 6;
  }

  public int concealedShuntsuCount() {
    return featureSum(19, 7);
  }

  public int ankoCount() {
    return featureSum(22, 7);
  }

  public int peikouCount() {
    return featureSum(25, 3);
  }

  public boolean everyMentsuContainsTerminalOrHonor() {
    return ((manFeatures & pinFeatures & souFeatures & honorFeatures) & 1L << 27) != 0L;
  }

  public boolean everyMentsuContainsTerminal() {
    return ((manFeatures & pinFeatures & souFeatures & honorFeatures) & 1L << 28) != 0L;
  }

  public int ankoFu() {
    return featureSum(29, 0x3f);
  }

  private int featureSum(int shift, int mask) {
    return (int) ((manFeatures >>> shift) & mask)
        + (int) ((pinFeatures >>> shift) & mask)
        + (int) ((souFeatures >>> shift) & mask)
        + (int) ((honorFeatures >>> shift) & mask);
  }

  private void advanceCombination() {
    if (++honorIndex < honorEnd) return;
    honorIndex = honorStart;
    if (++souIndex < souEnd) return;
    souIndex = souStart;
    if (++pinIndex < pinEnd) return;
    pinIndex = pinStart;
    if (++manIndex < manEnd) return;
    active = false;
  }

  public static int jantouTile(long code) {
    return (int) code & TILE_MASK;
  }

  public static int mentsuCount(long code) {
    return (int) (code >>> 6) & 7;
  }

  public static int mentsuTile(long code, int mentsuIndex) {
    return packedMentsu(code, mentsuIndex) & TILE_MASK;
  }

  public static boolean isShuntsu(long code, int mentsuIndex) {
    return (packedMentsu(code, mentsuIndex) & SEQUENCE_FLAG) != 0;
  }

  private static int packedMentsu(long code, int mentsuIndex) {
    if (mentsuIndex < 0 || mentsuIndex >= mentsuCount(code)) {
      throw new IndexOutOfBoundsException(mentsuIndex);
    }
    return (int) (code >>> (GROUPS_SHIFT + mentsuIndex * GROUP_BITS)) & GROUP_MASK;
  }

  private static int localJantouTile(int man, int pin, int sou, int honors) {
    int pair = localJantou(man);
    if (pair >= 0) return Tile.M1 + pair;
    if ((pair = localJantou(pin)) >= 0) return Tile.P1 + pair;
    if ((pair = localJantou(sou)) >= 0) return Tile.S1 + pair;
    return Tile.TON + localJantou(honors);
  }

  private static long pack(int jantouTile, int man, int pin, int sou, int honors) {
    long code = jantouTile;
    int groupOffset = 0;
    code = appendLocalGroups(code, groupOffset, man, Tile.M1);
    groupOffset += localMentsuCount(man);
    code = appendLocalGroups(code, groupOffset, pin, Tile.P1);
    groupOffset += localMentsuCount(pin);
    code = appendLocalGroups(code, groupOffset, sou, Tile.S1);
    groupOffset += localMentsuCount(sou);
    code = appendLocalGroups(code, groupOffset, honors, Tile.TON);
    groupOffset += localMentsuCount(honors);
    return code | (long) groupOffset << 6;
  }

  private static int localMentsuCount(int localPattern) {
    return localPattern >>> 24;
  }

  private static int localJantou(int localPattern) {
    return (localPattern >>> 20 & 0x0f) - 1;
  }

  private static long appendLocalGroups(
      long code, int groupOffset, int localPattern, int tileOffset) {
    int count = localMentsuCount(localPattern);
    for (int groupIndex = 0; groupIndex < count; groupIndex++) {
      int localGroup = localPattern >>> (groupIndex * 5) & 0x1f;
      int group = tileOffset + (localGroup & 0x0f);
      if ((localGroup & 0x10) != 0) group |= SEQUENCE_FLAG;
      code |= (long) group << (GROUPS_SHIFT + (groupOffset + groupIndex) * GROUP_BITS);
    }
    return code;
  }
}
