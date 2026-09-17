package com.epsilon.core;

/**
 * 麻雀牌の定数定義と牌操作ユーティリティ。
 *
 * <p>牌種ID (0-33): 0-8: 萬子 (1m-9m) 9-17: 筒子 (1p-9p) 18-26: 索子 (1s-9s) 27-33: 字牌 (東南西北白發中)
 *
 * <p>実牌ID (0-135): 各牌種4枚ずつ (tileId / 4 = 牌種ID)
 */
public final class Tile {

  private Tile() {}

  // セクション: 牌種数
  /** 牌種 ID が表す異なる牌種の総数。 */
  public static final int NUM_TILE_TYPES = 34;

  /** 四人麻雀で使う物理牌の総数。 */
  public static final int NUM_TILES = 136;

  /** 一つの牌種に存在する物理牌枚数。 */
  public static final int TILES_PER_TYPE = 4;

  /** 一つの数牌色に存在する牌種数。 */
  public static final int TILE_TYPES_PER_SUIT = 9;

  // セクション: 色定義
  /** 萬子を表す色 ID。 */
  public static final int SUIT_MAN = 0;

  /** 筒子を表す色 ID。 */
  public static final int SUIT_PIN = 1;

  /** 索子を表す色 ID。 */
  public static final int SUIT_SOU = 2;

  /** 字牌を表す色 ID。 */
  public static final int SUIT_HONOR = 3;

  // セクション: 萬子 (0-8)
  /** 一萬の牌種 ID。 */
  public static final int M1 = 0;

  /** 二萬の牌種 ID。 */
  public static final int M2 = 1;

  /** 三萬の牌種 ID。 */
  public static final int M3 = 2;

  /** 四萬の牌種 ID。 */
  public static final int M4 = 3;

  /** 五萬の牌種 ID。 */
  public static final int M5 = 4;

  /** 六萬の牌種 ID。 */
  public static final int M6 = 5;

  /** 七萬の牌種 ID。 */
  public static final int M7 = 6;

  /** 八萬の牌種 ID。 */
  public static final int M8 = 7;

  /** 九萬の牌種 ID。 */
  public static final int M9 = 8;

  // セクション: 筒子 (9-17)
  /** 一筒の牌種 ID。 */
  public static final int P1 = 9;

  /** 二筒の牌種 ID。 */
  public static final int P2 = 10;

  /** 三筒の牌種 ID。 */
  public static final int P3 = 11;

  /** 四筒の牌種 ID。 */
  public static final int P4 = 12;

  /** 五筒の牌種 ID。 */
  public static final int P5 = 13;

  /** 六筒の牌種 ID。 */
  public static final int P6 = 14;

  /** 七筒の牌種 ID。 */
  public static final int P7 = 15;

  /** 八筒の牌種 ID。 */
  public static final int P8 = 16;

  /** 九筒の牌種 ID。 */
  public static final int P9 = 17;

  // セクション: 索子 (18-26)
  /** 一索の牌種 ID。 */
  public static final int S1 = 18;

  /** 二索の牌種 ID。 */
  public static final int S2 = 19;

  /** 三索の牌種 ID。 */
  public static final int S3 = 20;

  /** 四索の牌種 ID。 */
  public static final int S4 = 21;

  /** 五索の牌種 ID。 */
  public static final int S5 = 22;

  /** 六索の牌種 ID。 */
  public static final int S6 = 23;

  /** 七索の牌種 ID。 */
  public static final int S7 = 24;

  /** 八索の牌種 ID。 */
  public static final int S8 = 25;

  /** 九索の牌種 ID。 */
  public static final int S9 = 26;

  // セクション: 字牌 (27-33)
  /** 東の牌種 ID。 */
  public static final int TON = 27;

  /** 南の牌種 ID。 */
  public static final int NAN = 28;

  /** 西の牌種 ID。 */
  public static final int SHA = 29;

  /** 北の牌種 ID。 */
  public static final int PEI = 30;

  /** 白の牌種 ID。 */
  public static final int HAKU = 31;

  /** 發の牌種 ID。 */
  public static final int HATSU = 32;

  /** 中の牌種 ID。 */
  public static final int CHUN = 33;

  /** 么九牌（老頭牌・字牌）に対応する34-ビット牌種マスク。 */
  public static final long TERMINAL_OR_HONOR_TYPE_MASK =
      (1L << M1)
          | (1L << M9)
          | (1L << P1)
          | (1L << P9)
          | (1L << S1)
          | (1L << S9)
          | (1L << TON)
          | (1L << NAN)
          | (1L << SHA)
          | (1L << PEI)
          | (1L << HAKU)
          | (1L << HATSU)
          | (1L << CHUN);

  // セクション: 赤ドラ実牌ID (各5のコピー 0)
  /** 赤五萬に割り当てる物理牌 ID。 */
  public static final int AKA_M5_ID = M5 * TILES_PER_TYPE;

  /** 赤五筒に割り当てる物理牌 ID。 */
  public static final int AKA_P5_ID = P5 * TILES_PER_TYPE;

  /** 赤五索に割り当てる物理牌 ID。 */
  public static final int AKA_S5_ID = S5 * TILES_PER_TYPE;

  // セクション: 么九牌 (端牌・字牌)
  private static final boolean[] TERMINAL_OR_HONOR = new boolean[NUM_TILE_TYPES];
  private static final boolean[] TERMINAL = new boolean[NUM_TILE_TYPES];
  private static final boolean[] HONOR = new boolean[NUM_TILE_TYPES];

  static {
    for (int i = 0; i < NUM_TILE_TYPES; i++) {
      HONOR[i] = i >= TON;
      if (i < 27) {
        int num = i % 9;
        TERMINAL[i] = num == 0 || num == 8;
      }
      TERMINAL_OR_HONOR[i] = TERMINAL[i] || HONOR[i];
    }
  }

  // セクション: 牌名
  private static final String[] TILE_NAMES = {
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "東", "南", "西", "北", "白", "發", "中"
  };

  /**
   * 値が牌種 ID の範囲内かを返す。
   *
   * @param tileType 検証する値
   * @return 0以上34未満なら {@code true}
   */
  public static boolean isValidType(int tileType) {
    return tileType >= 0 && tileType < NUM_TILE_TYPES;
  }

  /**
   * 値が牌種 ID の範囲内であることを検証する。
   *
   * @param tileType 検証する値
   * @param name 例外メッセージに含める引数名
   * @throws IllegalArgumentException 値が0以上34未満でない場合
   */
  public static void requireValidType(int tileType, String name) {
    if (!isValidType(tileType)) {
      throw new IllegalArgumentException(
          name + " must be 0-" + (NUM_TILE_TYPES - 1) + ", got " + tileType);
    }
  }

  /**
   * 物理牌 ID から牌種 ID を取得する。
   *
   * @param tileId 0から135の物理牌 ID
   * @return 0から33の牌種 ID
   */
  public static int typeOf(int tileId) {
    return tileId >> 2; // tileId / 4
  }

  /**
   * 牌種の色 ID を取得する。
   *
   * @param tileType 牌種 ID
   * @return {@link #SUIT_MAN} から {@link #SUIT_HONOR} の色 ID
   */
  public static int suitOf(int tileType) {
    if (tileType >= 27) {
      return SUIT_HONOR;
    }
    return tileType / 9;
  }

  /**
   * 牌種の色内番号を取得する。
   *
   * @param tileType 牌種 ID
   * @return 数牌では0から8、字牌では0から6
   */
  public static int numberOf(int tileType) {
    if (tileType >= 27) {
      return tileType - 27;
    }
    return tileType % 9;
  }

  /**
   * 数牌かどうかを返す。
   *
   * @param tileType 牌種 ID
   * @return 萬子、筒子または索子なら {@code true}
   */
  public static boolean isNumberTile(int tileType) {
    return tileType < 27;
  }

  /**
   * 字牌かどうかを返す。
   *
   * @param tileType 牌種 ID
   * @return 風牌または三元牌なら {@code true}
   */
  public static boolean isHonor(int tileType) {
    return HONOR[tileType];
  }

  /**
   * 老頭牌かどうかを返す。
   *
   * @param tileType 牌種 ID
   * @return 数牌の1または9なら {@code true}
   */
  public static boolean isTerminal(int tileType) {
    return TERMINAL[tileType];
  }

  /**
   * 么九牌かどうかを返す。
   *
   * @param tileType 牌種 ID
   * @return 老頭牌または字牌なら {@code true}
   */
  public static boolean isTerminalOrHonor(int tileType) {
    return TERMINAL_OR_HONOR[tileType];
  }

  /**
   * 三元牌かどうかを返す。
   *
   * @param tileType 牌種 ID
   * @return 白、發または中なら {@code true}
   */
  public static boolean isDragon(int tileType) {
    return tileType >= HAKU && tileType <= CHUN;
  }

  /**
   * 風牌かどうかを返す。
   *
   * @param tileType 牌種 ID
   * @return 東、南、西または北なら {@code true}
   */
  public static boolean isWind(int tileType) {
    return tileType >= TON && tileType <= PEI;
  }

  /**
   * 牌種 ID から短い表示名を取得する。
   *
   * @param tileType 牌種 ID
   * @return 例: {@code 5m} または {@code 東}
   */
  public static String name(int tileType) {
    return TILE_NAMES[tileType];
  }

  /**
   * 物理牌 ID が赤ドラかどうかを返す。
   *
   * @param physicalTileId 物理牌 ID
   * @return 赤五萬、赤五筒または赤五索なら {@code true}
   */
  public static boolean isAka(int physicalTileId) {
    return physicalTileId == AKA_M5_ID
        || physicalTileId == AKA_P5_ID
        || physicalTileId == AKA_S5_ID;
  }

  /**
   * 牌種が赤ドラを持ち得るかを返す。
   *
   * @param tileType 牌種 ID
   * @return 五萬、五筒または五索なら {@code true}
   */
  public static boolean canBeAka(int tileType) {
    return tileType == M5 || tileType == P5 || tileType == S5;
  }

  /**
   * ドラ表示牌から実際のドラ牌種を取得する。
   *
   * @param indicator ドラ表示牌の牌種 ID
   * @return 数牌、風牌、三元牌それぞれの循環規則を適用したドラ牌種 ID
   */
  public static int doraFrom(int indicator) {
    if (indicator >= 27) {
      // 字牌: 東→南→西→北→東, 白→發→中→白
      if (indicator <= PEI) {
        return TON + (indicator - TON + 1) % 4;
      } else {
        return HAKU + (indicator - HAKU + 1) % 3;
      }
    }
    // 数牌: 1→2→...→9→1
    int suit = suitOf(indicator);
    int num = numberOf(indicator);
    return suit * 9 + (num + 1) % 9;
  }
}
