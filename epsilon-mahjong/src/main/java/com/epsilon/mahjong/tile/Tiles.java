package com.epsilon.mahjong.tile;

/**
 * 牌の番号。種類は 0〜33 で、萬子 1〜9、筒子 1〜9、索子 1〜9、東南西北、白發中の順に並ぶ。物理 id は、種類 × 4 ＋ 種類の中での
 * 番号（0〜3。天鳳と同じ番号）。種類の集合は、k 番目のビットが種類 k を表す 34 ビットのマスクで持つ。種類と物理 id の引数は、
 * 範囲を確かめない。
 */
public final class Tiles {

  /** ドラが一巡りするグループ。各グループは連続した種類の範囲で、宣言の順につなげると 0〜33 の全ての種類になる。 */
  public enum Group {
    MAN(M1, M9),
    PIN(P1, P9),
    SOU(S1, S9),
    WIND(TON, PEI),
    DRAGON(HAKU, CHUN);

    private static final Group[] OF_KIND = ofKind();

    private final int first;
    private final int last;

    Group(int first, int last) {
      this.first = first;
      this.last = last;
    }

    public int first() {
      return first;
    }

    public int last() {
      return last;
    }

    public int size() {
      return last - first + 1;
    }

    /** 萬子・筒子・索子（数牌の色）か。 */
    public boolean suited() {
      return last < TON;
    }

    public static Group of(int kind) {
      return OF_KIND[kind];
    }

    private static Group[] ofKind() {
      Group[] groups = new Group[KINDS];
      for (Group group : values()) {
        for (int kind = group.first; kind <= group.last; kind++) {
          groups[kind] = group;
        }
      }
      return groups;
    }
  }

  public static final int KINDS = 34;
  public static final int COPIES = 4;

  /** 物理 id の数。どの {@link TileSet} の物理 id もこの範囲に入る。 */
  public static final int PHYSICAL_IDS = KINDS * COPIES;

  /** 1 色の数牌の種類の数。 */
  public static final int SUIT_SIZE = 9;

  /** 数牌の色（萬子・筒子・索子）の数。赤 5 がありうる種類の数でもある。 */
  public static final int NUMBER_SUITS = 3;

  /** 5 が色の中で何番目か（0 始まり）。 */
  public static final int FIVE = 4;

  /** 赤 5 に割り当てる、種類の中での番号。 */
  public static final int RED_COPY = 0;

  public static final int M1 = 0;
  public static final int M2 = 1;
  public static final int M3 = 2;
  public static final int M4 = 3;
  public static final int M5 = 4;
  public static final int M6 = 5;
  public static final int M7 = 6;
  public static final int M8 = 7;
  public static final int M9 = 8;
  public static final int P1 = 9;
  public static final int P2 = 10;
  public static final int P3 = 11;
  public static final int P4 = 12;
  public static final int P5 = 13;
  public static final int P6 = 14;
  public static final int P7 = 15;
  public static final int P8 = 16;
  public static final int P9 = 17;
  public static final int S1 = 18;
  public static final int S2 = 19;
  public static final int S3 = 20;
  public static final int S4 = 21;
  public static final int S5 = 22;
  public static final int S6 = 23;
  public static final int S7 = 24;
  public static final int S8 = 25;
  public static final int S9 = 26;
  public static final int TON = 27;
  public static final int NAN = 28;
  public static final int SHA = 29;
  public static final int PEI = 30;
  public static final int HAKU = 31;
  public static final int HATSU = 32;
  public static final int CHUN = 33;

  public static final long ALL_KINDS = (1L << KINDS) - 1;

  /** 赤 5 がありうる種類（5m・5p・5s）。どれに赤があるかは {@link TileSet} が決める。 */
  public static final long FIVES = bit(M5) | bit(P5) | bit(S5);

  public static final long TERMINALS = bit(M1) | bit(M9) | bit(P1) | bit(P9) | bit(S1) | bit(S9);
  public static final long HONORS = range(TON, CHUN);
  public static final long TERMINALS_AND_HONORS = TERMINALS | HONORS;

  private Tiles() {}

  public static long bit(int kind) {
    return 1L << kind;
  }

  /** 連続した種類（両端を含む）のマスク。 */
  public static long range(int first, int last) {
    return ((1L << (last - first + 1)) - 1) << first;
  }

  public static boolean contains(long kinds, int kind) {
    return (kinds >>> kind & 1L) != 0;
  }

  public static int kind(int physicalId) {
    return physicalId >> 2;
  }

  public static int copy(int physicalId) {
    return physicalId & 3;
  }

  public static int physicalId(int kind, int copy) {
    return kind << 2 | copy;
  }

  /** グループの中での番号（0 始まり）。数牌は数 − 1、東南西北は 0〜3、白發中は 0〜2。 */
  public static int number(int kind) {
    return kind - Group.of(kind).first;
  }
}
