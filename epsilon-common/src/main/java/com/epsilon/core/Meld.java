package com.epsilon.core;

/**
 * 構成牌、鳴き元、赤牌由来まで確定した不変面子。
 *
 * <p>生成可能な値はクラス初期化時に一度だけ作られ、各ファクトリは単一の一次元テーブルから共有のインスタンスを返す。
 * したがって対局中の面子生成でオブジェクトや構成牌配列を割り当てない。チーは順子の先頭牌を受け取るため、牌配列の並べ替えも行わない。
 */
public final class Meld {

  /** 面子の種別と、種別だけで確定する構造情報。 */
  public enum Type {
    /** 上家の捨て牌で作る順子。 */
    CHI(3, false, false),

    /** 他家の捨て牌で作る刻子。 */
    PON(3, false, false),

    /** 他家の捨て牌で作る明槓子。 */
    DAIMINKAN(4, true, false),

    /** 自手牌4枚で作る暗槓子。 */
    ANKAN(4, true, true),

    /** 既存ポンへ4枚目を加えた槓子。 */
    KAKAN(4, true, false);

    private final int tileCount;
    private final boolean kan;
    private final boolean preservesMenzen;

    Type(int tileCount, boolean kan, boolean preservesMenzen) {
      this.tileCount = tileCount;
      this.kan = kan;
      this.preservesMenzen = preservesMenzen;
    }
  }

  /** 自家から見た面子の鳴き元。暗槓だけは {@link #NONE}。 */
  public enum RelativeSource {
    /** 他家由来を持たない暗槓。 */
    NONE(-1),

    /** 下家からの鳴き。 */
    SHIMOCHA(1),

    /** 対面からの鳴き。 */
    TOIMEN(2),

    /** 上家からの鳴き。 */
    KAMICHA(3);

    private final int playerOffset;

    RelativeSource(int playerOffset) {
      this.playerOffset = playerOffset;
    }

    /**
     * 自家を0とする相対席番号を返す。
     *
     * @return 暗槓では-1、下家1、対面2、上家3
     */
    public int playerOffset() {
      return playerOffset;
    }

    /**
     * 相対席番号を、鳴いた牌を捨てた相手の席に変換する。
     *
     * @param playerOffset 下家1、対面2、上家3
     * @return 対応する鳴き元
     */
    public static RelativeSource fromPlayerOffset(int playerOffset) {
      return switch (playerOffset) {
        case 1 -> SHIMOCHA;
        case 2 -> TOIMEN;
        case 3 -> KAMICHA;
        default -> throw new IllegalArgumentException("playerOffset out of range: " + playerOffset);
      };
    }
  }

  /** 面子に含まれる赤牌の物理的な由来。 */
  public enum AkaSource {
    /** 赤牌を含まない。 */
    NONE,

    /** 他家から鳴いた牌が赤牌。 */
    CALLED_TILE,

    /** 面子へ消費した自手牌が赤牌。 */
    CONSUMED_HAND_TILE,

    /** 加槓で追加した4枚目が赤牌。 */
    ADDED_KAN_TILE;

    /**
     * 副露面子の物理牌情報から赤牌由来を確定する。
     *
     * @param calledTileIsAka 他家から鳴いた牌が赤5なら{@code true}
     * @param consumedHandTileIsAka 自手牌から赤5を消費したなら{@code true}
     * @return 面子へ記録する赤牌由来
     */
    public static AkaSource forCall(boolean calledTileIsAka, boolean consumedHandTileIsAka) {
      if (calledTileIsAka && consumedHandTileIsAka) {
        throw new IllegalArgumentException("a meld cannot contain two red fives");
      }
      if (calledTileIsAka) {
        return CALLED_TILE;
      }
      return consumedHandTileIsAka ? CONSUMED_HAND_TILE : NONE;
    }
  }

  private static final int CHI_CALLED_POSITION_COUNT = 3;
  private static final int RELATIVE_SOURCE_COUNT = RelativeSource.values().length;
  private static final int AKA_SOURCE_COUNT = AkaSource.values().length;
  private static final int CANONICAL_TABLE_SIZE =
      Type.values().length
          * Tile.NUM_TILE_TYPES
          * RELATIVE_SOURCE_COUNT
          * CHI_CALLED_POSITION_COUNT
          * AKA_SOURCE_COUNT;
  private static final RelativeSource[] OPEN_SOURCES = {
    RelativeSource.SHIMOCHA, RelativeSource.TOIMEN, RelativeSource.KAMICHA
  };
  private static final Meld[] CANONICAL_MELDS = createCanonicalMelds();

  private final Type type;
  private final int baseTileType;
  private final int calledTileType;
  private final RelativeSource relativeSource;
  private final AkaSource akaSource;
  private final int canonicalIndex;

  private Meld(
      Type type,
      int baseTileType,
      int calledTileType,
      RelativeSource relativeSource,
      AkaSource akaSource) {
    this.type = type;
    this.baseTileType = baseTileType;
    this.calledTileType = calledTileType;
    this.relativeSource = relativeSource;
    this.akaSource = akaSource;
    this.canonicalIndex =
        canonicalIndex(type, baseTileType, calledTileType, relativeSource, akaSource);
  }

  /**
   * 赤牌を含まないチーを返す。
   *
   * @param sequenceBaseTileType 順子で最も小さい牌種
   * @param calledTileType 上家から鳴いた牌種
   * @return 共有の CHI
   */
  public static Meld chi(int sequenceBaseTileType, int calledTileType) {
    return chi(sequenceBaseTileType, calledTileType, AkaSource.NONE);
  }

  /**
   * 順子と赤牌由来を確定したチーを返す。
   *
   * @param sequenceBaseTileType 順子で最も小さい牌種
   * @param calledTileType 上家から鳴いた牌種
   * @param akaSource 赤牌の物理的な由来
   * @return 共有の CHI
   */
  public static Meld chi(int sequenceBaseTileType, int calledTileType, AkaSource akaSource) {
    requireChiSequence(sequenceBaseTileType, calledTileType);
    return canonical(
        Type.CHI, sequenceBaseTileType, calledTileType, RelativeSource.KAMICHA, akaSource);
  }

  /**
   * 赤牌を含まないポンを返す。
   *
   * @param tileType 刻子の牌種
   * @param relativeSource 捨て牌元の相対席
   * @return 共有の PON
   */
  public static Meld pon(int tileType, RelativeSource relativeSource) {
    return pon(tileType, relativeSource, AkaSource.NONE);
  }

  /**
   * 牌種、鳴き元、赤牌由来を確定したポンを返す。
   *
   * @param tileType 刻子の牌種
   * @param relativeSource 捨て牌元の相対席
   * @param akaSource 赤牌の物理的な由来
   * @return 共有の PON
   */
  public static Meld pon(int tileType, RelativeSource relativeSource, AkaSource akaSource) {
    Tile.requireValidType(tileType, "PON tile");
    requireOpenSource(relativeSource);
    return canonical(Type.PON, tileType, tileType, relativeSource, akaSource);
  }

  /**
   * 赤牌を含まない大明槓を返す。
   *
   * @param tileType 槓子の牌種
   * @param relativeSource 捨て牌元の相対席
   * @return 共有の DAIMINKAN
   */
  public static Meld daiminkan(int tileType, RelativeSource relativeSource) {
    return daiminkan(tileType, relativeSource, AkaSource.NONE);
  }

  /**
   * 牌種、鳴き元、赤牌由来を確定した大明槓を返す。
   *
   * @param tileType 槓子の牌種
   * @param relativeSource 捨て牌元の相対席
   * @param akaSource 赤牌の物理的な由来
   * @return 共有の DAIMINKAN
   */
  public static Meld daiminkan(int tileType, RelativeSource relativeSource, AkaSource akaSource) {
    Tile.requireValidType(tileType, "DAIMINKAN tile");
    requireOpenSource(relativeSource);
    return canonical(Type.DAIMINKAN, tileType, tileType, relativeSource, akaSource);
  }

  /**
   * 赤牌を含まない暗槓を返す。
   *
   * @param tileType 槓子の牌種
   * @return 共有の ANKAN
   */
  public static Meld ankan(int tileType) {
    return ankan(tileType, false);
  }

  /**
   * 自手牌の赤5を含むか確定した暗槓を返す。
   *
   * @param tileType 槓子の牌種
   * @param containsAkaTile 4枚に赤5を含むなら{@code true}
   * @return 共有の ANKAN
   */
  public static Meld ankan(int tileType, boolean containsAkaTile) {
    Tile.requireValidType(tileType, "ANKAN tile");
    AkaSource akaSource = containsAkaTile ? AkaSource.CONSUMED_HAND_TILE : AkaSource.NONE;
    return canonical(Type.ANKAN, tileType, -1, RelativeSource.NONE, akaSource);
  }

  /**
   * 既存ポンへ通常牌を加えた加槓を返す。
   *
   * @param pon 加槓元のポン
   * @return 鳴き元と既存赤牌由来を引き継ぐ共有の KAKAN
   */
  public static Meld kakan(Meld pon) {
    return kakan(pon, false);
  }

  /**
   * 既存ポンへ4枚目を加えた加槓を返す。
   *
   * @param pon 加槓元のポン
   * @param addedTileIsAka 追加する4枚目が赤5なら{@code true}
   * @return 鳴き元と赤牌由来を保持する共有の KAKAN
   */
  public static Meld kakan(Meld pon, boolean addedTileIsAka) {
    if (pon.type != Type.PON) {
      throw new IllegalArgumentException("KAKAN requires a PON meld");
    }
    if (addedTileIsAka && pon.containsAkaTile()) {
      throw new IllegalArgumentException("KAKAN cannot contain two red fives");
    }
    AkaSource resultingAkaSource = addedTileIsAka ? AkaSource.ADDED_KAN_TILE : pon.akaSource;
    return canonical(
        Type.KAKAN, pon.baseTileType, pon.calledTileType, pon.relativeSource, resultingAkaSource);
  }

  /**
   * 面子種別を返す。
   *
   * @return CHI、PONまたは各KAN種別
   */
  public Type type() {
    return type;
  }

  /**
   * 自家から見た鳴き元を返す。
   *
   * @return 暗槓では{@link RelativeSource#NONE}、副露面子では他家
   */
  public RelativeSource relativeSource() {
    return relativeSource;
  }

  /**
   * 面子内の赤牌由来を返す。
   *
   * @return 赤牌を含まなければ{@link AkaSource#NONE}
   */
  public AkaSource akaSource() {
    return akaSource;
  }

  /**
   * チーでは順子の最小牌種、それ以外では刻子・槓子の牌種を返す。
   *
   * @return 面子の基準牌種
   */
  public int baseTileType() {
    return baseTileType;
  }

  /**
   * 面子内の指定位置にある牌種を返す。構成牌配列は生成しない。
   *
   * @param index 0始まりの構成牌インデックス
   * @return 対応する牌種
   */
  public int tileAt(int index) {
    if (index < 0 || index >= type.tileCount) {
      throw new IndexOutOfBoundsException("index=" + index + " size=" + type.tileCount);
    }
    return type == Type.CHI ? baseTileType + index : baseTileType;
  }

  /**
   * 面子を構成する牌種の34-ビットマスクを返す。
   *
   * @return チーでは3牌種、それ以外では基準牌種だけが立ったマスク
   */
  public long tileTypeMask() {
    return type == Type.CHI ? 0b111L << baseTileType : 1L << baseTileType;
  }

  /**
   * 他家から鳴いた牌種を返す。
   *
   * @return 副露面子の鳴いた牌種。暗槓では-1
   */
  public int calledTileType() {
    return calledTileType;
  }

  /**
   * 面子に含まれる赤5の牌種を返す。
   *
   * @return 赤5の牌種。赤5を含まなければ {@code -1}
   */
  public int akaTileType() {
    return containsAkaTile() ? Tile.M5 + Tile.suitOf(baseTileType) * Tile.TILE_TYPES_PER_SUIT : -1;
  }

  /**
   * 面子が赤牌を含むかを返す。
   *
   * @return 赤牌を含むなら{@code true}
   */
  public boolean containsAkaTile() {
    return akaSource != AkaSource.NONE;
  }

  /**
   * 面子へ消費した自手牌が赤牌かを返す。
   *
   * @return 赤牌由来が{@link AkaSource#CONSUMED_HAND_TILE}なら{@code true}
   */
  public boolean consumedHandTileIsAka() {
    return akaSource == AkaSource.CONSUMED_HAND_TILE;
  }

  /**
   * 加槓で追加した4枚目が赤牌かを返す。
   *
   * @return 赤牌由来が{@link AkaSource#ADDED_KAN_TILE}なら{@code true}
   */
  public boolean addedTileIsAka() {
    return akaSource == AkaSource.ADDED_KAN_TILE;
  }

  boolean isKakanUpgradeOf(Meld pon) {
    if (type != Type.KAKAN
        || pon.type != Type.PON
        || baseTileType != pon.baseTileType
        || relativeSource != pon.relativeSource) {
      return false;
    }
    if (akaSource == AkaSource.ADDED_KAN_TILE) {
      return pon.akaSource == AkaSource.NONE;
    }
    return akaSource == pon.akaSource;
  }

  /**
   * 面子を作っても門前が維持されるかを返す。
   *
   * @return 暗槓だけ{@code true}
   */
  public boolean preservesMenzen() {
    return type.preservesMenzen;
  }

  /**
   * 面子がいずれかの槓子かを返す。
   *
   * @return 大明槓・暗槓・加槓なら{@code true}
   */
  public boolean isKan() {
    return type.kan;
  }

  /**
   * 面子の構成牌数を返す。
   *
   * @return 槓子は4、それ以外は3
   */
  public int size() {
    return type.tileCount;
  }

  @Override
  public boolean equals(Object object) {
    return this == object || object instanceof Meld other && canonicalIndex == other.canonicalIndex;
  }

  @Override
  public int hashCode() {
    return canonicalIndex;
  }

  /** 共有の面子テーブル内の安定したIDを返す。 */
  public int canonicalId() {
    return canonicalIndex;
  }

  /** 共有の IDに対応する面子を返す。 */
  public static Meld fromCanonicalId(int canonicalId) {
    if (canonicalId < 0 || canonicalId >= CANONICAL_MELDS.length) {
      throw new IllegalArgumentException("canonical meld ID out of range: " + canonicalId);
    }
    Meld meld = CANONICAL_MELDS[canonicalId];
    if (meld == null) {
      throw new IllegalArgumentException("unused canonical meld ID: " + canonicalId);
    }
    return meld;
  }

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder(type.name()).append('[');
    for (int tileIndex = 0; tileIndex < type.tileCount; tileIndex++) {
      if (tileIndex > 0) {
        result.append(',');
      }
      result.append(Tile.name(tileAt(tileIndex)));
    }
    return result.append(']').toString();
  }

  private static Meld[] createCanonicalMelds() {
    Meld[] melds = new Meld[CANONICAL_TABLE_SIZE];
    registerChiMelds(melds);
    registerTripletMelds(melds);
    return melds;
  }

  private static void registerChiMelds(Meld[] melds) {
    for (int suit = 0; suit < 3; suit++) {
      int suitBaseTileType = suit * 9;
      for (int sequenceStart = 0; sequenceStart < 7; sequenceStart++) {
        int sequenceBaseTileType = suitBaseTileType + sequenceStart;
        boolean containsFive = sequenceStart <= 4 && sequenceStart + 2 >= 4;
        for (int calledPosition = 0; calledPosition < CHI_CALLED_POSITION_COUNT; calledPosition++) {
          int calledTileType = sequenceBaseTileType + calledPosition;
          register(
              melds,
              new Meld(
                  Type.CHI,
                  sequenceBaseTileType,
                  calledTileType,
                  RelativeSource.KAMICHA,
                  AkaSource.NONE));
          if (Tile.canBeAka(calledTileType)) {
            register(
                melds,
                new Meld(
                    Type.CHI,
                    sequenceBaseTileType,
                    calledTileType,
                    RelativeSource.KAMICHA,
                    AkaSource.CALLED_TILE));
          } else if (containsFive) {
            register(
                melds,
                new Meld(
                    Type.CHI,
                    sequenceBaseTileType,
                    calledTileType,
                    RelativeSource.KAMICHA,
                    AkaSource.CONSUMED_HAND_TILE));
          }
        }
      }
    }
  }

  private static void registerTripletMelds(Meld[] melds) {
    for (int tileType = 0; tileType < Tile.NUM_TILE_TYPES; tileType++) {
      register(melds, new Meld(Type.ANKAN, tileType, -1, RelativeSource.NONE, AkaSource.NONE));
      if (Tile.canBeAka(tileType)) {
        register(
            melds,
            new Meld(Type.ANKAN, tileType, -1, RelativeSource.NONE, AkaSource.CONSUMED_HAND_TILE));
      }

      for (RelativeSource source : OPEN_SOURCES) {
        registerOpenMelds(melds, tileType, source, AkaSource.NONE);
        if (Tile.canBeAka(tileType)) {
          registerOpenMelds(melds, tileType, source, AkaSource.CALLED_TILE);
          registerOpenMelds(melds, tileType, source, AkaSource.CONSUMED_HAND_TILE);
          register(
              melds, new Meld(Type.KAKAN, tileType, tileType, source, AkaSource.ADDED_KAN_TILE));
        }
      }
    }
  }

  private static void registerOpenMelds(
      Meld[] melds, int tileType, RelativeSource source, AkaSource akaSource) {
    register(melds, new Meld(Type.PON, tileType, tileType, source, akaSource));
    register(melds, new Meld(Type.DAIMINKAN, tileType, tileType, source, akaSource));
    register(melds, new Meld(Type.KAKAN, tileType, tileType, source, akaSource));
  }

  private static void register(Meld[] melds, Meld meld) {
    melds[meld.canonicalIndex] = meld;
  }

  private static Meld canonical(
      Type type,
      int baseTileType,
      int calledTileType,
      RelativeSource relativeSource,
      AkaSource akaSource) {
    Meld meld =
        CANONICAL_MELDS[
            canonicalIndex(type, baseTileType, calledTileType, relativeSource, akaSource)];
    if (meld == null) {
      throw new IllegalArgumentException(
          "illegal meld: type="
              + type
              + " baseTileType="
              + baseTileType
              + " calledTileType="
              + calledTileType
              + " relativeSource="
              + relativeSource
              + " akaSource="
              + akaSource);
    }
    return meld;
  }

  private static int canonicalIndex(
      Type type,
      int baseTileType,
      int calledTileType,
      RelativeSource relativeSource,
      AkaSource akaSource) {
    int calledPosition = type == Type.CHI ? calledTileType - baseTileType : 0;
    return ((((type.ordinal() * Tile.NUM_TILE_TYPES + baseTileType) * RELATIVE_SOURCE_COUNT
                        + relativeSource.ordinal())
                    * CHI_CALLED_POSITION_COUNT
                + calledPosition)
            * AKA_SOURCE_COUNT)
        + akaSource.ordinal();
  }

  private static void requireChiSequence(int sequenceBaseTileType, int calledTileType) {
    Tile.requireValidType(sequenceBaseTileType, "CHI sequence base");
    Tile.requireValidType(calledTileType, "CHI called tile");
    if (!Tile.isNumberTile(sequenceBaseTileType) || Tile.numberOf(sequenceBaseTileType) > 6) {
      throw new IllegalArgumentException(
          "CHI sequence base must be a suited 1-7 tile: " + sequenceBaseTileType);
    }
    int calledPosition = calledTileType - sequenceBaseTileType;
    if (calledPosition < 0 || calledPosition >= CHI_CALLED_POSITION_COUNT) {
      throw new IllegalArgumentException(
          "CHI called tile must be in the sequence: " + calledTileType);
    }
  }

  private static void requireOpenSource(RelativeSource relativeSource) {
    if (relativeSource == RelativeSource.NONE) {
      throw new IllegalArgumentException("open meld requires another-player source");
    }
  }
}
