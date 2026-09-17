package com.epsilon.core;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;

/**
 * 実際に選択できる行動と、保存やモデル入力に使う固定IDとの対応を表す不変の値。
 *
 * <p>同じ牌種でも赤牌と通常牌、手出しとツモ切りは区別する。チー・ポン・槓も消費する牌の組み合わせによって別のIDを持つ。他家から鳴いた五が赤牌かどうかは、行動IDだけでは決まらない場合があるため、{@link
 * MeldAkaTileInclusion#DEPENDS_ON_CALLED_TILE}で表し、実際の捨て牌情報と合わせて判定する。
 *
 * <p>{@link #toIndex()}と{@link
 * #fromIndex(int)}は定数時間で相互変換する。IDの定義を変える場合は、入力スキーマとチェックポイントの互換性も更新する必要がある。
 */
public final class Action implements Serializable {

  @Serial private static final long serialVersionUID = 1L;

  /**
   * 行動が属する排他的な意味グループ。
   *
   * <p>対局行動の分類、方策モデルの分岐、入力特徴で共通利用する。{@link #isResponse()} は他家の打牌・槓宣言に対する応答か、自分の手番中の選択かを表す。
   */
  public enum Group {
    /** 通常打牌。 */
    DAHAI(false),

    /** CHI・PON・DAIMINKANによる他家打牌への鳴き。 */
    MELD(true),

    /** 自分の手番で行うANKAN・KAKAN。 */
    KAN(false),

    /** RIICHI宣言を伴う打牌。 */
    RIICHI(false),

    /** TSUMO和了。 */
    TSUMO(false),

    /** RON和了。 */
    RON(true),

    /** 他家の打牌または槓宣言への応答を見送る。 */
    PASS(true),

    /** 九種九牌による途中流局宣言。 */
    KYUSHU(false);

    private final boolean responseAction;

    Group(boolean responseAction) {
      this.responseAction = responseAction;
    }

    /**
     * 他家の打牌・槓宣言に対する応答グループかを返す。
     *
     * @return 応答行動なら {@code true}
     */
    public boolean isResponse() {
      return responseAction;
    }
  }

  /**
   * エンジンが実行可能な行動種別。
   *
   * <p>仮想的なDECLINE/CONTINUEは含めず、実際の合法手だけを表す。各種別は意味グループ、槓かどうか、物理牌選択をIDに保持するかを自身で宣言するため、呼び出し側で種別集合を重複管理しない。
   */
  public enum Type {
    /** 通常打牌。 */
    DAHAI(Group.DAHAI, false, true),

    /** 上家の打牌から順子を作る鳴き。 */
    CHI(Group.MELD, false, true),

    /** 他家の打牌から刻子を作る鳴き。 */
    PON(Group.MELD, false, true),

    /** 他家の打牌から槓子を作る大明槓。 */
    DAIMINKAN(Group.MELD, true, false),

    /** 手牌4枚を使う暗槓。 */
    ANKAN(Group.KAN, true, false),

    /** 既存のポンへ4枚目を加える加槓。 */
    KAKAN(Group.KAN, true, false),

    /** リーチ宣言を伴う打牌。 */
    RIICHI_DAHAI(Group.RIICHI, false, true),

    /** ツモ和了。 */
    TSUMO_AGARI(Group.TSUMO, false, false),

    /** ロン和了。 */
    RON_AGARI(Group.RON, false, false),

    /** 他家行動への応答を見送る。 */
    PASS(Group.PASS, false, false),

    /** 九種九牌による途中流局を宣言する。 */
    KYUSHU_KYUHAI(Group.KYUSHU, false, false);

    private final Group group;
    private final boolean kanAction;
    private final boolean requiresTileSelection;

    Type(Group group, boolean kanAction, boolean requiresTileSelection) {
      this.group = group;
      this.kanAction = kanAction;
      this.requiresTileSelection = requiresTileSelection;
    }

    /**
     * この種別が属する排他的な意味グループを返す。
     *
     * @return 行動の意味グループ
     */
    public Group group() {
      return group;
    }

    /**
     * 他家の捨て牌または槓宣言に対する応答かを返す。
     *
     * @return 応答行動なら {@code true}
     */
    public boolean isResponse() {
      return group.isResponse();
    }

    /**
     * チー・ポン・大明槓の応答かを返す。
     *
     * @return 鳴き行動なら {@code true}
     */
    public boolean isCall() {
      return group == Group.MELD;
    }

    /**
     * 大明槓・暗槓・加槓のいずれかを返す。
     *
     * @return 槓行動なら {@code true}
     */
    public boolean isKan() {
      return kanAction;
    }

    /**
     * 手番中の暗槓・加槓かを返す。大明槓はMELD応答なので含めない。
     *
     * @return 手番中の槓行動なら {@code true}
     */
    public boolean isTurnKan() {
      return group == Group.KAN;
    }

    /**
     * 赤牌・ツモ切りを含む物理牌選択を行動自身が保持する種別かを返す。
     *
     * @return {@link TileSelection} が必要なら {@code true}
     */
    public boolean requiresTileSelection() {
      return requiresTileSelection;
    }

    /**
     * この行動が面子を新設または加槓するかを返す。
     *
     * @return 面子を変更する行動なら {@code true}
     */
    public boolean createsMeld() {
      return isCall() || isTurnKan();
    }

    /**
     * 成立時に全家の一発権を消す鳴きまたは槓かを返す。
     *
     * @return 一発を中断する行動なら {@code true}
     */
    public boolean interruptsIppatsu() {
      return createsMeld();
    }
  }

  /**
   * 行動が明示的に選ぶ物理牌の赤牌・ツモ切り情報。
   *
   * <p>打牌では赤/非赤と手出し/ツモ切りの直積、鳴きでは手牌から消費する同牌種の赤/非赤を区別する。物理牌選択を持たない行動は {@link #NOT_APPLICABLE} を使う。
   */
  public enum TileSelection {
    /** 物理牌選択を持たない行動。 */
    NOT_APPLICABLE,

    /** 非赤の手出し、または鳴きで手牌から消費する非赤牌。 */
    NON_AKA,

    /** 赤牌の手出し、または鳴きで手牌から消費する赤牌。 */
    AKA,

    /** 非赤牌のツモ切り。 */
    TSUMOGIRI,

    /** 赤牌のツモ切り。 */
    TSUMOGIRI_AKA;

    /**
     * 行動が明示的に赤牌を選んでいるかを返す。
     *
     * @return 赤牌選択なら {@code true}
     */
    public boolean usesAkaTileFromHand() {
      return this == AKA || this == TSUMOGIRI_AKA;
    }

    /**
     * 打牌がツモ切りかを返す。打牌以外の選択では常に {@code false}。
     *
     * @return ツモ切りなら {@code true}
     */
    public boolean isTsumogiri() {
      return this == TSUMOGIRI || this == TSUMOGIRI_AKA;
    }

    /**
     * 打牌の物理属性から対応する選択値を返す。
     *
     * @param usesAkaTileFromHand 赤牌を捨てるか
     * @param isTsumogiri ツモ切りか
     * @return 赤/非赤と手出し/ツモ切りを保持する選択値
     */
    public static TileSelection forDiscard(boolean usesAkaTileFromHand, boolean isTsumogiri) {
      if (isTsumogiri) {
        return usesAkaTileFromHand ? TSUMOGIRI_AKA : TSUMOGIRI;
      }
      return usesAkaTileFromHand ? AKA : NON_AKA;
    }
  }

  /**
   * 行動 IDだけから判定できる、生成後面子への赤牌包含条件。
   *
   * <p>{@link #DEPENDS_ON_CALLED_TILE}
   * はチー・ポンで他家の5を取り込む場合に使う。最終的な面子は捨て牌イベントの赤牌フラグを加えて解決するため、この値を「赤牌なし」として扱ってはならない。
   */
  public enum MeldAkaTileInclusion {
    /** 面子を生成しない行動。 */
    NOT_APPLICABLE,

    /** 行動 IDだけで、生成後面子に赤牌がないと確定する。 */
    NONE,

    /** 行動 IDだけで、生成後面子に赤牌があると確定する。 */
    GUARANTEED,

    /** 他家から鳴いた物理牌が赤牌かどうかで決まる。 */
    DEPENDS_ON_CALLED_TILE;

    /**
     * 鳴いた牌の物理IDによらず赤牌を含むかを返す。
     *
     * @return 赤牌包含が確定しているなら {@code true}
     */
    public boolean isGuaranteed() {
      return this == GUARANTEED;
    }

    /**
     * 鳴いた牌が赤なら含むケースも含め、赤牌を持ち得るかを返す。
     *
     * @return 生成後面子が赤牌を含み得るなら {@code true}
     */
    public boolean mayContainAkaTile() {
      return this == GUARANTEED || this == DEPENDS_ON_CALLED_TILE;
    }

    private boolean resolve(boolean calledTileIsAka) {
      return switch (this) {
        case NONE -> false;
        case GUARANTEED -> true;
        case DEPENDS_ON_CALLED_TILE -> calledTileIsAka;
        case NOT_APPLICABLE -> throw new IllegalStateException("Action does not create a meld");
      };
    }
  }

  /** DAHAIとRIICHI_DAHAIが共有する、赤牌・ツモ切りを区別した物理打牌識別情報数。 */
  public static final int DISCARD_IDENTITY_COUNT = 74;

  /** 固定行動 ID空間の要素数。合法候補数やネットワーク出力幅ではない。 */
  public static final int ACTION_SPACE_SIZE =
      DISCARD_IDENTITY_COUNT + 81 + 37 + 102 + DISCARD_IDENTITY_COUNT + 4;

  /**
   * 実行する行動種別。
   *
   * @serial 実行する行動種別
   */
  private final Type type;

  /**
   * 対象牌種。牌を対象にしない行動では {@code -1}。
   *
   * @serial 対象牌種または {@code -1}
   */
  private final int tileType;

  /**
   * チーの昇順3牌種。チー以外では {@code null}。
   *
   * @serial チーの昇順3牌種または {@code null}
   */
  private final int[] chiTileTypes;

  /**
   * 赤牌と自摸切りを含む物理牌選択。
   *
   * @serial 物理牌選択
   */
  private final TileSelection tileSelection;

  /**
   * 生成後面子へ赤牌が含まれる条件。
   *
   * @serial 赤牌包含条件
   */
  private final MeldAkaTileInclusion meldAkaTileInclusion;

  /** 生成時に確定する派生ID。Java復号時は元の意味情報から共有のインスタンスへ戻す。 */
  private final transient int actionIndex;

  private Action(Type type, int tileType, int[] chiTileTypes, TileSelection tileSelection) {
    this.type = type;
    this.tileType = tileType;
    this.chiTileTypes = chiTileTypes;
    this.tileSelection = tileSelection;
    if ((type == Type.CHI) != (chiTileTypes != null)) {
      throw new IllegalArgumentException("CHI alone must carry chi tile types: " + type);
    }
    if (type.requiresTileSelection() == (tileSelection == TileSelection.NOT_APPLICABLE)) {
      throw new IllegalArgumentException(
          "Tile selection does not match action type: " + type + " / " + tileSelection);
    }
    this.meldAkaTileInclusion = classifyMeldAkaTileInclusion(type, tileType, tileSelection);
    this.actionIndex = computeIndex();
  }

  /**
   * 指定牌種を非赤の手出しとして捨てる行動を返す。
   *
   * @param tileType 捨てる牌種ID（0-33）
   * @return 共有の打牌行動
   */
  public static Action dahai(int tileType) {
    return fromIndex(dahaiIndex(tileType, TileSelection.NON_AKA));
  }

  /**
   * 指定牌種を手出しする行動を返す。
   *
   * @param tileType 捨てる牌種ID（0-33）
   * @param usesAkaTileFromHand 赤牌を選ぶなら {@code true}
   * @return 共有の打牌行動
   */
  public static Action dahai(int tileType, boolean usesAkaTileFromHand) {
    return fromIndex(
        dahaiIndex(tileType, usesAkaTileFromHand ? TileSelection.AKA : TileSelection.NON_AKA));
  }

  /**
   * 赤牌とツモ切りを明示した打牌行動を返す。
   *
   * @param tileType 捨てる牌種ID（0-33）
   * @param usesAkaTileFromHand 赤牌を選ぶなら {@code true}
   * @param isTsumogiri ツモ切りなら {@code true}
   * @return 共有の打牌行動
   */
  public static Action dahai(int tileType, boolean usesAkaTileFromHand, boolean isTsumogiri) {
    return fromIndex(
        dahaiIndex(tileType, TileSelection.forDiscard(usesAkaTileFromHand, isTsumogiri)));
  }

  /**
   * 物理牌選択を明示した打牌行動を返す。
   *
   * @param tileType 捨てる牌種ID（0-33）
   * @param tileSelection 赤牌・手出し・ツモ切りの組合せ
   * @return 共有の打牌行動
   */
  public static Action dahai(int tileType, TileSelection tileSelection) {
    return fromIndex(dahaiIndex(tileType, tileSelection));
  }

  /**
   * 手牌から赤牌を使わないCHI 行動を返す。
   *
   * @param chiTileTypes 昇順に並べた順子3牌種
   * @param calledTileType 他家から鳴く牌種
   * @return 共有の CHI 行動
   */
  public static Action chi(int[] chiTileTypes, int calledTileType) {
    return fromIndex(chiActionIndex(chiTileTypes, calledTileType, false));
  }

  /**
   * 順子と手牌側の赤牌選択を明示したCHI 行動を返す。
   *
   * @param chiTileTypes 昇順に並べた順子3牌種
   * @param calledTileType 他家から鳴く牌種
   * @param consumeAkaFromHand 手牌から赤5を消費するなら {@code true}
   * @return 共有の CHI 行動
   */
  public static Action chi(int[] chiTileTypes, int calledTileType, boolean consumeAkaFromHand) {
    return fromIndex(chiActionIndex(chiTileTypes, calledTileType, consumeAkaFromHand));
  }

  /**
   * 順子の先頭牌から、手牌側で赤牌を使わないCHI 行動を返す。
   *
   * @param sequenceBaseTileType 順子で最も小さい牌種
   * @param calledTileType 他家から鳴く牌種
   * @return 共有の CHI 行動
   */
  public static Action chiSequence(int sequenceBaseTileType, int calledTileType) {
    return fromIndex(chiSequenceActionIndex(sequenceBaseTileType, calledTileType, false));
  }

  /**
   * 順子の先頭牌と手牌側の赤牌選択を明示したCHI 行動を返す。
   *
   * @param sequenceBaseTileType 順子で最も小さい牌種
   * @param calledTileType 他家から鳴く牌種
   * @param consumeAkaFromHand 手牌から赤5を消費するなら {@code true}
   * @return 共有の CHI 行動
   */
  public static Action chiSequence(
      int sequenceBaseTileType, int calledTileType, boolean consumeAkaFromHand) {
    return fromIndex(
        chiSequenceActionIndex(sequenceBaseTileType, calledTileType, consumeAkaFromHand));
  }

  /**
   * 手牌から赤牌を使わないPON 行動を返す。
   *
   * @param tileType ポンする牌種ID（0-33）
   * @return 共有の PON 行動
   */
  public static Action pon(int tileType) {
    return fromIndex(ponIndex(tileType, false));
  }

  /**
   * 手牌側の赤牌選択を明示したPON 行動を返す。
   *
   * @param tileType ポンする牌種ID（0-33）
   * @param consumeAkaFromHand 手牌から赤5を消費するなら {@code true}
   * @return 共有の PON 行動
   */
  public static Action pon(int tileType, boolean consumeAkaFromHand) {
    return fromIndex(ponIndex(tileType, consumeAkaFromHand));
  }

  /**
   * 指定牌種の大明槓行動を返す。
   *
   * @param tileType 槓する牌種ID（0-33）
   * @return 共有の DAIMINKAN 行動
   */
  public static Action daiminkan(int tileType) {
    return fromIndex(kanIndex(Type.DAIMINKAN, tileType));
  }

  /**
   * 指定牌種の暗槓行動を返す。
   *
   * @param tileType 槓する牌種ID（0-33）
   * @return 共有の ANKAN 行動
   */
  public static Action ankan(int tileType) {
    return fromIndex(kanIndex(Type.ANKAN, tileType));
  }

  /**
   * 指定牌種の加槓行動を返す。
   *
   * @param tileType 槓する牌種ID（0-33）
   * @return 共有の KAKAN 行動
   */
  public static Action kakan(int tileType) {
    return fromIndex(kanIndex(Type.KAKAN, tileType));
  }

  /**
   * 非赤牌の手出しを伴うRIICHI 行動を返す。
   *
   * @param discardTileType 宣言牌の牌種ID（0-33）
   * @return 共有の RIICHI_DAHAI 行動
   */
  public static Action riichiDahai(int discardTileType) {
    return fromIndex(riichiDahaiIndex(discardTileType, TileSelection.NON_AKA));
  }

  /**
   * 赤牌選択を明示した手出しRIICHI 行動を返す。
   *
   * @param discardTileType 宣言牌の牌種ID（0-33）
   * @param usesAkaTileFromHand 赤牌を選ぶなら {@code true}
   * @return 共有の RIICHI_DAHAI 行動
   */
  public static Action riichiDahai(int discardTileType, boolean usesAkaTileFromHand) {
    return fromIndex(
        riichiDahaiIndex(
            discardTileType, usesAkaTileFromHand ? TileSelection.AKA : TileSelection.NON_AKA));
  }

  /**
   * 赤牌とツモ切りを明示したRIICHI 行動を返す。
   *
   * @param discardTileType 宣言牌の牌種ID（0-33）
   * @param usesAkaTileFromHand 赤牌を選ぶなら {@code true}
   * @param isTsumogiri ツモ切りなら {@code true}
   * @return 共有の RIICHI_DAHAI 行動
   */
  public static Action riichiDahai(
      int discardTileType, boolean usesAkaTileFromHand, boolean isTsumogiri) {
    return fromIndex(
        riichiDahaiIndex(
            discardTileType, TileSelection.forDiscard(usesAkaTileFromHand, isTsumogiri)));
  }

  /**
   * 物理牌選択を明示したRIICHI 行動を返す。
   *
   * @param discardTileType 宣言牌の牌種ID（0-33）
   * @param tileSelection 赤牌・手出し・ツモ切りの組合せ
   * @return 共有の RIICHI_DAHAI 行動
   */
  public static Action riichiDahai(int discardTileType, TileSelection tileSelection) {
    return fromIndex(riichiDahaiIndex(discardTileType, tileSelection));
  }

  /**
   * ツモ和了行動を返す。
   *
   * @return 共有される一意な TSUMO_AGARI 行動
   */
  public static Action tsumoAgari() {
    return fromIndex(OFF_TSUMO);
  }

  /**
   * ロン和了行動を返す。
   *
   * @return 共有される一意な RON_AGARI 行動
   */
  public static Action ronAgari() {
    return fromIndex(OFF_RON);
  }

  /**
   * 応答を見送るPASS 行動を返す。
   *
   * @return 共有される一意な PASS 行動
   */
  public static Action pass() {
    return fromIndex(OFF_PASS);
  }

  /**
   * 九種九牌行動を返す。
   *
   * @return 共有される一意な KYUSHU_KYUHAI 行動
   */
  public static Action kyushuKyuhai() {
    return fromIndex(OFF_KYUSHU_KYUHAI);
  }

  /**
   * エンジンで実行する行動種別を返す。
   *
   * @return 行動種別
   */
  public Type type() {
    return type;
  }

  /**
   * 行動の主牌種を返す。
   *
   * @return 打牌・鳴き・槓の対象牌種。牌を伴わない行動では {@code -1}
   */
  public int tileType() {
    return tileType;
  }

  /**
   * CHIで生成する順子3牌種を返す。
   *
   * @return 呼び出し元から変更できないよう複製した昇順3要素配列
   * @throws IllegalStateException CHI以外の行動で呼び出した場合
   */
  public int[] chiTileTypes() {
    if (type != Type.CHI) {
      throw new IllegalStateException("Only CHI carries chi tile types: " + type);
    }
    return chiTileTypes.clone();
  }

  /**
   * CHIで生成する順子の先頭牌種を返す。
   *
   * <p>エンジン内部で順子全体を走査する場合に、3要素配列の防御コピーを作らず利用する。
   *
   * @return 順子で最も小さい牌種
   * @throws IllegalStateException CHI以外の行動で呼び出した場合
   */
  public int chiSequenceBaseTileType() {
    if (type != Type.CHI) {
      throw new IllegalStateException("Only CHI carries a sequence: " + type);
    }
    return chiTileTypes[0];
  }

  /**
   * 行動が保持する物理牌選択を返す。
   *
   * @return 赤牌・手出し・ツモ切りの選択。対象外行動では {@link TileSelection#NOT_APPLICABLE}
   */
  public TileSelection tileSelection() {
    return tileSelection;
  }

  /**
   * 行動 IDだけから求まる、生成後面子への赤牌包含条件。
   *
   * <p>CHI/PONで鳴いた牌種が赤5の場合だけ、実際の鳴いた牌の物理IDが必要になる。
   *
   * @return 行動 IDだけから確定する赤牌包含条件
   */
  public MeldAkaTileInclusion meldAkaTileInclusion() {
    return meldAkaTileInclusion;
  }

  /**
   * キャッシュ済み行動を使い、行動空間 IDから赤牌包含条件をO(1)で取得する。
   *
   * @param actionIndex 固定行動 ID
   * @return 行動 IDだけから確定する赤牌包含条件
   */
  public static MeldAkaTileInclusion meldAkaTileInclusion(int actionIndex) {
    return fromIndex(actionIndex).meldAkaTileInclusion;
  }

  /**
   * 鳴いた牌の赤情報まで反映した、生成後面子の厳密な赤牌有無を返す。
   *
   * @param calledTileIsAka 他家から鳴く物理牌が赤牌なら {@code true}
   * @return 生成後面子が赤牌を含むなら {@code true}
   * @throws IllegalStateException 面子を生成しない行動で呼び出した場合
   */
  public boolean resultingMeldContainsAkaTile(boolean calledTileIsAka) {
    return meldAkaTileInclusion.resolve(calledTileIsAka);
  }

  /**
   * 行動が赤牌を明示的に選択するかを返す。完成面子の赤牌有無とは異なる。
   *
   * @return 手牌または打牌から赤牌を選ぶ行動なら {@code true}
   */
  public boolean usesAkaTileFromHand() {
    return tileSelection.usesAkaTileFromHand();
  }

  /**
   * ツモ切り行動かを返す。
   *
   * @return ツモ切りのDAHAIまたはRIICHI_DAHAIなら {@code true}
   */
  public boolean isTsumogiri() {
    return tileSelection.isTsumogiri();
  }

  /**
   * DAHAI と RIICHI_DAHAI を同じ物理打牌へ束ねる共有のキー。
   *
   * @return DAHAI 行動空間 インデックス。打牌でなければ -1。
   */
  public int discardIdentityIndex() {
    return switch (type) {
      case DAHAI, RIICHI_DAHAI -> dahaiIndex(tileType, tileSelection);
      default -> -1;
    };
  }

  // アクション空間マッピング (種別単位で配置, 372 インデックス)
  //
  // [打牌]   0- 73
  //   0- 33: 手出し           (34)
  //  34- 67: ツモ切り         (34)
  //  68- 70: 赤手出し         ( 3)
  //  71- 73: 赤ツモ切り       ( 3)
  //
  // [チー]  74-154
  //  74-136: 非赤             (63)
  // 137-154: 赤               (18)
  //
  // [ポン] 155-191
  // 155-188: 非赤             (34)
  // 189-191: 赤               ( 3)
  //
  // [槓]  192-293
  // 192-225: 大明槓           (34)
  // 226-259: 暗槓             (34)
  // 260-293: 加槓             (34)
  //
  // [リーチ] 294-367
  // 294-327: 手出し           (34)
  // 328-361: ツモ切り         (34)
  // 362-364: 赤手出し         ( 3)
  // 365-367: 赤ツモ切り       ( 3)
  //
  // [特殊] 368-371
  //     368: ツモ
  //     369: ロン
  //     370: パス
  //     371: 九種九牌

  // 各種別の先頭オフセット
  private static final int OFF_DAHAI = 0; // 34
  private static final int OFF_DAHAI_TSUMOGIRI = 34; // 34
  private static final int OFF_DAHAI_AKA = 68; //  3
  private static final int OFF_DAHAI_TSUMOGIRI_AKA = 71; //  3
  private static final int OFF_CHI = 74; // 63
  private static final int OFF_CHI_AKA = 137; // 18
  private static final int OFF_PON = 155; // 34
  private static final int OFF_PON_AKA = 189; //  3
  private static final int OFF_DAIMINKAN = 192; // 34
  private static final int OFF_ANKAN = 226; // 34
  private static final int OFF_KAKAN = 260; // 34
  private static final int OFF_RIICHI = 294; // 34
  private static final int OFF_RIICHI_TSUMOGIRI = 328; // 34
  private static final int OFF_RIICHI_AKA = 362; //  3
  private static final int OFF_RIICHI_TSUMOGIRI_AKA = 365; //  3
  private static final int OFF_TSUMO = 368;
  private static final int OFF_RON = 369;
  private static final int OFF_PASS = 370;
  private static final int OFF_KYUSHU_KYUHAI = 371;

  /**
   * この行動を固定ID空間へ符号化する。
   *
   * @return {@code 0 <= id < ACTION_SPACE_SIZE} を満たす共有の行動 ID
   */
  public int toIndex() {
    return actionIndex;
  }

  private int computeIndex() {
    return switch (type) {
      case DAHAI -> dahaiIndex(tileType, tileSelection);
      case CHI ->
          tileSelection.usesAkaTileFromHand()
              ? OFF_CHI_AKA + encodeChiIndexAka(chiTileTypes, tileType)
              : OFF_CHI + encodeChiIndex(chiTileTypes, tileType);
      case PON -> ponIndex(tileType, tileSelection.usesAkaTileFromHand());
      case DAIMINKAN, ANKAN, KAKAN -> kanIndex(type, tileType);
      case RIICHI_DAHAI -> riichiDahaiIndex(tileType, tileSelection);
      case TSUMO_AGARI -> OFF_TSUMO;
      case RON_AGARI -> OFF_RON;
      case PASS -> OFF_PASS;
      case KYUSHU_KYUHAI -> OFF_KYUSHU_KYUHAI;
    };
  }

  /**
   * 固定行動 ID空間に含まれるインデックスかを判定する。
   *
   * @param actionIndex 判定する整数
   * @return {@code 0 <= actionIndex < ACTION_SPACE_SIZE} なら {@code true}
   */
  public static boolean isValidIndex(int actionIndex) {
    return actionIndex >= 0 && actionIndex < ACTION_SPACE_SIZE;
  }

  /**
   * 固定行動 IDを事前生成済みの共有の行動へ復号する。
   *
   * @param actionIndex 固定行動 ID
   * @return プロセス内で共有される不変の行動
   * @throws IllegalArgumentException IDが固定行動空間の外なら発生
   */
  public static Action fromIndex(int actionIndex) {
    if (!isValidIndex(actionIndex)) {
      throw new IllegalArgumentException("Invalid action index: " + actionIndex);
    }
    return ActionCache.ACTIONS[actionIndex];
  }

  private static Action decodeUncached(int actionIndex) {
    // 打牌
    if (actionIndex < OFF_DAHAI_TSUMOGIRI) {
      return new Action(Type.DAHAI, actionIndex - OFF_DAHAI, null, TileSelection.NON_AKA);
    }
    if (actionIndex < OFF_DAHAI_AKA) {
      return new Action(
          Type.DAHAI, actionIndex - OFF_DAHAI_TSUMOGIRI, null, TileSelection.TSUMOGIRI);
    }
    if (actionIndex < OFF_DAHAI_TSUMOGIRI_AKA) {
      return new Action(
          Type.DAHAI, akaTileTypeFromIndex(actionIndex - OFF_DAHAI_AKA), null, TileSelection.AKA);
    }
    if (actionIndex < OFF_CHI) {
      return new Action(
          Type.DAHAI,
          akaTileTypeFromIndex(actionIndex - OFF_DAHAI_TSUMOGIRI_AKA),
          null,
          TileSelection.TSUMOGIRI_AKA);
    }
    // チー
    if (actionIndex < OFF_CHI_AKA) {
      return decodeChiUncached(actionIndex - OFF_CHI, false);
    }
    if (actionIndex < OFF_PON) {
      return decodeChiUncached(actionIndex - OFF_CHI_AKA, true);
    }
    // ポン
    if (actionIndex < OFF_PON_AKA) {
      return new Action(Type.PON, actionIndex - OFF_PON, null, TileSelection.NON_AKA);
    }
    if (actionIndex < OFF_DAIMINKAN) {
      return new Action(
          Type.PON, akaTileTypeFromIndex(actionIndex - OFF_PON_AKA), null, TileSelection.AKA);
    }
    // 槓
    if (actionIndex < OFF_ANKAN) {
      return new Action(
          Type.DAIMINKAN, actionIndex - OFF_DAIMINKAN, null, TileSelection.NOT_APPLICABLE);
    }
    if (actionIndex < OFF_KAKAN) {
      return new Action(Type.ANKAN, actionIndex - OFF_ANKAN, null, TileSelection.NOT_APPLICABLE);
    }
    if (actionIndex < OFF_RIICHI) {
      return new Action(Type.KAKAN, actionIndex - OFF_KAKAN, null, TileSelection.NOT_APPLICABLE);
    }
    // リーチ
    if (actionIndex < OFF_RIICHI_TSUMOGIRI) {
      return new Action(Type.RIICHI_DAHAI, actionIndex - OFF_RIICHI, null, TileSelection.NON_AKA);
    }
    if (actionIndex < OFF_RIICHI_AKA) {
      return new Action(
          Type.RIICHI_DAHAI, actionIndex - OFF_RIICHI_TSUMOGIRI, null, TileSelection.TSUMOGIRI);
    }
    if (actionIndex < OFF_RIICHI_TSUMOGIRI_AKA) {
      return new Action(
          Type.RIICHI_DAHAI,
          akaTileTypeFromIndex(actionIndex - OFF_RIICHI_AKA),
          null,
          TileSelection.AKA);
    }
    if (actionIndex < OFF_TSUMO) {
      return new Action(
          Type.RIICHI_DAHAI,
          akaTileTypeFromIndex(actionIndex - OFF_RIICHI_TSUMOGIRI_AKA),
          null,
          TileSelection.TSUMOGIRI_AKA);
    }
    // 特殊
    if (actionIndex == OFF_TSUMO) {
      return new Action(Type.TSUMO_AGARI, -1, null, TileSelection.NOT_APPLICABLE);
    }
    if (actionIndex == OFF_RON) {
      return new Action(Type.RON_AGARI, -1, null, TileSelection.NOT_APPLICABLE);
    }
    if (actionIndex == OFF_PASS) {
      return new Action(Type.PASS, -1, null, TileSelection.NOT_APPLICABLE);
    }
    return new Action(Type.KYUSHU_KYUHAI, -1, null, TileSelection.NOT_APPLICABLE);
  }

  private static int dahaiIndex(int tileType, TileSelection tileSelection) {
    Tile.requireValidType(tileType, "DAHAI tile");
    return switch (tileSelection) {
      case NON_AKA -> OFF_DAHAI + tileType;
      case AKA -> OFF_DAHAI_AKA + akaTileTypeIndex(tileType);
      case TSUMOGIRI -> OFF_DAHAI_TSUMOGIRI + tileType;
      case TSUMOGIRI_AKA -> OFF_DAHAI_TSUMOGIRI_AKA + akaTileTypeIndex(tileType);
      case NOT_APPLICABLE -> throw new IllegalArgumentException("DAHAI requires a tile selection");
    };
  }

  private static int riichiDahaiIndex(int tileType, TileSelection tileSelection) {
    Tile.requireValidType(tileType, "RIICHI_DAHAI tile");
    return switch (tileSelection) {
      case NON_AKA -> OFF_RIICHI + tileType;
      case AKA -> OFF_RIICHI_AKA + akaTileTypeIndex(tileType);
      case TSUMOGIRI -> OFF_RIICHI_TSUMOGIRI + tileType;
      case TSUMOGIRI_AKA -> OFF_RIICHI_TSUMOGIRI_AKA + akaTileTypeIndex(tileType);
      case NOT_APPLICABLE ->
          throw new IllegalArgumentException("RIICHI_DAHAI requires a tile selection");
    };
  }

  private static int chiActionIndex(
      int[] chiTileTypes, int calledTileType, boolean consumeAkaFromHand) {
    int[] normalizedTileTypes = normalizeChiTileTypes(chiTileTypes, calledTileType);
    return consumeAkaFromHand
        ? OFF_CHI_AKA + encodeChiIndexAka(normalizedTileTypes, calledTileType)
        : OFF_CHI + encodeChiIndex(normalizedTileTypes, calledTileType);
  }

  private static int chiSequenceActionIndex(
      int sequenceBaseTileType, int calledTileType, boolean consumeAkaFromHand) {
    requireChiSequence(sequenceBaseTileType, calledTileType);
    int calledPosition = calledTileType - sequenceBaseTileType;
    return consumeAkaFromHand
        ? OFF_CHI_AKA + encodeChiIndexAka(sequenceBaseTileType, calledPosition)
        : OFF_CHI + encodeChiIndex(sequenceBaseTileType, calledPosition);
  }

  private static int ponIndex(int tileType, boolean consumeAkaFromHand) {
    Tile.requireValidType(tileType, "PON tile");
    return consumeAkaFromHand ? OFF_PON_AKA + akaTileTypeIndex(tileType) : OFF_PON + tileType;
  }

  private static int kanIndex(Type type, int tileType) {
    if (!Tile.isValidType(tileType)) {
      throw new IllegalArgumentException(type + " tile must be in [0, 33]: " + tileType);
    }
    return switch (type) {
      case DAIMINKAN -> OFF_DAIMINKAN + tileType;
      case ANKAN -> OFF_ANKAN + tileType;
      case KAKAN -> OFF_KAKAN + tileType;
      default -> throw new IllegalArgumentException("Not a kan action type: " + type);
    };
  }

  private static int akaTileTypeIndex(int tileType) {
    if (tileType == Tile.M5) {
      return 0;
    }
    if (tileType == Tile.P5) {
      return 1;
    }
    if (tileType == Tile.S5) {
      return 2;
    }
    throw new IllegalStateException(
        "Aka tile selection requires an aka-capable tile type: " + tileType);
  }

  private static int akaTileTypeFromIndex(int akaTileTypeIndex) {
    return switch (akaTileTypeIndex) {
      case 0 -> Tile.M5;
      case 1 -> Tile.P5;
      case 2 -> Tile.S5;
      default ->
          throw new IllegalArgumentException("Invalid aka tile type index: " + akaTileTypeIndex);
    };
  }

  private static int[] normalizeChiTileTypes(int[] chiTileTypes, int calledTileType) {
    requireChiTileCount(chiTileTypes);
    Tile.requireValidType(calledTileType, "CHI calledTileType");

    int firstTileType = chiTileTypes[0];
    int secondTileType = chiTileTypes[1];
    int thirdTileType = chiTileTypes[2];
    Tile.requireValidType(firstTileType, "CHI tileType");
    Tile.requireValidType(secondTileType, "CHI tileType");
    Tile.requireValidType(thirdTileType, "CHI tileType");

    int lowestTileType = Math.min(firstTileType, Math.min(secondTileType, thirdTileType));
    int highestTileType = Math.max(firstTileType, Math.max(secondTileType, thirdTileType));
    int middleTileType =
        firstTileType + secondTileType + thirdTileType - lowestTileType - highestTileType;

    if (!isChiSequence(lowestTileType, middleTileType, highestTileType)) {
      throw new IllegalArgumentException(
          "CHI tiles must be a same-suit sequence: "
              + Tile.name(lowestTileType)
              + ","
              + Tile.name(middleTileType)
              + ","
              + Tile.name(highestTileType));
    }
    if (calledTileType != lowestTileType
        && calledTileType != middleTileType
        && calledTileType != highestTileType) {
      throw new IllegalArgumentException(
          "CHI calledTileType must be present in chiTileTypes: " + calledTileType);
    }
    return new int[] {lowestTileType, middleTileType, highestTileType};
  }

  private static void requireChiTileCount(int[] chiTileTypes) {
    if (chiTileTypes.length != 3) {
      throw new IllegalArgumentException(
          "CHI requires exactly 3 chiTileTypes, got " + chiTileTypes.length);
    }
  }

  private static boolean isChiSequence(
      int lowestTileType, int middleTileType, int highestTileType) {
    return Tile.isNumberTile(lowestTileType)
        && Tile.numberOf(lowestTileType) <= 6
        && middleTileType == lowestTileType + 1
        && highestTileType == lowestTileType + 2;
  }

  private static void requireChiSequence(int sequenceBaseTileType, int calledTileType) {
    Tile.requireValidType(sequenceBaseTileType, "CHI sequenceBaseTileType");
    Tile.requireValidType(calledTileType, "CHI calledTileType");
    if (!Tile.isNumberTile(sequenceBaseTileType) || Tile.numberOf(sequenceBaseTileType) > 6) {
      throw new IllegalArgumentException(
          "CHI sequenceBaseTileType must start a same-suit sequence: " + sequenceBaseTileType);
    }
    int calledPosition = calledTileType - sequenceBaseTileType;
    if (calledPosition < 0 || calledPosition > 2) {
      throw new IllegalArgumentException(
          "CHI calledTileType must be in the sequence: " + calledTileType);
    }
  }

  private static int encodeChiIndex(int[] chiTileTypes, int calledTileType) {
    return encodeChiIndex(chiTileTypes[0], calledTileType - chiTileTypes[0]);
  }

  private static int encodeChiIndex(int sequenceBaseTileType, int calledPosition) {
    int suitIndex = Tile.suitOf(sequenceBaseTileType);
    int sequenceStartNumber = Tile.numberOf(sequenceBaseTileType);
    return suitIndex * 21 + sequenceStartNumber * 3 + calledPosition;
  }

  private static int encodeChiIndexAka(int[] chiTileTypes, int calledTileType) {
    return encodeChiIndexAka(chiTileTypes[0], calledTileType - chiTileTypes[0]);
  }

  private static int encodeChiIndexAka(int sequenceBaseTileType, int calledPosition) {
    int suitIndex = Tile.suitOf(sequenceBaseTileType);
    int sequenceStartNumber = Tile.numberOf(sequenceBaseTileType);
    int akaPatternIndex;
    if (sequenceStartNumber == 2 && calledPosition == 0) {
      akaPatternIndex = 0;
    } else if (sequenceStartNumber == 2 && calledPosition == 1) {
      akaPatternIndex = 1;
    } else if (sequenceStartNumber == 3 && calledPosition == 0) {
      akaPatternIndex = 2;
    } else if (sequenceStartNumber == 3 && calledPosition == 2) {
      akaPatternIndex = 3;
    } else if (sequenceStartNumber == 4 && calledPosition == 1) {
      akaPatternIndex = 4;
    } else if (sequenceStartNumber == 4 && calledPosition == 2) {
      akaPatternIndex = 5;
    } else {
      throw new IllegalArgumentException(
          "Aka CHI must consume the red five from hand: sequenceBaseTileType="
              + sequenceBaseTileType
              + " calledPosition="
              + calledPosition);
    }
    return suitIndex * 6 + akaPatternIndex;
  }

  private static MeldAkaTileInclusion classifyMeldAkaTileInclusion(
      Type type, int tileType, TileSelection tileSelection) {
    if (!type.createsMeld()) {
      return MeldAkaTileInclusion.NOT_APPLICABLE;
    }
    if (tileSelection.usesAkaTileFromHand() || (type.isKan() && Tile.canBeAka(tileType))) {
      return MeldAkaTileInclusion.GUARANTEED;
    }
    if ((type == Type.CHI || type == Type.PON) && Tile.canBeAka(tileType)) {
      return MeldAkaTileInclusion.DEPENDS_ON_CALLED_TILE;
    }
    return MeldAkaTileInclusion.NONE;
  }

  private static Action decodeChiUncached(int chiIndex, boolean usesAkaTileFromHand) {
    int suitIndex;
    int sequenceStartNumber;
    int calledPosition;
    if (!usesAkaTileFromHand) {
      suitIndex = chiIndex / 21;
      int withinSuitIndex = chiIndex % 21;
      sequenceStartNumber = withinSuitIndex / 3;
      calledPosition = withinSuitIndex % 3;
    } else {
      suitIndex = chiIndex / 6;
      int akaPatternIndex = chiIndex % 6;
      switch (akaPatternIndex) {
        case 0 -> {
          sequenceStartNumber = 2;
          calledPosition = 0;
        }
        case 1 -> {
          sequenceStartNumber = 2;
          calledPosition = 1;
        }
        case 2 -> {
          sequenceStartNumber = 3;
          calledPosition = 0;
        }
        case 3 -> {
          sequenceStartNumber = 3;
          calledPosition = 2;
        }
        case 4 -> {
          sequenceStartNumber = 4;
          calledPosition = 1;
        }
        case 5 -> {
          sequenceStartNumber = 4;
          calledPosition = 2;
        }
        default -> throw new IllegalArgumentException("Invalid aka chi index: " + chiIndex);
      }
    }
    int sequenceBaseTileType = suitIndex * 9 + sequenceStartNumber;
    int[] sequenceTileTypes =
        new int[] {sequenceBaseTileType, sequenceBaseTileType + 1, sequenceBaseTileType + 2};
    return new Action(
        Type.CHI,
        sequenceTileTypes[calledPosition],
        sequenceTileTypes,
        usesAkaTileFromHand ? TileSelection.AKA : TileSelection.NON_AKA);
  }

  /**
   * 復号した値を固定行動空間の共有インスタンスへ置き換える。
   *
   * @return {@link #toIndex()} が指すプロセス内共有インスタンス
   */
  @Serial
  private Object readResolve() {
    return fromIndex(computeIndex());
  }

  private static final class ActionCache {
    private static final Action[] ACTIONS = createActions();

    private static Action[] createActions() {
      Action[] actions = new Action[ACTION_SPACE_SIZE];
      for (int actionIndex = 0; actionIndex < actions.length; actionIndex++) {
        actions[actionIndex] = decodeUncached(actionIndex);
      }
      return actions;
    }
  }

  @Override
  public boolean equals(Object object) {
    if (this == object) {
      return true;
    }
    if (!(object instanceof Action other)) {
      return false;
    }
    return type == other.type
        && tileType == other.tileType
        && tileSelection == other.tileSelection
        && Arrays.equals(chiTileTypes, other.chiTileTypes);
  }

  @Override
  public int hashCode() {
    int hash = type.hashCode();
    hash = 31 * hash + tileType;
    hash = 31 * hash + tileSelection.hashCode();
    hash = 31 * hash + Arrays.hashCode(chiTileTypes);
    return hash;
  }

  @Override
  public String toString() {
    return switch (type) {
      case DAHAI ->
          "Dahai(" + Tile.name(tileType) + (tileSelection.isTsumogiri() ? ",T" : "") + ")";
      case CHI ->
          "Chi("
              + Tile.name(chiTileTypes[0])
              + Tile.name(chiTileTypes[1])
              + Tile.name(chiTileTypes[2])
              + ")";
      case PON -> "Pon(" + Tile.name(tileType) + ")";
      case DAIMINKAN -> "Daiminkan(" + Tile.name(tileType) + ")";
      case ANKAN -> "Ankan(" + Tile.name(tileType) + ")";
      case KAKAN -> "Kakan(" + Tile.name(tileType) + ")";
      case RIICHI_DAHAI ->
          "RiichiDahai(" + Tile.name(tileType) + (tileSelection.isTsumogiri() ? ",T" : "") + ")";
      case TSUMO_AGARI -> "TsumoAgari";
      case RON_AGARI -> "RonAgari";
      case PASS -> "Pass";
      case KYUSHU_KYUHAI -> "KyushuKyuhai";
    };
  }
}
