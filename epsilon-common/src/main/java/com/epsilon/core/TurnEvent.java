package com.epsilon.core;

/** 現在の判断境界を生んだ直近イベント。局の永続状態とは分けて一件だけ保持する。 */
public sealed interface TurnEvent permits TurnEvent.None, TurnEvent.TileEvent {

  /** 判断境界に対応する直近イベントがない状態。 */
  enum None implements TurnEvent {
    /** 判断境界に対応する直近イベントがないことを表す。 */
    INSTANCE
  }

  /** ツモ牌を取得した場所。 */
  enum DrawSource {
    /** 通常の牌山。 */
    WALL,

    /** 槓成立後の嶺上牌。 */
    RINSHAN
  }

  /** 槍槓応答の原因になった槓種別。 */
  enum KanKind {
    /** 暗槓。国士無双の槍槓判定だけが応答可能。 */
    ANKAN,

    /** 加槓。 */
    KAKAN,

    /** 大明槓。 */
    DAIMINKAN
  }

  /** 判断対象となる牌を伴うイベント。 */
  sealed interface TileEvent extends TurnEvent permits Draw, ResponseSource {

    /**
     * イベント主体の席番号を返す。
     *
     * @return 席番号（0-3）
     */
    int player();

    /**
     * イベント対象牌の牌種を返す。
     *
     * @return 牌種ID（0-33）。非公開牌なら {@code -1}
     */
    int tileType();

    /**
     * イベント対象の物理牌が赤牌かを返す。
     *
     * @return 赤牌なら {@code true}
     */
    boolean isAkaTile();
  }

  /** 打牌または槓宣言に対する他家の応答元。 */
  sealed interface ResponseSource extends TileEvent permits Discard, KanAttempt {}

  /**
   * ツモ判断中の牌。
   *
   * <dl>
   *   <dt>{@code player}
   *   <dd>ツモした席番号（0-3）
   *   <dt>{@code physicalTileId}
   *   <dd>実牌ID（0-135）。非公開牌を表す場合は {@code -1}
   *   <dt>{@code drawSource}
   *   <dd>通常の牌山または嶺上牌のどちらから引いたか
   *   <dt>{@code doraRevealPending}
   *   <dd>この打牌後に保留中の槓ドラを表示するなら {@code true}
   * </dl>
   */
  final class Draw implements TileEvent {

    private int player;
    private int physicalTileId;
    private DrawSource drawSource;
    private boolean doraRevealPending;

    public Draw(int player, int physicalTileId, DrawSource drawSource, boolean doraRevealPending) {
      set(player, physicalTileId, drawSource, doraRevealPending);
    }

    Draw set(int player, int physicalTileId, DrawSource drawSource, boolean doraRevealPending) {
      this.player = player;
      this.physicalTileId = physicalTileId;
      this.drawSource = drawSource;
      this.doraRevealPending = doraRevealPending;
      return this;
    }

    @Override
    public int player() {
      return player;
    }

    public int physicalTileId() {
      return physicalTileId;
    }

    public DrawSource drawSource() {
      return drawSource;
    }

    public boolean doraRevealPending() {
      return doraRevealPending;
    }

    @Override
    public int tileType() {
      return physicalTileId >= 0 ? Tile.typeOf(physicalTileId) : -1;
    }

    @Override
    public boolean isAkaTile() {
      return physicalTileId >= 0 && Tile.isAka(physicalTileId);
    }

    /**
     * 嶺上牌からのツモかを返す。
     *
     * @return 嶺上ツモなら {@code true}
     */
    public boolean isRinshanDraw() {
      return drawSource == DrawSource.RINSHAN;
    }
  }

  /**
   * 河への打牌。
   *
   * <dl>
   *   <dt>{@code player}
   *   <dd>打牌した席番号（0-3）
   *   <dt>{@code tileType}
   *   <dd>捨てられた牌種ID（0-33）
   *   <dt>{@code isAkaTile}
   *   <dd>捨てられた物理牌が赤牌なら {@code true}
   * </dl>
   */
  final class Discard implements ResponseSource {

    private int player;
    private int tileType;
    private boolean akaTile;

    public Discard(int player, int tileType, boolean isAkaTile) {
      set(player, tileType, isAkaTile);
    }

    Discard set(int player, int tileType, boolean isAkaTile) {
      this.player = player;
      this.tileType = tileType;
      this.akaTile = isAkaTile;
      return this;
    }

    @Override
    public int player() {
      return player;
    }

    @Override
    public int tileType() {
      return tileType;
    }

    @Override
    public boolean isAkaTile() {
      return akaTile;
    }
  }

  /**
   * 槍槓判断中の槓宣言。
   *
   * <dl>
   *   <dt>{@code player}
   *   <dd>槓を宣言した席番号（0-3）
   *   <dt>{@code tileType}
   *   <dd>槓対象の牌種ID（0-33）
   *   <dt>{@code isAkaTile}
   *   <dd>槍槓対象として公開された追加牌が赤牌なら {@code true}
   *   <dt>{@code kanKind}
   *   <dd>宣言された槓の種類
   *   <dt>{@code doraRevealPending}
   *   <dd>応答解決後に槓ドラ表示が必要なら {@code true}
   * </dl>
   */
  final class KanAttempt implements ResponseSource {

    private int player;
    private int tileType;
    private boolean akaTile;
    private KanKind kanKind;
    private boolean doraRevealPending;

    public KanAttempt(
        int player, int tileType, boolean isAkaTile, KanKind kanKind, boolean doraRevealPending) {
      set(player, tileType, isAkaTile, kanKind, doraRevealPending);
    }

    KanAttempt set(
        int player, int tileType, boolean isAkaTile, KanKind kanKind, boolean doraRevealPending) {
      this.player = player;
      this.tileType = tileType;
      this.akaTile = isAkaTile;
      this.kanKind = kanKind;
      this.doraRevealPending = doraRevealPending;
      return this;
    }

    @Override
    public int player() {
      return player;
    }

    @Override
    public int tileType() {
      return tileType;
    }

    @Override
    public boolean isAkaTile() {
      return akaTile;
    }

    public KanKind kanKind() {
      return kanKind;
    }

    public boolean doraRevealPending() {
      return doraRevealPending;
    }
  }
}
