package com.epsilon.core;

/** 麻雀の局全体の状態を管理するクラス。 4人のプレイヤー情報、山、ドラ、局情報を保持する。 */
public final class GameState {

  /** 4人打ちで固定されるプレイヤー数。 */
  public static final int NUM_PLAYERS = 4;

  /** 王牌14枚を除く、局開始時に通常ツモ可能な牌数。 */
  public static final int LIVE_WALL_SIZE = 122;

  /** 半荘東南戦の局数（東1〜南4）。 */
  public static final int HANCHAN_KYOKU_COUNT = 8;

  // 状態を直接読む符号化処理やルール判定のための内部アクセサー。既存の取得メソッドと同じフィールドを返す。
  public int currentPlayer() {
    return currentPlayer;
  }

  public int relativePosition(int observer, int target) {
    return getRelativePosition(observer, target);
  }

  public int roundIndex() {
    return kyokuIndex;
  }

  public int dealer() {
    return oya;
  }

  public int honba() {
    return honba;
  }

  public int riichiSticks() {
    return kyotakuCount;
  }

  public int roundWindTileType() {
    return getBakaze();
  }

  public int seatWindTileType(int player) {
    return getJikaze(player);
  }

  public int turnNumber() {
    return getTurnNumber();
  }

  public int score(int player) {
    return scores[player];
  }

  public boolean isBeforeFirstDiscard(int player) {
    return isFirstDraw(player);
  }

  public boolean firstTurnCallOccurred() {
    return isFirstTurnCallOccurred();
  }

  public TurnEvent turnEvent() {
    return turnEvent;
  }

  // セクション: 局情報
  private int kyokuIndex; // 局番号 (0=東1局, 1=東2局, ..., 7=南4局)
  private int oya; // 親のプレイヤーインデックス (0-3)
  private int honba; // 本場
  private int kyotakuCount; // 供託リーチ棒数

  // セクション: プレイヤー情報
  private final Hand[] hands; // 各プレイヤーの手牌
  private final River[] rivers; // 各プレイヤーの河
  private final int[] scores; // 各プレイヤーの点数
  private final boolean[] riichi; // リーチ状態
  private final boolean[] doubleRiichi; // ダブルリーチ状態（宣言時に確定）
  private final boolean[] ippatsu; // 一発フラグ
  private final boolean[] temporaryFuriten;
  private final RoundPublicStateIndex publicStateIndex;
  private final DoraState doraState;

  // セクション: 山
  private final Wall wall;
  private final PublicObservation[] observations = new PublicObservation[NUM_PLAYERS];

  /** 停止中の局面を特定席の公開情報として借用する。 */
  public PublicObservation publicObservation(int player) {
    PublicObservation observation = observations[player];
    if (observation == null)
      observations[player] = observation = new PublicObservation(this, player);
    return observation;
  }

  // セクション: ターン情報
  private int currentPlayer; // 現在の手番プレイヤー
  private int dahaiCount; // 通常打牌カウンタ (巡目計算用)
  private int publicEventCount; // 打牌・面子・槓の公開記録に共通するイベント連番。加槓は槍槓応答前の宣言時に記録する。
  private final TurnEvent.Draw drawEvent =
      new TurnEvent.Draw(-1, -1, TurnEvent.DrawSource.WALL, false);
  private final TurnEvent.Discard discardEvent = new TurnEvent.Discard(-1, -1, false);
  private final TurnEvent.KanAttempt kanAttemptEvent =
      new TurnEvent.KanAttempt(-1, -1, false, TurnEvent.KanKind.ANKAN, false);
  private TurnEvent turnEvent = TurnEvent.None.INSTANCE;

  /** 通常の乱数で初期化される牌山を持つ対局状態を作る。 */
  public GameState() {
    this(Wall.forGameState());
  }

  /**
   * 決定的な牌山シードを持つ対局状態を作る。
   *
   * @param wallSeed 牌山シャッフルのシード
   */
  public GameState(long wallSeed) {
    this(Wall.forGameState(wallSeed));
  }

  private GameState(Wall wall) {
    hands = new Hand[NUM_PLAYERS];
    rivers = new River[NUM_PLAYERS];
    scores = new int[NUM_PLAYERS];
    riichi = new boolean[NUM_PLAYERS];
    doubleRiichi = new boolean[NUM_PLAYERS];
    ippatsu = new boolean[NUM_PLAYERS];
    temporaryFuriten = new boolean[NUM_PLAYERS];
    publicStateIndex = new RoundPublicStateIndex();
    doraState = new DoraState();
    this.wall = wall;

    for (int player = 0; player < NUM_PLAYERS; player++) {
      hands[player] = new Hand();
      rivers[player] = new River();
      scores[player] = 25000;
    }
  }

  /**
   * 局状態、手牌、河、牌山を内部の可変データも含めて複製する。
   *
   * @param other 複製元の状態
   */
  private GameState(GameState other) {
    this.kyokuIndex = other.kyokuIndex;
    this.oya = other.oya;
    this.honba = other.honba;
    this.kyotakuCount = other.kyotakuCount;

    this.hands = new Hand[NUM_PLAYERS];
    this.rivers = new River[NUM_PLAYERS];
    this.scores = other.scores.clone();
    this.riichi = other.riichi.clone();
    this.doubleRiichi = other.doubleRiichi.clone();
    this.ippatsu = other.ippatsu.clone();
    this.temporaryFuriten = other.temporaryFuriten.clone();
    this.publicStateIndex = new RoundPublicStateIndex(other.publicStateIndex);
    this.doraState = new DoraState(other.doraState);

    for (int player = 0; player < NUM_PLAYERS; player++) {
      this.hands[player] = other.hands[player].snapshotCopy();
      this.rivers[player] = new River(other.rivers[player]);
    }

    this.wall = new Wall(other.wall);

    this.currentPlayer = other.currentPlayer;
    this.dahaiCount = other.dahaiCount;
    this.publicEventCount = other.publicEventCount;
    this.turnEvent =
        switch (other.turnEvent) {
          case TurnEvent.Draw draw ->
              drawEvent.set(
                  draw.player(),
                  draw.physicalTileId(),
                  draw.drawSource(),
                  draw.doraRevealPending());
          case TurnEvent.Discard discard ->
              discardEvent.set(discard.player(), discard.tileType(), discard.isAkaTile());
          case TurnEvent.KanAttempt kan ->
              kanAttemptEvent.set(
                  kan.player(),
                  kan.tileType(),
                  kan.isAkaTile(),
                  kan.kanKind(),
                  kan.doraRevealPending());
          case TurnEvent.None ignored -> TurnEvent.None.INSTANCE;
        };
  }

  /** スナップショットの保存や対局の分岐に使う、元の状態と独立したコピー。 */
  public GameState snapshotCopy() {
    return new GameState(this);
  }

  /**
   * 進行情報と局内状態を初期化し、新しい牌山をシャッフルする。
   *
   * @param kyokuIndex 東1を0とする通算局番号
   * @param oya 親の席
   * @param honba 本場数
   * @param kyotakuCount 供託リーチ棒数
   */
  public void startRound(int kyokuIndex, int oya, int honba, int kyotakuCount) {
    resetRound(kyokuIndex, oya, honba, kyotakuCount);
    wall.init();
    revealDoraFromWall();
  }

  /**
   * 牌譜再生用に局内状態だけを初期化し、牌山のシャッフルは行わない。
   *
   * <p>呼び出し側は別途 {@link #initializeWallForReconstruction(int[], int)} で牌譜再生用牌山を設定する。
   *
   * @param kyokuIndex 東1を0とする通算局番号
   * @param oya 親の席
   * @param honba 本場数
   * @param kyotakuCount 供託リーチ棒数
   */
  public void startRoundForReconstruction(int kyokuIndex, int oya, int honba, int kyotakuCount) {
    resetRound(kyokuIndex, oya, honba, kyotakuCount);
  }

  private void resetRound(int kyokuIndex, int oya, int honba, int kyotakuCount) {
    this.kyokuIndex = kyokuIndex;
    this.oya = oya;
    this.honba = honba;
    this.kyotakuCount = kyotakuCount;

    for (int player = 0; player < NUM_PLAYERS; player++) {
      hands[player].clear();
      rivers[player].clear();
      riichi[player] = false;
      doubleRiichi[player] = false;
      ippatsu[player] = false;
      temporaryFuriten[player] = false;
    }

    currentPlayer = oya;
    dahaiCount = 0;
    publicEventCount = 0;
    turnEvent = TurnEvent.None.INSTANCE;
    publicStateIndex.reset();
    doraState.reset();
  }

  /** 配牌を行う。 */
  public void dealInitialHands() {
    // 各プレイヤーに4枚×3回 = 12枚
    for (int round = 0; round < 3; round++) {
      for (int playerOffset = 0; playerOffset < NUM_PLAYERS; playerOffset++) {
        int player = (oya + playerOffset) % NUM_PLAYERS;
        for (int packetTileIndex = 0; packetTileIndex < 4; packetTileIndex++) {
          int physicalTileId = wall.draw();
          hands[player].addPhysicalTile(physicalTileId);
        }
      }
    }
    // 各プレイヤーに1枚ずつ = 合計13枚
    for (int playerOffset = 0; playerOffset < NUM_PLAYERS; playerOffset++) {
      int player = (oya + playerOffset) % NUM_PLAYERS;
      int physicalTileId = wall.draw();
      hands[player].addPhysicalTile(physicalTileId);
    }
  }

  /**
   * 指定プレイヤーの自風牌種を返す。
   *
   * @param player 対象席
   * @return {@link Tile#TON} から始まる自風牌種
   */
  public int getJikaze(int player) {
    int offset = (player - oya + NUM_PLAYERS) % NUM_PLAYERS;
    return Tile.TON + offset;
  }

  /**
   * 指定プレイヤーから見た相対位置を返す。
   *
   * @param observerPlayer 観測基準の席
   * @param targetPlayer 対象席
   * @return 同席なら0、下家1、対面2、上家3
   */
  public int getRelativePosition(int observerPlayer, int targetPlayer) {
    return (targetPlayer - observerPlayer + NUM_PLAYERS) % NUM_PLAYERS;
  }

  /**
   * 現在の場風牌種を返す。
   *
   * @return 東場なら {@link Tile#TON}、南場なら南
   */
  public int getBakaze() {
    return Tile.TON + kyokuIndex / NUM_PLAYERS;
  }

  /**
   * 東1を0とする通算局番号を返す。
   *
   * @return 現在の局番号
   */
  public int getKyokuIndex() {
    return kyokuIndex;
  }

  /**
   * 現在の親席を返す。
   *
   * @return 親のプレイヤーインデックス
   */
  public int getOya() {
    return oya;
  }

  /**
   * 現在の本場数を返す。
   *
   * @return 本場数
   */
  public int getHonba() {
    return honba;
  }

  /**
   * 現在の供託リーチ棒数を返す。
   *
   * @return 供託数
   */
  public int getKyotakuCount() {
    return kyotakuCount;
  }

  /**
   * 精算後の供託リーチ棒数を設定する。
   *
   * @param kyotakuCount 新しい供託数
   */
  public void setKyotakuCount(int kyotakuCount) {
    this.kyotakuCount = kyotakuCount;
  }

  /**
   * 指定プレイヤーの変更可能手牌を返す。
   *
   * @param player 対象席
   * @return エンジンが所有する手牌
   */
  public Hand hand(int player) {
    return hands[player];
  }

  /** AI入力と局内判定で共有する増分公開局面インデックスを返す。 */
  public RoundPublicStateIndex publicState() {
    return publicStateIndex;
  }

  /** 局内のドラ状態を管理するオブジェクトを返す。非公開の裏ドラ情報を含むため、AIへ直接渡してはならない。 */
  public DoraState doraState() {
    return doraState;
  }

  /**
   * 指定プレイヤーの変更可能河を返す。
   *
   * @param player 対象席
   * @return エンジンが所有する河
   */
  public River river(int player) {
    return rivers[player];
  }

  /**
   * 指定プレイヤーの現在得点を返す。
   *
   * @param player 対象席
   * @return 点棒数
   */
  public int getScore(int player) {
    return scores[player];
  }

  /**
   * 指定プレイヤーの得点を置き換える。
   *
   * @param player 対象席
   * @param score 新しい点棒数
   */
  public void setScore(int player, int score) {
    scores[player] = score;
  }

  /**
   * 指定プレイヤーの得点へ増減を加える。
   *
   * @param player 対象席
   * @param delta 加算する点棒。支払いは負値
   */
  public void addScore(int player, int delta) {
    scores[player] += delta;
  }

  /**
   * 指定プレイヤーのリーチが成立済みかを返す。
   *
   * @param player 対象席
   * @return リーチ中なら {@code true}
   */
  public boolean isRiichi(int player) {
    return riichi[player];
  }

  /**
   * 指定プレイヤーのリーチ状態を設定する。
   *
   * @param player 対象席
   * @param value 新しいリーチ状態
   */
  public void setRiichi(int player, boolean value) {
    riichi[player] = value;
    publicStateIndex.setRiichi(player, value || doubleRiichi[player]);
  }

  /**
   * 指定プレイヤーのリーチがダブルリーチかを返す。
   *
   * @param player 対象席
   * @return ダブルリーチなら {@code true}
   */
  public boolean isDoubleRiichi(int player) {
    return doubleRiichi[player];
  }

  /**
   * 指定プレイヤーのダブルリーチ状態を設定する。
   *
   * @param player 対象席
   * @param value 新しいダブルリーチ状態
   */
  public void setDoubleRiichi(int player, boolean value) {
    doubleRiichi[player] = value;
    publicStateIndex.setRiichi(player, value || riichi[player]);
  }

  /**
   * 指定プレイヤーの一発権が残っているかを返す。
   *
   * @param player 対象席
   * @return 一発が有効なら {@code true}
   */
  public boolean isIppatsu(int player) {
    return ippatsu[player];
  }

  /**
   * 指定プレイヤーの一発状態を設定する。
   *
   * @param player 対象席
   * @param value 新しい一発状態
   */
  public void setIppatsu(int player, boolean value) {
    ippatsu[player] = value;
  }

  /**
   * 指定プレイヤーがまだ第一打を行っていないかを返す。
   *
   * @param player 対象席
   * @return 河が空なら {@code true}
   */
  public boolean isFirstDraw(int player) {
    return rivers[player].size() == 0;
  }

  /**
   * 指定プレイヤーが同巡内フリテン中かを返す。
   *
   * @param player 対象席
   * @return 一時フリテンなら {@code true}
   */
  public boolean isTemporaryFuriten(int player) {
    return temporaryFuriten[player];
  }

  /**
   * 指定プレイヤーを同巡内フリテンへ移す。
   *
   * @param player 対象席
   */
  public void enterTemporaryFuriten(int player) {
    temporaryFuriten[player] = true;
  }

  /** 槓による短縮を含む王牌を除く牌山の残り枚数。 */
  public int remainingWallTiles() {
    return wall.remaining();
  }

  /** 王牌を除く牌山を使い切ったかを返す。 */
  public boolean isWallExhausted() {
    return wall.isExhausted();
  }

  /** 記録処理やスナップショットの作成時に、公開済みのドラ表示牌を物理牌IDで取得する。 */
  public int doraIndicatorPhysicalTileId(int index) {
    requireRevealedDoraIndex(index);
    return wall.doraIndicatorPhysicalTileIdAt(index);
  }

  /** 記録処理やスナップショットの作成時に、公開済みの裏ドラ表示牌を物理牌IDで取得する。 */
  public int uraDoraIndicatorPhysicalTileId(int index) {
    requireRevealedDoraIndex(index);
    return wall.uraDoraIndicatorPhysicalTileIdAt(index);
  }

  private void requireRevealedDoraIndex(int index) {
    if (index < 0 || index >= doraState.indicatorCount())
      throw new IndexOutOfBoundsException(index);
  }

  /** 王牌を除く牌山から次の物理牌を取り出す。 */
  public int drawLiveWallTile() {
    return wall.draw();
  }

  /** 王牌から次の嶺上物理牌を取り出し、王牌を除く牌山を一枚短縮する。 */
  public int drawRinshanTile() {
    return wall.drawFromDeadWall();
  }

  /** 牌譜再構成で、牌を参照せず王牌を除く牌山カーソルを一枚進める。 */
  public void consumeReconstructionLiveWallDraw() {
    wall.consumeDraw();
  }

  /** 牌譜再構成で、牌を参照せず嶺上カーソルと王牌を除く牌山上限を一枚進める。 */
  public void consumeReconstructionRinshanDraw() {
    wall.consumeDeadWallDraw();
  }

  /**
   * 現在手番のプレイヤーを返す。
   *
   * @return 現在の手番の席
   */
  public int getCurrentPlayer() {
    return currentPlayer;
  }

  /**
   * 現在手番を指定プレイヤーへ設定する。
   *
   * @param player 新たに手番となる席
   */
  public void setCurrentPlayer(int player) {
    this.currentPlayer = player;
  }

  /**
   * 通常打牌4回を1巡とした 0始まりの巡目を返す。
   *
   * @return 現在の巡目
   */
  public int getTurnNumber() {
    return dahaiCount / NUM_PLAYERS;
  }

  /**
   * 鳴かれた牌を含む全河の打牌枚数合計を返す。
   *
   * <p>次に打たれる牌の局内通し連番として使う。
   *
   * @return 全プレイヤーの河要素数合計
   */
  public int totalRiverDahaiCount() {
    return publicStateIndex.totalDiscardCount();
  }

  /** 通常打牌（鳴き後打牌を除く）を記録し、4打牌ごとに巡目を進める。 */
  public void recordNormalDahai() {
    dahaiCount++;
  }

  /**
   * 現在の自摸・打牌・槓の宣言の情報を返す。
   *
   * @return 最新の手番イベント
   */
  public TurnEvent getTurnEvent() {
    return turnEvent;
  }

  /**
   * 自摸を現在の手番イベントとして記録する。
   *
   * @param player 自摸した席
   * @param physicalTileId 自摸牌の物理 ID
   * @param drawSource 通常山または嶺上
   * @param doraRevealPending 打牌後に新ドラ表示が必要なら {@code true}
   * @return 記録した自摸イベント
   */
  public TurnEvent.Draw recordDraw(
      int player, int physicalTileId, TurnEvent.DrawSource drawSource, boolean doraRevealPending) {
    TurnEvent.Draw draw = drawEvent.set(player, physicalTileId, drawSource, doraRevealPending);
    turnEvent = draw;
    return draw;
  }

  /**
   * 槍槓の判定前に、槓宣言を現在の手番イベントとして記録する。
   *
   * @param player 槓を宣言した席
   * @param tileType 槓子の牌種
   * @param isAkaTile 対象牌が赤牌なら {@code true}
   * @param kanKind 暗槓または加槓の種別
   * @return 記録した槓の宣言
   */
  public TurnEvent.KanAttempt recordKanAttempt(
      int player, int tileType, boolean isAkaTile, TurnEvent.KanKind kanKind) {
    boolean doraRevealPending =
        turnEvent instanceof TurnEvent.Draw draw && draw.doraRevealPending();
    TurnEvent.KanAttempt kanAttempt =
        kanAttemptEvent.set(player, tileType, isAkaTile, kanKind, doraRevealPending);
    turnEvent = kanAttempt;
    return kanAttempt;
  }

  /** 現在の手番イベントを {@link TurnEvent.None} へ戻す。 */
  public void clearTurnEvent() {
    turnEvent = TurnEvent.None.INSTANCE;
  }

  /**
   * いずれかのプレイヤーがまだ第一打を行っていないかを返す。
   *
   * @return 全員の第一打が済む前なら {@code true}
   */
  public boolean isFirstTurn() {
    for (River river : rivers) {
      if (river.size() == 0) {
        return true;
      }
    }
    return false;
  }

  /**
   * 第一巡中に鳴きまたは槓が既に発生したかを返す。
   *
   * @return いずれかの手牌が面子を持てば {@code true}
   */
  public boolean isFirstTurnCallOccurred() {
    for (Hand hand : hands) {
      if (hand.meldCount() != 0) {
        return true;
      }
    }
    return false;
  }

  /** 次のプレイヤーに手番を移す。 */
  public void advancePlayer() {
    currentPlayer = (currentPlayer + 1) % NUM_PLAYERS;
  }

  /** 全プレイヤーの一発フラグを解除。 */
  public void clearAllIppatsu() {
    for (int player = 0; player < NUM_PLAYERS; player++) {
      ippatsu[player] = false;
    }
  }

  /**
   * 場の4槓上限未満かつ海底前で、追加の槓が可能かを返す。
   *
   * @return 槓可能なら {@code true}
   */
  public boolean canKan() {
    return publicStateIndex.totalKanCount() < 4 && wall.remaining() > 0;
  }

  /**
   * 打牌を河、公開局面インデックス、現在手番イベントへ不可分に反映する。
   *
   * <p>非リーチプレイヤーの一時フリテン解除も同じ処理で行う。
   */
  public TurnEvent.Discard commitDiscard(
      int player,
      int tileType,
      int turnNumber,
      boolean riichiDeclaration,
      boolean aka,
      boolean tsumogiri,
      int sequence) {
    River river = rivers[player];
    river.append(
        tileType, turnNumber, riichiDeclaration, aka, tsumogiri, sequence, publicEventCount);
    publicEventCount++;
    publicStateIndex.addDiscard(player, river.discard(river.size() - 1));
    if (!riichi[player]) {
      temporaryFuriten[player] = false;
    }
    TurnEvent.Discard discard = discardEvent.set(player, tileType, aka);
    turnEvent = discard;
    return discard;
  }

  /** テスト・手動再構築用に現在値から時系列付加情報を補って打牌を反映する。 */
  public TurnEvent.Discard commitDiscard(int player, int tileType, boolean aka) {
    return commitDiscard(
        player, tileType, getTurnNumber(), false, aka, false, totalRiverDahaiCount());
  }

  /** 最後の打牌を鳴かれた状態へ移し、可視牌の二重計上を防ぐ。 */
  public void markLastDiscardCalled(int player) {
    River river = rivers[player];
    River.Discard discard = river.discard(river.size() - 1);
    if (!discard.called()) {
      river.markLastCalled();
      publicStateIndex.markDiscardCalled(discard);
    }
  }

  /** 指定席の手牌へ面子を追加し、公開局面インデックスを同時に更新する。 */
  public void addMeld(int player, Meld meld, int afterRiverIndex) {
    hands[player].addMeld(meld, afterRiverIndex, publicEventCount);
    publicEventCount++;
    publicStateIndex.addMeld(player, meld);
  }

  /** 指定席のポンを加槓へ置換し、追加公開牌と槓数を更新する。 */
  public void replacePonWithKakan(int player, Meld pon, Meld kakan, int kanAfterRiverIndex) {
    hands[player].replacePonWithKakan(pon, kakan, kanAfterRiverIndex, publicEventCount);
    publicEventCount++;
    publicStateIndex.replacePonWithKakan(player, kakan);
  }

  /** 牌譜再生用の完全牌山を設定する。 */
  public void initializeWall(int[] physicalTileIds) {
    clearDoraVisibility();
    wall.initWithFullWall(physicalTileIds);
    revealDoraFromWall();
  }

  /** 公開情報だけを持つ牌譜再生牌山を設定する。 */
  public void initializeWallForReconstruction(int[] doraIndicatorTileTypes, int drawIndex) {
    clearDoraVisibility();
    wall.initForReconstruction(drawIndex);
    for (int indicatorTileType : doraIndicatorTileTypes) {
      revealReconstructedDora(indicatorTileType);
    }
  }

  /** 次のドラ表示牌を公開し、公開局面インデックスを同時に更新する。 */
  public void revealDoraIndicator() {
    revealDoraFromWall();
  }

  /** 牌譜再生で指定されたドラ表示牌を公開する。 */
  public void revealDoraIndicator(int indicatorTileType) {
    revealReconstructedDora(indicatorTileType);
  }

  /** 牌譜で確定した表示牌の赤識別も公開状態へ反映する。 */
  public void revealDoraIndicator(int indicatorTileType, boolean indicatorIsAka) {
    revealReconstructedDora(indicatorTileType, indicatorIsAka);
  }

  private void revealDoraFromWall() {
    int slot = doraState.indicatorCount();
    int indicatorPhysicalTileId = wall.doraIndicatorPhysicalTileIdAt(slot);
    int uraIndicatorPhysicalTileId = wall.uraDoraIndicatorPhysicalTileIdAt(slot);
    revealDora(
        Tile.typeOf(indicatorPhysicalTileId),
        Tile.isAka(indicatorPhysicalTileId),
        Tile.typeOf(uraIndicatorPhysicalTileId),
        true);
  }

  private void revealReconstructedDora(int indicatorTileType) {
    revealReconstructedDora(indicatorTileType, false);
  }

  private void revealReconstructedDora(int indicatorTileType, boolean indicatorIsAka) {
    int slot = doraState.indicatorCount();
    wall.setReconstructedDoraIndicator(slot, indicatorTileType, indicatorIsAka);
    revealDora(indicatorTileType, indicatorIsAka, 0, false);
  }

  private void revealDora(
      int indicatorTileType, boolean indicatorIsAka, int uraIndicatorTileType, boolean uraKnown) {
    doraState.reveal(indicatorTileType, indicatorIsAka, uraIndicatorTileType, uraKnown);
    publicStateIndex.addDoraIndicatorVisibility(indicatorTileType);
  }

  private void clearDoraVisibility() {
    for (int index = 0; index < doraState.indicatorCount(); index++) {
      publicStateIndex.removeDoraIndicatorVisibility(doraState.indicatorTileType(index));
    }
    doraState.reset();
  }
}
