package com.epsilon.client.tenhou;

/**
 * 天鳳プロトコルのイベント型。
 *
 * <p>{@code physicalTileId} は実牌ID (0-135)、{@code tileType} は牌種ID (0-33)。
 */
public sealed interface TenhouEvent {

  /** 局状態を更新するイベント。接続・ロビー制御イベントは含まない。 */
  sealed interface GameEvent extends TenhouEvent
      permits Init, Draw, Dahai, Naki, ReachDeclared, ReachAccepted, NewDora, Agari, Ryukyoku {}

  // 認証・ロビー
  /**
   * HELO 応答。
   *
   * @param auth サーバーが返した認証トークン
   */
  record Helo(String auth) implements TenhouEvent {}

  /**
   * GO メッセージ。
   *
   * @param type 天鳳プロトコルの対局種別ビットマスク
   * @param lobby ロビー番号
   */
  record Go(int type, int lobby) implements TenhouEvent {}

  /**
   * UN メッセージ。
   *
   * @param names 席番号順のURL 符号化済みプレイヤー名
   */
  record Un(String[] names) implements TenhouEvent {}

  // 局開始
  /**
   * INIT メッセージ。局開始情報。
   *
   * @param roundIndex 東1局=0、東2局=1、…
   * @param honba 本場
   * @param kyotaku 供託本数
   * @param dealer 親プレイヤー番号
   * @param doraIndicatorTileType ドラ表示牌の牌種ID
   * @param scores 点単位の持ち点
   * @param initialHandPhysicalTileIds 自分の手牌の実牌ID配列
   */
  record Init(
      int roundIndex,
      int honba,
      int kyotaku,
      int dealer,
      int doraIndicatorTileType,
      int[] scores,
      int[] initialHandPhysicalTileIds)
      implements GameEvent {}

  // ゲーム進行
  /**
   * ツモ（ドロー）。
   *
   * @param player ツモした席番号（0-3）
   * @param physicalTileId 自家なら実牌ID（0-135）、非公開の他家牌なら {@code -1}
   * @param prompt このツモに付随するアクション要求
   */
  record Draw(int player, int physicalTileId, ActionPrompt prompt) implements GameEvent {
    /**
     * 行動要求を伴わない自摸イベントを生成する。
     *
     * @param player 自摸した席番号
     * @param physicalTileId 自家の実牌 ID。他家の非公開牌なら {@code -1}
     */
    public Draw(int player, int physicalTileId) {
      this(player, physicalTileId, ActionPrompt.none());
    }
  }

  /**
   * 打牌（ディスカード）。
   *
   * @param player 打牌した席番号（0-3）
   * @param physicalTileId 捨てられた実牌ID（0-135）
   * @param tsumogiri ツモ切りなら {@code true}
   * @param prompt この打牌に付随する応答要求
   */
  record Dahai(int player, int physicalTileId, boolean tsumogiri, ActionPrompt prompt)
      implements GameEvent {
    /**
     * 行動要求を伴わない打牌イベントを生成する。
     *
     * @param player 打牌した席番号
     * @param physicalTileId 捨てられた実牌 ID
     * @param tsumogiri 自摸切りなら {@code true}
     */
    public Dahai(int player, int physicalTileId, boolean tsumogiri) {
      this(player, physicalTileId, tsumogiri, ActionPrompt.none());
    }
  }

  /**
   * デコード済みの鳴き。
   *
   * @param player 鳴いた席番号（0-3）
   * @param meld プロトコル値から復元した面子
   * @param prompt 鳴き後の打牌など、このイベントに付随するアクション要求
   */
  record Naki(int player, TenhouMeldDecoder.DecodedMeld meld, ActionPrompt prompt)
      implements GameEvent {
    /**
     * 行動要求を伴わない副露イベントを生成する。
     *
     * @param player 鳴いた席番号
     * @param meld プロトコル値から復元した面子
     */
    public Naki(int player, TenhouMeldDecoder.DecodedMeld meld) {
      this(player, meld, ActionPrompt.none());
    }
  }

  /**
   * リーチ宣言。直後の打牌が宣言牌になる。
   *
   * @param player 宣言した席番号（0-3）
   */
  record ReachDeclared(int player) implements GameEvent {}

  /**
   * リーチ成立。
   *
   * @param player 成立した席番号（0-3）
   * @param scores 点単位の席番号順スコアスナップショット。プロトコルで省略された場合は {@code null}
   */
  record ReachAccepted(int player, int[] scores) implements GameEvent {}

  /**
   * 新ドラ表示牌。
   *
   * @param indicatorTileType ドラ表示牌の牌種ID（0-33）
   */
  record NewDora(int indicatorTileType) implements GameEvent {}

  /**
   * 和了。
   *
   * @param winner 和了者の席番号（0-3）
   * @param fromPlayer 放銃者の席番号。ツモ和了では和了者と同じ
   * @param scores 精算後の点単位スコアを席番号順に並べた配列
   * @param gameEnd この和了で対局全体が終了するなら {@code true}
   */
  record Agari(int winner, int fromPlayer, int[] scores, boolean gameEnd) implements GameEvent {
    /**
     * 対局が継続する和了イベントを生成する。
     *
     * @param winner 和了者の席番号
     * @param fromPlayer 放銃者の席番号。自摸和了では和了者と同じ
     * @param scores 精算後の席別持ち点
     */
    public Agari(int winner, int fromPlayer, int[] scores) {
      this(winner, fromPlayer, scores, false);
    }
  }

  /**
   * 流局。
   *
   * @param scores 精算後の点単位スコアを席番号順に並べた配列
   * @param gameEnd この流局で対局全体が終了するなら {@code true}
   */
  record Ryukyoku(int[] scores, boolean gameEnd) implements GameEvent {
    /**
     * 対局が継続する流局イベントを生成する。
     *
     * @param scores 精算後の席別持ち点
     */
    public Ryukyoku(int[] scores) {
      this(scores, false);
    }
  }

  /** アクション要求が属するプロトコルフェーズ。 */
  enum PromptPhase {
    /** 行動要求なし。 */
    NONE,
    /** 自家の自摸番に対する要求。 */
    DRAW,
    /** 他家の打牌または副露に対する応答要求。 */
    RESPONSE
  }

  /**
   * Draw/Dahai/Naki に付随するサーバーのアクション要求。
   *
   * @param phase 要求が属するツモ番または応答フェーズ
   * @param mask サーバープロトコルが通知した合法アクションビットマスク
   */
  record ActionPrompt(PromptPhase phase, int mask) {
    private static final int PON = 1;
    private static final int KAN = 2;
    private static final int CHI = 4;
    private static final int RON = 8;
    private static final int TSUMO = 16;
    private static final int RIICHI = 32;
    private static final int KYUSHU = 64;

    /** 行動要求が存在しないことを表す共有値。 */
    public static final ActionPrompt NONE = new ActionPrompt(PromptPhase.NONE, 0);

    /**
     * 行動要求なしの共有値を返す。
     *
     * @return {@link #NONE}
     */
    public static ActionPrompt none() {
      return NONE;
    }

    /**
     * 自摸番に対する行動要求を生成する。
     *
     * @param mask サーバーが通知した合法行動ビットマスク
     * @return 自摸時の行動選択要求
     */
    public static ActionPrompt draw(int mask) {
      return new ActionPrompt(PromptPhase.DRAW, mask);
    }

    /**
     * 他家行動に対する応答要求を生成する。
     *
     * @param mask サーバーが通知した合法行動ビットマスク
     * @return 応答の行動選択要求。マスクが0なら {@link #NONE}
     */
    public static ActionPrompt response(int mask) {
      return mask == 0 ? NONE : new ActionPrompt(PromptPhase.RESPONSE, mask);
    }

    /**
     * サーバーから行動要求が通知されているかを返す。
     *
     * @return phase が {@link PromptPhase#NONE} でなければ {@code true}
     */
    public boolean present() {
      return phase != PromptPhase.NONE;
    }

    /**
     * 通常打牌を返せる自摸フェーズかを返す。
     *
     * @return 自摸フェーズなら {@code true}
     */
    public boolean canDahai() {
      return phase == PromptPhase.DRAW;
    }

    /**
     * 自摸和了を選択できるかを返す。
     *
     * @return 自摸フェーズで TSUMO ビットが立っていれば {@code true}
     */
    public boolean canTsumo() {
      return phase == PromptPhase.DRAW && has(TSUMO);
    }

    /**
     * リーチを選択できるかを返す。
     *
     * @return 自摸フェーズで RIICHI ビットが立っていれば {@code true}
     */
    public boolean canRiichi() {
      return phase == PromptPhase.DRAW && has(RIICHI);
    }

    /**
     * チーを選択できるかを返す。
     *
     * @return 応答フェーズで CHI ビットが立っていれば {@code true}
     */
    public boolean canChi() {
      return phase == PromptPhase.RESPONSE && has(CHI);
    }

    /**
     * ポンを選択できるかを返す。
     *
     * @return 応答フェーズで PON ビットが立っていれば {@code true}
     */
    public boolean canPon() {
      return phase == PromptPhase.RESPONSE && has(PON);
    }

    /**
     * 現在フェーズでいずれかの槓を選択できるかを返す。
     *
     * @return 行動要求が存在し KAN ビットが立っていれば {@code true}
     */
    public boolean canKan() {
      return present() && has(KAN);
    }

    /**
     * ロン和了を選択できるかを返す。
     *
     * @return 応答フェーズで RON ビットが立っていれば {@code true}
     */
    public boolean canRon() {
      return phase == PromptPhase.RESPONSE && has(RON);
    }

    /**
     * 九種九牌を選択できるかを返す。
     *
     * @return 自摸フェーズで KYUSHU ビットが立っていれば {@code true}
     */
    public boolean canKyushu() {
      return phase == PromptPhase.DRAW && has(KYUSHU);
    }

    private boolean has(int flag) {
      return (mask & flag) != 0;
    }
  }

  // 制御
  /** TAIKYOKU/SAIKAI による対局開始通知。ライブプロトコルでは自席は常に P0。 */
  record GameStart() implements TenhouEvent {}

  /** 生存確認応答。 */
  record KeepAlive() implements TenhouEvent {}

  /**
   * 既知だが対局制御に不要なメッセージ。
   *
   * @param tag XMLタグ名
   */
  record Ignored(String tag) implements TenhouEvent {}

  /** 対局終了。 */
  record EndGame() implements TenhouEvent {}

  /**
   * 解析不能メッセージ。
   *
   * @param raw 解析前の受信文字列
   */
  record Unknown(String raw) implements TenhouEvent {}
}
