package com.epsilon.engine;

/**
 * ゲーム記録用インターフェース。
 *
 * <p>GameEngine が局の進行イベントを通知するためのコールバックであり、 天鳳形式ログ出力など各種記録フォーマットの基盤として使う。
 */
public interface GameRecorder {

  /**
   * 局終了時にレコーダーへ渡す確定データ。
   *
   * @param finalScores 供託精算後の点数を席番号順に並べた配列
   * @param uraDoraPhysicalTileIds 和了時に公開する裏ドラ表示牌の実牌ID配列
   */
  record RoundEnd(int[] finalScores, int[] uraDoraPhysicalTileIds) {

    /**
     * 局終了配列を防御的に複製して保持する。
     *
     * @param finalScores 供託精算後の点数
     * @param uraDoraPhysicalTileIds 公開する裏ドラ表示牌の物理 ID
     */
    public RoundEnd {
      finalScores = finalScores.clone();
      uraDoraPhysicalTileIds = uraDoraPhysicalTileIds.clone();
    }

    /**
     * 確定得点を複製して返す。
     *
     * @return 席番号順の確定得点
     */
    @Override
    public int[] finalScores() {
      return finalScores.clone();
    }

    /**
     * 裏ドラ表示牌を複製して返す。
     *
     * @return 表示順の物理牌 ID
     */
    @Override
    public int[] uraDoraPhysicalTileIds() {
      return uraDoraPhysicalTileIds.clone();
    }
  }

  /**
   * 局の開始を記録する。
   *
   * @param roundIndex 東1局を0とする通算局インデックス
   * @param honba 本場数
   * @param kyotakuCount 供託本数
   * @param startingScores 開始時点数を席番号順に並べた配列
   */
  void startRound(int roundIndex, int honba, int kyotakuCount, int[] startingScores);

  /**
   * 配牌を記録する。
   *
   * @param player 配牌を受けた席番号（0-3）
   * @param concealedTileCounts 34牌種別の手牌枚数
   * @param concealedAkaMask 配牌に含まれる赤5の3-ビットマスク
   */
  void recordHaipai(int player, int[] concealedTileCounts, int concealedAkaMask);

  /**
   * 局開始時のドラ表示牌を記録する。
   *
   * @param physicalTileIds 表示順の実牌ID配列
   */
  void recordDoraIndicators(int[] physicalTileIds);

  /**
   * 槓で追加されたドラ表示牌を記録する。
   *
   * @param physicalTileId 追加表示牌の実牌ID
   */
  default void recordNewDora(int physicalTileId) {}

  /**
   * 通常ツモを記録する。
   *
   * @param player ツモした席番号（0-3）
   * @param tileType ツモ牌の牌種ID
   * @param isAkaTile ツモ牌が赤牌なら {@code true}
   */
  void recordDraw(int player, int tileType, boolean isAkaTile);

  /**
   * 嶺上ツモを記録する。
   *
   * @param player ツモした席番号（0-3）
   * @param tileType ツモ牌の牌種ID
   * @param isAkaTile ツモ牌が赤牌なら {@code true}
   */
  default void recordRinshanDraw(int player, int tileType, boolean isAkaTile) {
    recordDraw(player, tileType, isAkaTile);
  }

  /**
   * 打牌を記録する。
   *
   * @param player 打牌した席番号（0-3）
   * @param tileType 打牌の牌種ID
   * @param isAkaTile 打牌が赤牌なら {@code true}
   * @param tsumogiri ツモ切りなら {@code true}
   * @param riichi この打牌がリーチ宣言牌なら {@code true}
   */
  void recordDahai(int player, int tileType, boolean isAkaTile, boolean tsumogiri, boolean riichi);

  /**
   * リーチ宣言牌がロンされず、供託が成立したことを記録する。
   *
   * @param player リーチが成立した席番号（0-3）
   */
  void recordRiichiAccepted(int player);

  /**
   * チーを記録する。
   *
   * @param player 鳴いた席番号（0-3）
   * @param chiTileTypes 生成した順子3牌種
   * @param calledTileType 上家から鳴いた牌種
   * @param calledTileIsAka 鳴いた牌が赤牌なら {@code true}
   * @param consumedHandTileIsAka 手牌から赤牌を消費したなら {@code true}
   */
  void recordChi(
      int player,
      int[] chiTileTypes,
      int calledTileType,
      boolean calledTileIsAka,
      boolean consumedHandTileIsAka);

  /**
   * ポンを記録する。
   *
   * @param player 鳴いた席番号（0-3）
   * @param tileType 刻子の牌種ID
   * @param sourcePlayerOffset 鳴いた席から見た捨て牌元の相対席
   * @param calledTileIsAka 鳴いた牌が赤牌なら {@code true}
   * @param consumedHandTileIsAka 手牌から赤牌を消費したなら {@code true}
   */
  void recordPon(
      int player,
      int tileType,
      int sourcePlayerOffset,
      boolean calledTileIsAka,
      boolean consumedHandTileIsAka);

  /**
   * 大明槓を記録する。
   *
   * @param player 鳴いた席番号（0-3）
   * @param tileType 槓子の牌種ID
   * @param sourcePlayerOffset 鳴いた席から見た捨て牌元の相対席
   * @param calledTileIsAka 鳴いた牌が赤牌なら {@code true}
   * @param consumedHandTileIsAka 手牌側の槓子に赤牌を含むなら {@code true}
   */
  void recordDaiminkan(
      int player,
      int tileType,
      int sourcePlayerOffset,
      boolean calledTileIsAka,
      boolean consumedHandTileIsAka);

  /**
   * 暗槓を記録する。
   *
   * @param player 槓した席番号（0-3）
   * @param tileType 槓子の牌種ID
   * @param containsAkaTile 槓子に赤牌を含むなら {@code true}
   */
  void recordAnkan(int player, int tileType, boolean containsAkaTile);

  /**
   * 加槓を記録する。
   *
   * @param player 槓した席番号（0-3）
   * @param tileType 槓子の牌種ID
   * @param addedTileIsAka 追加した4枚目が赤牌なら {@code true}
   */
  void recordKakan(int player, int tileType, boolean addedTileIsAka);

  /**
   * 和了・流局を、確定済みの局結果として記録する。
   *
   * @param result 供託精算前の局結果
   */
  void recordRoundResult(RoundResult result);

  /**
   * 局の終了を記録する。
   *
   * @param roundEnd 供託精算後の点数と、表示する裏ドラ
   */
  void endRound(RoundEnd roundEnd);
}
