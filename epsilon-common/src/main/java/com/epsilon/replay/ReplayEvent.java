package com.epsilon.replay;

/** 異なる牌譜形式に共通する、再生用の正規化されたイベント。 */
public sealed interface ReplayEvent {

  /**
   * 他家の行動に応答したプレイヤーの席を返す。牌譜から応答時の判断を復元するために使う。
   *
   * @return 応答した席。応答イベントでなければ-1
   */
  default int responseActor() {
    return -1;
  }

  /** 対局開始を表す。 */
  record StartGame() implements ReplayEvent {}

  /**
   * 局開始時の公開状態と配牌を表す。
   *
   * @param wind 場風
   * @param kyoku 場内の局番号
   * @param honba 本場数
   * @param kyotaku 供託リーチ棒数
   * @param oya 親の絶対席
   * @param doraIndicatorTileType 最初のドラ表示牌種
   * @param initialHandPhysicalTileIds 席ごとの配牌物理 ID
   * @param doraIndicatorAka 最初の表示牌が赤牌か
   * @param scores 席ごとの開始点
   */
  record StartKyoku(
      int wind,
      int kyoku,
      int honba,
      int kyotaku,
      int oya,
      int doraIndicatorTileType,
      int[][] initialHandPhysicalTileIds,
      int[] scores,
      boolean doraIndicatorAka)
      implements ReplayEvent {
    public StartKyoku(
        int wind,
        int kyoku,
        int honba,
        int kyotaku,
        int oya,
        int doraIndicatorTileType,
        int[][] initialHandPhysicalTileIds,
        int[] scores) {
      this(
          wind,
          kyoku,
          honba,
          kyotaku,
          oya,
          doraIndicatorTileType,
          initialHandPhysicalTileIds,
          scores,
          false);
    }
  }

  /**
   * 山からの自摸を表す。
   *
   * @param actor 自摸した席
   * @param physicalTileId 自摸牌の物理 ID
   */
  record Tsumo(int actor, int physicalTileId, boolean inferred) implements ReplayEvent {
    public Tsumo(int actor, int physicalTileId) {
      this(actor, physicalTileId, false);
    }
  }

  /**
   * 打牌を表す。
   *
   * @param actor 打牌した席
   * @param physicalTileId 打牌の物理 ID
   * @param tsumogiri ツモ切りなら {@code true}
   */
  record Dahai(int actor, int physicalTileId, boolean tsumogiri, boolean inferred)
      implements ReplayEvent {
    public Dahai(int actor, int physicalTileId, boolean tsumogiri) {
      this(actor, physicalTileId, tsumogiri, false);
    }
  }

  /**
   * チー成立を表す。
   *
   * @param actor 鳴いた席
   * @param target 捨て牌元の席
   * @param calledPhysicalTileId 鳴いた捨て牌の物理 ID
   * @param consumedPhysicalTileIds 手牌から消費した牌の物理 ID
   */
  record Chi(int actor, int target, int calledPhysicalTileId, int[] consumedPhysicalTileIds)
      implements ReplayEvent {
    /** {@inheritDoc} */
    @Override
    public int responseActor() {
      return actor;
    }
  }

  /**
   * ポン成立を表す。
   *
   * @param actor 鳴いた席
   * @param target 捨て牌元の席
   * @param calledPhysicalTileId 鳴いた捨て牌の物理 ID
   * @param consumedPhysicalTileIds 手牌から消費した牌の物理 ID
   */
  record Pon(int actor, int target, int calledPhysicalTileId, int[] consumedPhysicalTileIds)
      implements ReplayEvent {
    /** {@inheritDoc} */
    @Override
    public int responseActor() {
      return actor;
    }
  }

  /**
   * 大明槓成立を表す。
   *
   * @param actor 鳴いた席
   * @param target 捨て牌元の席
   * @param calledPhysicalTileId 鳴いた捨て牌の物理 ID
   * @param consumedPhysicalTileIds 手牌から消費した牌の物理 ID
   */
  record Daiminkan(int actor, int target, int calledPhysicalTileId, int[] consumedPhysicalTileIds)
      implements ReplayEvent {
    /** {@inheritDoc} */
    @Override
    public int responseActor() {
      return actor;
    }
  }

  /**
   * 暗槓成立を表す。
   *
   * @param actor 槓した席
   * @param consumedPhysicalTileIds 手牌から消費した4牌の物理 ID
   */
  record Ankan(int actor, int[] consumedPhysicalTileIds) implements ReplayEvent {}

  /**
   * 加槓成立を表す。
   *
   * @param actor 加槓した席
   * @param addedPhysicalTileId 既存ポンへ加えた牌の物理 ID
   */
  record Kakan(int actor, int addedPhysicalTileId) implements ReplayEvent {}

  /**
   * リーチ宣言打牌の宣言段階を表す。
   *
   * @param actor 宣言した席
   */
  record Reach(int actor) implements ReplayEvent {}

  /**
   * リーチ成立と供託支払いを表す。
   *
   * @param actor リーチが成立した席
   */
  record ReachAccepted(int actor) implements ReplayEvent {}

  /**
   * 槓後などに追加されたドラ表示牌を表す。
   *
   * @param indicatorTileType ドラ表示牌種
   */
  record Dora(int indicatorTileType, boolean indicatorAka) implements ReplayEvent {
    public Dora(int indicatorTileType) {
      this(indicatorTileType, false);
    }
  }

  /**
   * 和了とその点棒移動を表す。
   *
   * @param actor 和了した席
   * @param target ロン対象。ツモ和了では行動したプレイヤーと同じ
   * @param winTileType 和了牌種
   * @param deltas 席ごとの点棒増減
   * @param details 元牌譜の採点内訳。未記録なら null
   */
  record Hora(int actor, int target, int winTileType, int[] deltas, WinDetails details)
      implements ReplayEvent {
    public Hora(int actor, int target, int winTileType, int[] deltas) {
      this(actor, target, winTileType, deltas, null);
    }

    /** {@inheritDoc} */
    @Override
    public int responseActor() {
      return actor != target ? actor : -1;
    }
  }

  /**
   * 流局とその点棒移動を表す。
   *
   * @param deltas 席ごとの点棒増減
   */
  record Ryukyoku(int[] deltas, String reason) implements ReplayEvent {
    public Ryukyoku(int[] deltas) {
      this(deltas, null);
    }
  }

  /** 局末イベント列の終了を表す。 */
  record EndKyoku() implements ReplayEvent {}

  /**
   * 対局終了時の確定得点を表す。
   *
   * @param scores 席ごとの最終得点
   */
  record EndGame(int[] scores) implements ReplayEvent {}

  /**
   * 牌譜上で応答を見送ったプレイヤーを明示する。
   *
   * @param actor 応答を見送った席
   */
  record None(int actor) implements ReplayEvent {}
}
