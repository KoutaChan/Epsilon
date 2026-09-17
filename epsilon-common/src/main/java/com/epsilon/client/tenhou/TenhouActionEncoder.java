package com.epsilon.client.tenhou;

import com.epsilon.core.Action;
import com.epsilon.core.Hand;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;

/**
 * AI の行動を 天鳳のWebSocketプロトコルで使用するJSONメッセージに変換する。
 *
 * <p>天鳳のWebSocketプロトコルの応答形式:
 *
 * <ul>
 *   <li>打牌: {@code {"tag":"D","p":<physicalTileId>}}
 *   <li>ツモ和了: {@code {"tag":"N","type":7}}
 *   <li>ロン: {@code {"tag":"N","type":6}}
 *   <li>ポン: {@code {"tag":"N","type":1,"hai0":<X>,"hai1":<Y>}}
 *   <li>大明槓: {@code {"tag":"N","type":2}}
 *   <li>チー: {@code {"tag":"N","type":3,"hai0":<X>,"hai1":<Y>}}
 *   <li>暗槓: {@code {"tag":"N","type":4,"hai":<X>}}
 *   <li>加槓: {@code {"tag":"N","type":5,"hai":<X>}}
 *   <li>パス: {@code {"tag":"N"}}
 *   <li>九種九牌: {@code {"tag":"N","type":9}}
 * </ul>
 */
public final class TenhouActionEncoder {

  private TenhouActionEncoder() {}

  /**
   * 打牌用の簡易変換。
   *
   * @param physicalTileId 実牌ID
   * @return JSON 文字列
   */
  public static String encodeDahai(int physicalTileId) {
    return "{\"tag\":\"D\",\"p\":" + physicalTileId + "}";
  }

  /**
   * リーチ宣言メッセージを生成する。
   *
   * @return {@code REACH} メッセージの JSON 文字列
   */
  public static String encodeReach() {
    return "{\"tag\":\"REACH\"}";
  }

  /**
   * 次局準備完了メッセージを生成する。
   *
   * @return {@code NEXTREADY} メッセージの JSON 文字列
   */
  public static String encodeNextReady() {
    return "{\"tag\":\"NEXTREADY\"}";
  }

  /**
   * 対局確認メッセージを生成する。
   *
   * @return {@code GOK} メッセージの JSON 文字列
   */
  public static String encodeGok() {
    return "{\"tag\":\"GOK\"}";
  }

  private static String encodeCallWithTwoTiles(
      int tenhouActionTypeCode, int firstPhysicalTileId, int secondPhysicalTileId) {
    return "{\"tag\":\"N\",\"type\":"
        + tenhouActionTypeCode
        + ",\"hai0\":"
        + firstPhysicalTileId
        + ",\"hai1\":"
        + secondPhysicalTileId
        + "}";
  }

  /** 行動と自手牌から天鳳プロトコルメッセージ列を生成する。 */
  public static String[] encodeAction(Action action, Hand hand, TurnEvent turnEvent) {
    return switch (action.type()) {
      case DAHAI -> {
        int physicalTileId = selectDahaiPhysicalTileId(action, hand, turnEvent);
        yield new String[] {encodeDahai(physicalTileId)};
      }
      case RIICHI_DAHAI -> {
        int physicalTileId = selectDahaiPhysicalTileId(action, hand, turnEvent);
        yield new String[] {encodeReach(), encodeDahai(physicalTileId)};
      }
      case CHI -> new String[] {encodeChi(action, hand)};
      case PON -> new String[] {encodePon(action, hand)};
      case DAIMINKAN -> new String[] {encodeNAction(2)};
      case ANKAN ->
          new String[] {encodeNActionWithTile(4, hand.firstPhysicalTileId(action.tileType()))};
      case KAKAN ->
          new String[] {encodeNActionWithTile(5, selectKakanPhysicalTileId(action, hand))};
      case RON_AGARI -> new String[] {encodeNAction(6)};
      case TSUMO_AGARI -> new String[] {encodeNAction(7)};
      case PASS -> new String[] {"{\"tag\":\"N\"}"};
      case KYUSHU_KYUHAI -> new String[] {encodeNAction(9)};
    };
  }

  private static int selectDahaiPhysicalTileId(Action action, Hand hand, TurnEvent turnEvent) {
    if (action.isTsumogiri() && turnEvent instanceof TurnEvent.Draw draw) {
      return draw.physicalTileId();
    }
    return hand.physicalTileId(action.tileType(), action.usesAkaTileFromHand(), 0);
  }

  private static String encodeChi(Action action, Hand hand) {
    int calledTileType = action.tileType();
    boolean consumeAkaFromHand = action.usesAkaTileFromHand();
    int first = -1;
    int second = -1;
    boolean akaTileAlreadySelected = false;
    boolean calledTileSkipped = false;
    int base = action.chiSequenceBaseTileType();
    for (int tileType = base; tileType < base + 3; tileType++) {
      if (tileType == calledTileType && !calledTileSkipped) {
        calledTileSkipped = true;
        continue;
      }
      boolean usesAkaTileFromHand =
          consumeAkaFromHand && !akaTileAlreadySelected && Tile.canBeAka(tileType);
      int selected = hand.physicalTileId(tileType, usesAkaTileFromHand, 0);
      if (first < 0) {
        first = selected;
      } else {
        second = selected;
      }
      if (usesAkaTileFromHand) {
        akaTileAlreadySelected = true;
      }
    }
    return encodeCallWithTwoTiles(3, first, second);
  }

  private static String encodePon(Action action, Hand hand) {
    int tileType = action.tileType();
    int first;
    int second;
    if (action.usesAkaTileFromHand()) {
      first = hand.physicalTileId(tileType, true, 0);
      second = hand.physicalTileId(tileType, false, 0);
    } else {
      first = hand.physicalTileId(tileType, false, 0);
      second = hand.physicalTileId(tileType, false, 1);
    }
    return encodeCallWithTwoTiles(1, first, second);
  }

  private static int selectKakanPhysicalTileId(Action action, Hand hand) {
    int tileType = action.tileType();
    return hand.hasAkaTile(tileType)
        ? hand.physicalTileId(tileType, true, 0)
        : hand.firstPhysicalTileId(tileType);
  }

  private static String encodeNAction(int tenhouActionTypeCode) {
    return "{\"tag\":\"N\",\"type\":" + tenhouActionTypeCode + "}";
  }

  private static String encodeNActionWithTile(int tenhouActionTypeCode, int physicalTileId) {
    return "{\"tag\":\"N\",\"type\":" + tenhouActionTypeCode + ",\"hai\":" + physicalTileId + "}";
  }
}
