package com.epsilon.replay.format;

import java.util.List;

/** JSON と protobuf の復号結果を同じ雀魂固有の型へ揃える。 */
sealed interface MahjongSoulRecord {
  record Round(
      int wind,
      int dealer,
      int honba,
      int deposits,
      int[] scores,
      List<List<String>> hands,
      List<String> indicators)
      implements MahjongSoulRecord {}

  record Draw(int seat, String tile, List<String> indicators, Riichi riichi)
      implements MahjongSoulRecord {}

  record Discard(int seat, String tile, boolean riichi, boolean tsumogiri, List<String> indicators)
      implements MahjongSoulRecord {}

  record Call(int seat, int type, List<String> tiles, int[] sources, Riichi riichi)
      implements MahjongSoulRecord {}

  record Kan(int seat, int type, String tile, List<String> indicators)
      implements MahjongSoulRecord {}

  record Win(List<Winner> winners, int[] deltas, int[] scores, GameEnd end)
      implements MahjongSoulRecord {}

  record ExhaustiveDraw(int[] deltas, GameEnd end) implements MahjongSoulRecord {}

  record AbortiveDraw(int reason, GameEnd end, Riichi riichi) implements MahjongSoulRecord {}

  record Riichi(int seat, boolean failed) {}

  record GameEnd(boolean complete, int[] scores) {
    static final GameEnd NONE = new GameEnd(false, null);
  }

  record Fan(int id, int value) {}

  record Winner(
      int seat,
      String tile,
      boolean tsumo,
      Boolean riichi,
      boolean yakuman,
      Integer count,
      Integer fu,
      Integer ronPoints,
      Integer dealerPayment,
      Integer childPayment,
      List<Fan> fans,
      List<String> uraIndicators) {}
}
