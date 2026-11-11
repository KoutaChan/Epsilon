package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;

/** テスト用に、偶然のイベントを好きな順に並べた記録を作る。ルールに合わない並びも作れる。 */
public final class ChanceLogs {

  private ChanceLogs() {}

  /**
   * 指定した種類の順にイベントを並べた記録を作る。牌の種類が重ならないように、イベントごとに違う種類の牌を使う。
   *
   * @param variant ルール
   * @param kinds イベントの種類の並び
   * @return 空の手牌 1 つで始まり、ツモと嶺上牌は全て席 0 が引く記録
   */
  public static ChanceLog of(Variant variant, ChanceKind... kinds) {
    ChanceLog log = new ChanceLog(variant);
    log.begin(new Hand[] {new Hand(variant)}, 0);
    for (int index = 0; index < kinds.length; index++) {
      int tile = variant.tiles().physicalId(4 * (index + 1));
      switch (kinds[index]) {
        case DRAW -> log.appendDraw(0, tile);
        case RINSHAN -> log.appendRinshan(0, tile);
        case INDICATOR -> log.appendIndicator(tile);
        case URA -> log.appendUra(tile);
      }
    }
    return log;
  }
}
