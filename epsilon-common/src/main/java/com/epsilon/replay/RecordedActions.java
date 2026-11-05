package com.epsilon.replay;

import com.epsilon.core.Action;
import com.epsilon.core.Tile;
import com.epsilon.replay.ReplayEvent.Ankan;
import com.epsilon.replay.ReplayEvent.Chi;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Daiminkan;
import com.epsilon.replay.ReplayEvent.Kakan;
import com.epsilon.replay.ReplayEvent.Pon;

/** 牌譜の出来事が表す手。打牌・チー・ポンは判断で確かめたこの手で遷移する（手から出す赤もこの手で決まる）。 */
final class RecordedActions {
  private RecordedActions() {}

  /** @param riichi 牌譜がこの打牌の前に同じ席のリーチを宣言した */
  static Action discard(Dahai discard, boolean riichi) {
    int type = Tile.typeOf(discard.physicalTileId());
    boolean red = Tile.isAka(discard.physicalTileId());
    return riichi
        ? Action.riichiDahai(type, red, discard.tsumogiri())
        : Action.dahai(type, red, discard.tsumogiri());
  }

  static Action ankan(Ankan kan) {
    return Action.ankan(Tile.typeOf(kan.consumedPhysicalTileIds()[0]));
  }

  static Action kakan(Kakan kan) {
    return Action.kakan(Tile.typeOf(kan.addedPhysicalTileId()));
  }

  /** 順子の先頭は鳴いた牌と手から出した 2 枚の最も小さい種類。手から赤を出せば赤を使う手。 */
  static Action chi(Chi chi) {
    int called = Tile.typeOf(chi.calledPhysicalTileId());
    int[] consumed = chi.consumedPhysicalTileIds();
    int base = Math.min(called, Math.min(Tile.typeOf(consumed[0]), Tile.typeOf(consumed[1])));
    return Action.chiSequence(base, called, containsAka(consumed));
  }

  /** 手から赤を出せば赤を使う手。遷移はこの手の {@link Action#usesAkaTileFromHand} で消費する赤を決める。 */
  static Action pon(Pon pon) {
    return Action.pon(
        Tile.typeOf(pon.calledPhysicalTileId()), containsAka(pon.consumedPhysicalTileIds()));
  }

  static Action daiminkan(Daiminkan kan) {
    return Action.daiminkan(Tile.typeOf(kan.calledPhysicalTileId()));
  }

  private static boolean containsAka(int[] physicalTileIds) {
    for (int physicalTileId : physicalTileIds) if (Tile.isAka(physicalTileId)) return true;
    return false;
  }
}
