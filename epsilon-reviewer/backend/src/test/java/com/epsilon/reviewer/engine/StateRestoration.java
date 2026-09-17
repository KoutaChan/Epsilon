package com.epsilon.reviewer.engine;

import com.epsilon.reviewer.dto.*;
import java.util.ArrayList;
import java.util.List;

/** 保存済みの配列差分を復元するテスト用デコーダー。符号化側の実装を使わずに復元結果を検証する。 */
final class StateRestoration {
  static TableState at(List<StateFrame> frames, int stateId) {
    int index = 0;
    while (frames.get(index).stateId() != stateId) index++;
    int checkpoint = index;
    while (frames.get(checkpoint).checkpoint() == null) checkpoint--;
    TableState state = frames.get(checkpoint).checkpoint();
    for (int i = checkpoint + 1; i <= index; i++) state = apply(state, frames.get(i).delta());
    return state;
  }

  private static TableState apply(TableState before, StateDelta delta) {
    var players = new ArrayList<>(before.players());
    for (PlayerDelta update : delta.players()) {
      PlayerState old = players.get(update.seat());
      players.set(
          update.seat(),
          new PlayerState(
              update.score() == null ? old.score() : update.score(),
              update.riichi() == null ? old.riichi() : update.riichi(),
              patch(old.hand(), update.hand()),
              update.draw() == null ? old.draw() : update.draw().tile(),
              patch(old.river(), update.river()),
              patch(old.melds(), update.melds())));
    }
    return new TableState(
        players,
        patch(before.dora(), delta.dora()),
        delta.honba() == null ? before.honba() : delta.honba(),
        delta.kyotaku() == null ? before.kyotaku() : delta.kyotaku(),
        delta.remaining() == null ? before.remaining() : delta.remaining(),
        delta.activeSeat() == null ? before.activeSeat() : delta.activeSeat());
  }

  private static <T> List<T> patch(List<T> previous, ArrayPatch<T> patch) {
    if (patch == null) return previous;
    var next = new ArrayList<>(previous);
    next.subList(patch.index(), patch.index() + patch.removeCount()).clear();
    next.addAll(patch.index(), patch.values());
    return next;
  }
}
