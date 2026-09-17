package com.epsilon.reviewer.engine;

import com.epsilon.core.GameState;
import com.epsilon.reviewer.dto.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** 直前の表示状態を保持し、32局面ごとの全状態と、その間の局面の差分を生成する。 */
final class ReviewStateSequence {
  private final List<StateFrame> frames = new ArrayList<>();
  private TableState previous;
  private int previousStateId = -1;

  List<StateFrame> frames() {
    return frames;
  }

  void captureState(int stateId, GameState state) {
    if (stateId == previousStateId) return;
    TableState current = ReviewSnapshots.table(state);
    frames.add(
        frames.size() % 32 == 0
            ? new StateFrame(stateId, current, null)
            : new StateFrame(stateId, null, createStateDelta(previous, current)));
    previous = current;
    previousStateId = stateId;
  }

  private static StateDelta createStateDelta(TableState before, TableState after) {
    List<PlayerDelta> players = new ArrayList<>();
    for (int seat = 0; seat < 4; seat++) {
      PlayerState old = before.players().get(seat), next = after.players().get(seat);
      if (old.equals(next)) continue;
      players.add(
          new PlayerDelta(
              seat,
              changedInteger(old.score(), next.score()),
              old.riichi() == next.riichi() ? null : next.riichi(),
              createArrayPatch(old.hand(), next.hand()),
              Objects.equals(old.draw(), next.draw()) ? null : new DrawUpdate(next.draw()),
              createArrayPatch(old.river(), next.river()),
              createArrayPatch(old.melds(), next.melds())));
    }
    return new StateDelta(
        players,
        createArrayPatch(before.dora(), after.dora()),
        changedInteger(before.honba(), after.honba()),
        changedInteger(before.kyotaku(), after.kyotaku()),
        changedInteger(before.remaining(), after.remaining()),
        changedInteger(before.activeSeat(), after.activeSeat()));
  }

  private static Integer changedInteger(int before, int after) {
    return before == after ? null : after;
  }

  private static <T> ArrayPatch<T> createArrayPatch(List<T> before, List<T> after) {
    int prefix = 0, oldEnd = before.size(), newEnd = after.size();
    while (prefix < oldEnd
        && prefix < newEnd
        && Objects.equals(before.get(prefix), after.get(prefix))) prefix++;
    if (prefix == oldEnd && prefix == newEnd) return null;
    while (oldEnd > prefix
        && newEnd > prefix
        && Objects.equals(before.get(oldEnd - 1), after.get(newEnd - 1))) {
      oldEnd--;
      newEnd--;
    }
    // subListを保持すると変更前後の配列全体が残るため、保存する差分だけを所有する。
    return new ArrayPatch<>(
        prefix, oldEnd - prefix, new ArrayList<>(after.subList(prefix, newEnd)));
  }
}
