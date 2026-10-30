package com.epsilon.engine;

import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;

/** 麻雀ルールのシナリオで共有する、牌姿からの最小構築。 */
final class MahjongFixtures {
  private MahjongFixtures() {}

  static int[] counts(String notation) {
    int[] result = new int[Tile.NUM_TILE_TYPES];
    int start = 0;
    for (int end = 0; end < notation.length(); end++) {
      char suit = notation.charAt(end);
      int offset =
          switch (suit) {
            case 'm' -> 0;
            case 'p' -> 9;
            case 's' -> 18;
            case 'z' -> 27;
            default -> -1;
          };
      if (offset < 0) continue;
      for (int digit = start; digit < end; digit++) {
        result[offset + Character.digit(notation.charAt(digit), 10) - 1]++;
      }
      start = end + 1;
    }
    return result;
  }

  static void hand(GameState state, int seat, String notation) {
    Hand hand = state.hand(seat);
    hand.clear();
    int[] counts = counts(notation);
    for (int tile = 0; tile < counts.length; tile++) {
      for (int copy = 0; copy < counts[tile]; copy++) hand.add(tile);
    }
  }

  /** 元の席がその牌を打ってから、指定席がそれを鳴いた副露を足す（鳴きは記録の最後の打牌を鳴く）。 */
  static void call(GameState state, int seat, Meld meld) {
    int source = (seat + meld.relativeSource().playerOffset()) % GameState.NUM_PLAYERS;
    state.commitDiscard(source, meld.calledTileType(), false);
    state.addMeld(seat, meld, state.river(seat).size());
  }

  static GameState state() {
    GameState state = new GameState(17);
    state.startRound(0, 0, 0, 0);
    return state;
  }
}
