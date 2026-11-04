package com.epsilon.reviewer.engine.view;

import java.util.ArrayList;
import java.util.List;

/**
 * 1 席の表示の局面。
 *
 * @param hand 伏せた手牌（ツモ牌を除く）。{@link TileCode#order} の順
 * @param draw 見えているツモ牌。無ければ null
 */
public record PlayerState(
    int score,
    boolean riichi,
    List<String> hand,
    String draw,
    List<RiverTile> river,
    List<Meld> melds) {
  public PlayerState withScore(int score) {
    return new PlayerState(score, riichi, hand, draw, river, melds);
  }

  /** @param hand {@link TileCode#order} の順 */
  public PlayerState withHand(List<String> hand, String draw) {
    return new PlayerState(score, riichi, hand, draw, river, melds);
  }

  public PlayerState withRiverTile(int index, RiverTile tile) {
    List<RiverTile> changed = new ArrayList<>(river);
    changed.set(index, tile);
    return new PlayerState(score, riichi, hand, draw, List.copyOf(changed), melds);
  }
}
