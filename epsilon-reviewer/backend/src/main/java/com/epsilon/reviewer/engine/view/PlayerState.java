package com.epsilon.reviewer.engine.view;

import java.util.ArrayList;
import java.util.List;

/**
 * 1 つの席の表示用の局面。
 *
 * @param hand 伏せた手牌（ツモ牌を除く）。{@link TileCode#order} の順
 * @param draw 見えているツモ牌。無ければ null
 * @param melds 副露と暗槓（鳴いた順）
 * @param nuki 三人打ちで抜いた北の数
 */
public record PlayerState(
    int score,
    boolean riichi,
    List<String> hand,
    String draw,
    List<RiverTile> river,
    List<Meld> melds,
    int nuki) {
  public PlayerState withScore(int score) {
    return new PlayerState(score, riichi, hand, draw, river, melds, nuki);
  }

  /** @param hand 伏せた手牌。{@link TileCode#order} の順 */
  public PlayerState withHand(List<String> hand, String draw) {
    return new PlayerState(score, riichi, hand, draw, river, melds, nuki);
  }

  public PlayerState withRiverTile(int index, RiverTile tile) {
    List<RiverTile> changed = new ArrayList<>(river);
    changed.set(index, tile);
    return new PlayerState(score, riichi, hand, draw, List.copyOf(changed), melds, nuki);
  }
}
