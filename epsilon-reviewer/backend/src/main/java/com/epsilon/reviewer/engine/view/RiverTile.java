package com.epsilon.reviewer.engine.view;

/**
 * 河の 1 枚。
 *
 * @param riichi リーチの宣言牌
 * @param called 鳴かれた牌（河に残して数える）
 */
public record RiverTile(String tile, boolean tsumogiri, boolean riichi, boolean called) {
  /** 牌とツモ切りの印を替えた写し。宣言牌と鳴かれた印は保つ。 */
  public RiverTile withTile(String tile, boolean tsumogiri) {
    return new RiverTile(tile, tsumogiri, riichi, called);
  }
}
