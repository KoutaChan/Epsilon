package com.epsilon.reviewer.engine.view;

/**
 * 河の 1 枚。
 *
 * @param riichi 立直の宣言牌か
 * @param called 鳴かれた牌か（鳴かれても河に残して数える）
 */
public record RiverTile(String tile, boolean tsumogiri, boolean riichi, boolean called) {
  /** 牌とツモ切りの印だけを替えたコピーを返す。立直の宣言牌の印と、鳴かれた牌の印はそのまま。 */
  public RiverTile withTile(String tile, boolean tsumogiri) {
    return new RiverTile(tile, tsumogiri, riichi, called);
  }
}
