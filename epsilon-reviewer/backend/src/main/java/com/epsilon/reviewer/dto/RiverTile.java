package com.epsilon.reviewer.dto;

/** 河の鳴かれた牌も位置情報として保持する。 */
public record RiverTile(String tile, boolean tsumogiri, boolean riichi, boolean called) {}
