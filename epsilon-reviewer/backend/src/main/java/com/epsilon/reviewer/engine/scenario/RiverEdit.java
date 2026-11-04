package com.epsilon.reviewer.engine.scenario;

/**
 * 他家の河の 1 枚の変更。
 *
 * @param index 判断の局面の river[] の位置（鳴かれた牌も数える）
 */
public record RiverEdit(int seat, int index, String tile, boolean tsumogiri) {}
