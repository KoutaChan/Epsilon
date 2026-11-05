package com.epsilon.reviewer.engine.scenario;

/**
 * 河の 1 枚の変更。判断の席の打牌は、記録のツモ切りの印のまま牌だけを変える。
 *
 * @param index 判断の局面の river[] の位置（鳴かれた牌も数える）
 */
public record RiverEdit(int seat, int index, String tile, boolean tsumogiri) {}
