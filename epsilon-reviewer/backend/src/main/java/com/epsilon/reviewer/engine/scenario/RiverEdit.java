package com.epsilon.reviewer.engine.scenario;

/**
 * 河の牌を 1 枚変える。他家の打牌なら、牌とツモ切りの印を変えられる。判断する席の打牌は牌だけを変えられ、ツモ切りの印には記録と
 * 同じ値を渡す。
 *
 * @param index 判断の局面の {@code river[]} の位置（鳴かれた牌も数える）
 */
public record RiverEdit(int seat, int index, String tile, boolean tsumogiri) {}
