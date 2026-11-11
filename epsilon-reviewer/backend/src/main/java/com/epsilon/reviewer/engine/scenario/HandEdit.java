package com.epsilon.reviewer.engine.scenario;

/**
 * 判断する席の手牌を 1 枚入れ替える。牌の枚数は変わらない。
 *
 * @param index 判断の局面の {@code hand[]} の位置。{@code hand} の長さと同じ値ならツモ牌
 */
public record HandEdit(int index, String tile) {}
