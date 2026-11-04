package com.epsilon.reviewer.engine.scenario;

/**
 * 判断の席の手牌の 1 枚の入れ替え。牌の数は変わらない。
 *
 * @param index 判断の局面の hand[] の位置。hand の長さはツモ牌
 */
public record HandEdit(int index, String tile) {}
