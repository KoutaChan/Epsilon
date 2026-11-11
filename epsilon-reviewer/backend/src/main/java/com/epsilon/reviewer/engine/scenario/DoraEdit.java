package com.epsilon.reviewer.engine.scenario;

/**
 * めくられているドラ表示牌を 1 枚変える。
 *
 * @param index 0 が局の始まりの表示牌、1 から先が槓ドラの表示牌（めくった順）
 */
public record DoraEdit(int index, String tile) {}
