package com.epsilon.reviewer.engine.scenario;

/**
 * 開いているドラ表示牌の変更。
 *
 * @param index 0 が局の始まりの表示牌、1 からが槓ドラ（開いた順）
 */
public record DoraEdit(int index, String tile) {}
