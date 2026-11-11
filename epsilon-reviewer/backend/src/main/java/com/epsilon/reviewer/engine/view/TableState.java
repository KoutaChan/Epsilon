package com.epsilon.reviewer.engine.view;

import java.util.List;

/**
 * 画面に出す局面。
 *
 * @param players 席ごとの状態（絶対席の順）
 * @param dora めくったドラ表示牌（めくった順）
 * @param honba 本場の数
 * @param kyotaku 卓に出ている供託の本数
 * @param remaining 残りのツモの数
 * @param activeSeat 手番の席
 */
public record TableState(
    List<PlayerState> players,
    List<String> dora,
    int honba,
    int kyotaku,
    int remaining,
    int activeSeat) {}
