package com.epsilon.reviewer.engine.view;

import java.util.List;

/**
 * 表示の局面。players の位置が絶対の席。
 *
 * @param dora 開いたドラ表示牌（開いた順）
 * @param kyotaku 卓の上の供託の本数
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
