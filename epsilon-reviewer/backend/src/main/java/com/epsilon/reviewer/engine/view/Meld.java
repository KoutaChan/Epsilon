package com.epsilon.reviewer.engine.view;

import java.util.List;

/**
 * 表示用の副露。
 *
 * @param type CHI・PON・DAIMINKAN・ANKAN・KAKAN のどれか。三人打ちで抜いた北は NUKI
 * @param tiles 表示の順に並べた牌。チー・ポンは 3 枚、槓は 4 枚、抜いた北は 1 枚
 * @param from 鳴いた相手の席（絶対席）。暗槓と抜いた北は -1
 * @param calledIndex 横向きの牌の位置。暗槓と抜いた北は -1
 * @param addedIndex 加槓で加えた牌の位置（常に 3）。加槓でなければ -1
 */
public record Meld(String type, List<String> tiles, int from, int calledIndex, int addedIndex) {}
