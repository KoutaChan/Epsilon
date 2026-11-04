package com.epsilon.reviewer.engine.view;

import java.util.List;

/**
 * 表示の副露。
 *
 * @param type CHI・PON・DAIMINKAN・ANKAN・KAKAN
 * @param tiles CHI・PON は 3 枚、槓は 4 枚。表示の順
 * @param from 鳴いた元の絶対の席。暗槓は -1
 * @param calledIndex 横向きの牌の位置。暗槓は -1
 * @param addedIndex 加槓で加えた牌の位置（3）。他は -1
 */
public record Meld(String type, List<String> tiles, int from, int calledIndex, int addedIndex) {}
