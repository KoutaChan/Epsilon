package com.epsilon.reviewer.dto;

import java.util.List;

/**
 * 1 判断の合法手 1 つ。赤牌・ツモ切りを保持する。
 *
 * @param probability 全合法手で正規化した表示用の確率（0..1）
 * @param valueLoss 最善手の価値 − この手の価値（0 以上、最善手は 0）。価値を推定しないモデルは null（同じ判断の全候補で揃う）
 */
public record ActionCandidate(
    int actionId,
    String type,
    List<String> tiles,
    float probability,
    Float valueLoss,
    boolean chosen,
    boolean tsumogiri) {}
