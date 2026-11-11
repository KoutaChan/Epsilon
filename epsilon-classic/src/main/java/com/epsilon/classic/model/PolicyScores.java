package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;

/**
 * 方策ヘッドが出す、正規化していないスコア。方策のグラフ（{@link PolicyGraph}）がこれを合法手の確率にする。どれも計算の型。
 *
 * @param alternatives 分岐のスコア [行, {@link Alternative} の数]。分岐が無い行は 0
 * @param actions 合法手のスコア [行, 合法手]。同じ種類の候補どうしと、物理の打牌どうしを比べる値で、ダマと立直の打牌には同じ値が入る。合法手の無い位置は 0
 * @param riichi 同じ物理の打牌をダマで切るか立直するかのロジット [行, 合法手]。立直の打牌の位置に置き、ほかの位置は 0
 */
public record PolicyScores(NDArray alternatives, NDArray actions, NDArray riichi) {}
