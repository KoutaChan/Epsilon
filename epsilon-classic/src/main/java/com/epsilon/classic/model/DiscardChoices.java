package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;

/**
 * 同じ牌を切る打牌について、ダマか立直かの選択をまとめる。同じ牌（赤かどうか・ツモ切りかどうかまで同じ）を切るダマの打牌と立直の打牌は、
 * 1 つの物理の打牌にまとめる。代表は、ダマの打牌があればそれ、立直の打牌しか無ければ立直の打牌にする。マスクは合法手の位置ごとに
 * 0 か 1 の [行, 合法手] で、型は打牌のマスクと同じ。位置の配列は合法手の位置 ＋ 1 の整数 [行, 合法手] で、0 はその手が無いことを表す。
 *
 * @param damaMask ダマの打牌なら 1
 * @param riichiMask 立直の打牌なら 1
 * @param discardMask どちらかの打牌なら 1
 * @param representativeSlots 同じ物理の打牌の代表の位置
 * @param damaSlots 同じ物理の打牌のうち、ダマの打牌の位置
 * @param riichiSlots 同じ物理の打牌のうち、立直の打牌の位置
 * @param damaPresent 打牌の位置で、同じ物理の打牌にダマの打牌があれば 1
 * @param riichiPresent 打牌の位置で、同じ物理の打牌に立直の打牌があれば 1
 * @param representativeMask 物理の打牌ごとに 1 つだけ 1 になる、代表の位置
 */
record DiscardChoices(
    NDArray damaMask,
    NDArray riichiMask,
    NDArray discardMask,
    NDArray representativeSlots,
    NDArray damaSlots,
    NDArray riichiSlots,
    NDArray damaPresent,
    NDArray riichiPresent,
    NDArray representativeMask) {

  /**
   * @param routes 合法手どうしの位置の対応 [行, 合法手, 列]
   * @param damaMask ダマの打牌なら 1（浮動小数点の型）
   * @param riichiMask 立直の打牌なら 1（ダマと同じ型）
   * @param columns 合法手の列のオフセット
   */
  static DiscardChoices of(NDArray routes, NDArray damaMask, NDArray riichiMask, ActionColumns columns) {
    NDArray discard = damaMask.add(riichiMask);
    NDArray damaSlots = routes.get("..., {}", columns.damaSlot());
    NDArray riichiSlots = routes.get("..., {}", columns.riichiSlot());
    NDArray representative = routes.get("..., {}", columns.representative()).toType(discard.getDataType(), false);
    return new DiscardChoices(
        damaMask,
        riichiMask,
        discard,
        routes.get("..., {}", columns.representativeSlot()),
        damaSlots,
        riichiSlots,
        discard.mul(damaSlots.neq(0).toType(discard.getDataType(), false)),
        discard.mul(riichiSlots.neq(0).toType(discard.getDataType(), false)),
        discard.mul(representative));
  }
}
