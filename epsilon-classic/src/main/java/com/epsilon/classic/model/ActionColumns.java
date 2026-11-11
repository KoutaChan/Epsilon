package com.epsilon.classic.model;

import com.epsilon.classic.Architecture;
import com.epsilon.classic.input.ClassicFields.ActionInt;
import com.epsilon.classic.input.ClassicFields.ActionRoute;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.input.ClassicSchema.FieldLayout;
import com.epsilon.classic.input.ClassicVocabulary.ActionType;

/**
 * 方策ヘッドと方策のグラフが読む、合法手の列のオフセット。合法手には整数の列と、手どうしの位置を対応付ける列がある。
 * オフセットはどちらも、合法手 1 つ分の要素の中での位置。
 *
 * @param stride 整数の列の数（合法手 1 つ分の要素数）
 * @param id 行動の番号 ＋ 1 の列（0 は合法手の無い位置）
 * @param type 行動の種別 ＋ 1 の列
 * @param group 大分類 ＋ 1 の列
 * @param primaryTile 主な牌の種類 ＋ 1 の列
 * @param routeStride 位置を対応付ける列の数（合法手 1 つ分の要素数）
 * @param representative 同じ牌を切る打牌（物理の打牌）のうち、代表の手で 1 になる列
 * @param representativeSlot 代表の手の位置 ＋ 1 の列
 * @param damaSlot 同じ物理の打牌のうち、ダマの打牌の位置 ＋ 1 の列
 * @param riichiSlot 同じ物理の打牌のうち、立直の打牌の位置 ＋ 1 の列
 */
record ActionColumns(
    int stride,
    int id,
    int type,
    int group,
    int primaryTile,
    int routeStride,
    int representative,
    int representativeSlot,
    int damaSlot,
    int riichiSlot) {

  private static final ActionColumns[] BY_ARCHITECTURE = columns();

  /** スキーマの列のオフセット。アーキテクチャごとに作り置いたものを返す。 */
  static ActionColumns of(ClassicSchema schema) {
    return BY_ARCHITECTURE[schema.architecture().ordinal()];
  }

  private static ActionColumns[] columns() {
    Architecture[] architectures = Architecture.values();
    ActionColumns[] columns = new ActionColumns[architectures.length];
    for (Architecture architecture : architectures) {
      columns[architecture.ordinal()] = build(ClassicSchema.of(architecture));
    }
    return columns;
  }

  private static ActionColumns build(ClassicSchema schema) {
    FieldLayout<ActionInt> action = schema.layout(ActionInt.class);
    FieldLayout<ActionRoute> route = schema.layout(ActionRoute.class);
    return new ActionColumns(
        action.stride(),
        action.offset(ActionInt.ID),
        action.offset(ActionInt.TYPE),
        action.offset(ActionInt.GROUP),
        action.offset(ActionInt.PRIMARY_TILE),
        route.stride(),
        route.offset(ActionRoute.DISCARD_IDENTITY_REPRESENTATIVE),
        route.offset(ActionRoute.DISCARD_REPRESENTATIVE_SLOT),
        route.offset(ActionRoute.DAMA_SLOT),
        route.offset(ActionRoute.RIICHI_SLOT));
  }

  /** 種別の列に書く値（種別の番号 ＋ 1）。 */
  static int stored(ActionType type) {
    return type.ordinal() + 1;
  }
}
