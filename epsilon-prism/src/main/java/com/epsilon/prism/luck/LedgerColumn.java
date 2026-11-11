package com.epsilon.prism.luck;

import com.epsilon.prism.noise.LedgerGroup;

/**
 * 運の精算の帳簿の列。宣言の順が行に書く列の順で、行の形式の一部になる。どの列も、帳簿を受け取る席から見た向きで増分を足す。点の列の単位は
 * 千点、方策の列は効用で、ほかの列は牌の数か、ある牌を引いたかを表す 0 か 1 の増分。
 */
public enum LedgerColumn {
  /** 自分が一向聴で、向聴を下げる牌を引いたか。 */
  OWN_IMPROVE_1(LedgerGroup.IMPROVE, "ownImprove1"),
  /** 自分が二向聴で、向聴を下げる牌を引いたか。 */
  OWN_IMPROVE_2(LedgerGroup.IMPROVE, "ownImprove2"),
  /** 自分が三向聴以上で、向聴を下げる牌を引いたか。 */
  OWN_IMPROVE_3(LedgerGroup.IMPROVE, "ownImprove3"),
  /** 聴牌の自分がツモで和了して得る点。 */
  OWN_TSUMO(LedgerGroup.POINTS, "ownTsumo"),
  /** 立直した自分が引いた牌を、他家に栄和されて払う点（打つしかない）。 */
  OWN_FORCED_DEAL_IN(LedgerGroup.POINTS, "ownForcedDealIn"),
  /** 立直していない自分が引いた牌で、他家に栄和されうる点。 */
  OWN_DRAWN_DANGER(LedgerGroup.POINTS, "ownDrawnDanger"),
  /** 一向聴の他家が、向聴を下げる牌を引いたか。 */
  OPP_IMPROVE_1(LedgerGroup.IMPROVE, "oppImprove1"),
  /** 二向聴以上の他家が、向聴を下げる牌を引いたか。 */
  OPP_IMPROVE_2(LedgerGroup.IMPROVE, "oppImprove2"),
  /** 他家のツモの和了で自分が払う点。 */
  OPP_TSUMO(LedgerGroup.POINTS, "oppTsumo"),
  /** 立直した他家が引いた牌を、自分が栄和して得る点。 */
  OPP_FORCED_TO_SELF(LedgerGroup.POINTS, "oppForcedToSelf"),
  /** 立直した他家が引いた牌を、ほかの席が栄和して得る点。 */
  OPP_FORCED_TO_OTHER(LedgerGroup.POINTS, "oppForcedToOther"),
  /** 立直していない他家が引いた牌を、自分が栄和できる点。 */
  OPP_DRAWN_CHANCE(LedgerGroup.POINTS, "oppDrawnChance"),
  /** 槓ドラの表示で増えた、自分のドラの数。 */
  DORA_SELF(LedgerGroup.DORA, "doraSelf"),
  /** 槓ドラの表示で増えた、他家のドラの数の和。 */
  DORA_OPPONENTS(LedgerGroup.DORA, "doraOpponents"),
  /** 立直の和了の裏ドラの数（和了者・放銃者・ツモで払う席の重みつき）。 */
  URA(LedgerGroup.URA, "ura"),
  /** 自分の判断で、方策から引いた手の運。 */
  POLICY_SELF(LedgerGroup.POLICY, "policySelf"),
  /** 他家の判断で、方策から引いた手の運。 */
  POLICY_OPPONENT(LedgerGroup.POLICY, "policyOpponent");

  /** 列の数。 */
  public static final int COUNT = values().length;

  private final LedgerGroup group;
  private final String reportName;

  LedgerColumn(LedgerGroup group, String reportName) {
    this.group = group;
    this.reportName = reportName;
  }

  /** 計算と補正の単位にする組。 */
  public LedgerGroup group() {
    return group;
  }

  /** 報告と監査に書く名前。 */
  public String reportName() {
    return reportName;
  }
}
