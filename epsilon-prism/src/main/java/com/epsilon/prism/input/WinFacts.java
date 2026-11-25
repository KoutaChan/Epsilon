package com.epsilon.prism.input;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.engine.WinSettlementProjection.SettlementAssumption;
import com.epsilon.mahjong.rules.DealerOutcome;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;

/**
 * ロン・ツモの手について、厳密な入力の元になる事実を作る。元にするのは、エンジンがその場で出した和了の判定（点の移動、包、裏ドラの
 * 対象か、和了の後の順位）で、判断の記録に写してある。四人打ち・三人打ちの入力の形式は、この事実を厳密な入力の列に写す。
 *
 * <p>和了の後の点は、今の点に判断の記録の点の移動と本場を足したもの。ツモの点の移動は本場を含むが、ロンの点の移動は本場を含まないので、
 * ロンでは本場の分（{@link Variant#ronHonba} × 本場）を足す。ロンの本場を払うのは、包で放銃者のほかにも払う席があればその席、なければ
 * 放銃者。終局かどうかは、その点に場の供託を和了者の分として足した精算の後の点で、{@link Progression#finishes} が決める。順位には判断の
 * 記録にある和了の後の順位を使う。得点は判断の時点からの自分の点の変化（本場・供託を含み、裏ドラを含まない）。連荘は、親の和了で半荘が
 * 続くとき。
 */
public final class WinFacts {

  private final RankUtility utility;
  private int[] absolute = new int[0];
  private boolean ends;
  private float endUtility;
  private int rank;
  private int points;
  private boolean renchan;
  private boolean pao;
  private boolean uraEligible;

  /**
   * 終局の効用を渡して作る。
   *
   * @param utility 終局の効用
   */
  public WinFacts(RankUtility utility) {
    this.utility = utility;
  }

  /**
   * 和了 1 つについて事実を作る。
   *
   * @param o 判断の記録
   * @param win 判断の記録の中で何番目の和了か
   */
  public void load(Observation o, int win) {
    Variant variant = o.variant();
    if (absolute.length != variant.seats()) {
      absolute = new int[variant.seats()];
    }
    ends = settlesGame(o, win, absolute);
    boolean dealer = o.dealer() == 0;
    rank = o.winRank(win, 0);
    points = absolute[o.observer()] - o.score(0);
    endUtility = ends ? (float) utility.value(variant, rank) : 0f;
    renchan = dealer && !ends;
    pao = o.winPao(win);
    uraEligible = o.winUraEligible(win);
  }

  /** この和了で半荘が終わるか。 */
  public boolean ends() {
    return ends;
  }

  /** 半荘が終わるときの効用（続くなら 0）。 */
  public float utility() {
    return endUtility;
  }

  /** 和了の後の自分の順位（0 始まり）。 */
  public int rank() {
    return rank;
  }

  /** 得点（判断の時点からの自分の点の変化。本場・供託を含み、裏ドラを含まない）。 */
  public int points() {
    return points;
  }

  /** 親の和了で、半荘が続くか（連荘）。 */
  public boolean renchan() {
    return renchan;
  }

  /** 包が成り立つか。 */
  public boolean pao() {
    return pao;
  }

  /** 裏ドラの対象になるか（立直しているか）。 */
  public boolean uraEligible() {
    return uraEligible;
  }

  /** 観測にある精算をしたときの点を絶対席の順に書き、半荘が終わるかを返す。 */
  public static boolean settlesGame(Observation o, int win, int[] absolute) {
    Variant variant = o.variant();
    int seats = variant.seats();
    int observer = o.observer();
    for (int seat = 0; seat < seats; seat++) {
      int abs = observer + seat;
      if (abs >= seats) {
        abs -= seats;
      }
      absolute[abs] = o.score(seat) + o.winPayment(win, seat);
    }
    absolute[observer] += o.kyotaku() * Variant.DEPOSIT;
    if (o.winSettlement(win) == SettlementAssumption.SOLE_RON && o.honba() > 0) {
      int discarder = o.lastEventSeat();
      int payer = discarder;
      for (int seat = 1; seat < seats; seat++) {
        if (seat != discarder && o.winPayment(win, seat) < 0) {
          payer = seat;
        }
      }
      int honba = variant.ronHonba() * o.honba();
      int absPayer = observer + payer;
      if (absPayer >= seats) {
        absPayer -= seats;
      }
      absolute[observer] += honba;
      absolute[absPayer] -= honba;
    }
    return Progression.finishes(
        variant,
        o.kyoku(),
        absolute,
        o.dealer() == 0 ? DealerOutcome.CONTINUES : DealerOutcome.PASSES);
  }
}
