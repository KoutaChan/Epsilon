package com.epsilon.prism.eval;

import com.epsilon.core.PublicObservation;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.policy.SeatRouter;
import com.epsilon.util.SeedMixer;
import java.util.EnumSet;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;

/**
 * 評価の混成の席（測定専用）。類 classes の判断だけ other（Nano）に打たせ、ほかは base（PRISM）が打つ。openHand なら、自分の手に副露（門前を
 * 崩す面子）がある間の判断も全て other が打つ（鳴いた局の残りも Nano。鳴きを other に任せる設定では、副露があることは other が鳴いたことと同じ）。
 * Nano は評価と測定だけに使い、学習には入れない。
 *
 * <p>分け方: closedOnly なら類の判断のうち自分の手が門前の間だけを任せ、fraction &lt; 1 なら類の判断のうち硬貨（半荘・席・局・巡目と
 * 河の枚数からのハッシュ。牌の中身は見ない）が fraction 未満のものだけを任せる。
 */
public final class HybridSeat implements PolicyService, SeatRouter {
  private final PolicyService base;
  private final PolicyService other;
  private final Set<DecisionClass> classes;
  private final boolean openHand;
  private final boolean closedOnly;
  private final double fraction;
  private final LongAdder[] routed = {new LongAdder(), new LongAdder()};

  /**
   * @param base ふだん打つ方策（PRISM の版）
   * @param other classes の判断を打つ方策（Nano）
   * @param classes other に任せる類
   * @param openHand 副露がある間の判断も other に任せるか
   */
  public HybridSeat(PolicyService base, PolicyService other, Set<DecisionClass> classes, boolean openHand) {
    this(base, other, classes, openHand, false, 1);
  }

  /**
   * @param closedOnly 類の判断のうち、自分の手が門前の間だけを other に任せるか
   * @param fraction 類の判断のうち other に任せる割合（硬貨。1 なら全て）
   */
  public HybridSeat(
      PolicyService base,
      PolicyService other,
      Set<DecisionClass> classes,
      boolean openHand,
      boolean closedOnly,
      double fraction) {
    if (!(fraction >= 0 && fraction <= 1)) throw new IllegalArgumentException("fraction must be in [0, 1]: " + fraction);
    this.closedOnly = closedOnly;
    this.fraction = fraction;
    this.base = base;
    this.other = other;
    this.classes = classes.isEmpty() ? EnumSet.noneOf(DecisionClass.class) : EnumSet.copyOf(classes);
    this.openHand = openHand;
  }

  /** 席 seat の手に門前を崩す面子があるか。 */
  static boolean open(PublicObservation observation, int seat) {
    var hand = observation.hand(seat);
    for (int i = 0; i < hand.meldCount(); i++) if (!hand.meld(i).preservesMenzen()) return true;
    return false;
  }

  @Override
  public PolicyService route(PolicyRequest request) {
    boolean toOther =
        (classes.contains(request.decisionClass())
                && !(closedOnly && open(request.observation(), request.seat()))
                && (fraction >= 1 || coin(request) < fraction))
            || (openHand && open(request.observation(), request.seat()));
    routed[toOther ? 1 : 0].increment();
    return toOther ? other : base;
  }

  /** 判断ごとの硬貨（半荘・席・局・本場・巡目・河の枚数・類から。牌の中身は見ない）。 */
  static double coin(PolicyRequest request) {
    PublicObservation o = request.observation();
    long key = SeedMixer.mix64(request.game() * 0x9E3779B97F4A7C15L + request.seat());
    key = SeedMixer.mix64(key + o.roundIndex() * 1_000_003L + o.honba() * 10_007L);
    key = SeedMixer.mix64(key + o.turnNumber() * 131L + o.totalRiverDahaiCount() * 7L + request.decisionClass().ordinal());
    return CrnSeeds.unit(key);
  }

  /** base が打った判断の数と other が打った判断の数。 */
  public long[] routed() {
    return new long[] {routed[0].sum(), routed[1].sum()};
  }

  @Override
  public int version() {
    return base.version();
  }

  @Override
  public boolean inline() {
    return false;
  }
}
