package com.epsilon.prism.collect;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.DeviationTarget;
import java.util.Arrays;
import java.util.Objects;

/**
 * 根の判断のクラス 1 つに対する中立ロールインの決まり。主経路でそのクラスの判断になったら、確率 β で q から d を引いて写しを作る。写しの中
 * では、逸脱した席の後の判断を、その判断のクラスの率 ρ_br でラベル行にする。ラベル行は写しごとに上限まで。
 *
 * @param rate 写しを作る率 β
 * @param epsilonQ 逸脱の分布 q に混ぜる一様分布の割合 ε_q
 * @param target 逸脱 d を引く手の集合
 * @param laterRates 写しの中で逸脱した席の後の判断をラベル行にする率 ρ_br。その判断のクラスごとに、クラスの番号の順に並べる
 * @param maxLabels 写しごとのラベル行の上限
 */
public record RollInRule(double rate, double epsilonQ, DeviationTarget target, double[] laterRates, int maxLabels) {

  /** 写しを作らない決まり。そのルール（四人打ち・三人打ち）に無いクラスに使う。 */
  public static final RollInRule OFF = uniform(0, 0, DeviationTarget.ANY, 0, 0);

  public RollInRule {
    Objects.requireNonNull(target, "target");
    if (laterRates.length != DecisionClass.COUNT)
      throw new IllegalArgumentException("The copy label rates need one entry per class");
    requireRate(rate);
    requireRate(epsilonQ);
    for (double value : laterRates) requireRate(value);
    if (maxLabels < 0) throw new IllegalArgumentException("Invalid copy label limit " + maxLabels);
    laterRates = laterRates.clone();
  }

  /** 後の判断のクラスに依らず、ρ_br が同じ決まり。 */
  public static RollInRule uniform(
      double rate, double epsilonQ, DeviationTarget target, double laterRate, int maxLabels) {
    double[] laterRates = new double[DecisionClass.COUNT];
    Arrays.fill(laterRates, laterRate);
    return new RollInRule(rate, epsilonQ, target, laterRates, maxLabels);
  }

  /** 写しの中で、逸脱した席の判断のうちクラスが {@code c} のものをラベル行にする率 ρ_br。 */
  public double laterRate(DecisionClass c) {
    return laterRates[c.ordinal()];
  }

  private static void requireRate(double rate) {
    if (!(rate >= 0 && rate <= 1)) throw new IllegalArgumentException("Invalid rate " + rate);
  }
}
