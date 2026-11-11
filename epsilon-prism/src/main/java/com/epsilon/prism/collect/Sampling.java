package com.epsilon.prism.collect;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.core.ClassTables;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Openness;
import com.epsilon.prism.noise.NoiseSettings;
import com.epsilon.prism.noise.RateSettings;
import com.epsilon.prism.settings.LabelSettings;
import com.epsilon.prism.settings.PrismSettings;
import java.util.Arrays;
import java.util.Objects;

/**
 * 収集の率。率の表は判断のクラスの番号の順に並べる。{@link #OFF} は行も写しも作らない率。
 *
 * @param holdoutFraction ホールドアウト（HOLDOUT）にする半荘の割合
 * @param labelRates 主経路のラベル行の率 ρ。判断する席の手の形の番号ごとに、クラスの番号の順の表を持つ
 * @param measureAll クラスごとに、候補集合を全ての合法手（同値な手はまとめて代表 1 つ）にするか
 * @param maxCandidates 候補集合 C(o) の大きさの上限（同値な手をまとめた後）
 * @param epsilonC 候補集合に一様に引いた 1 手を足す確率 ε_C
 * @param rollIn 根の判断のクラスごとの中立ロールイン（クラスの番号の順）
 */
public record Sampling(
    double holdoutFraction,
    double[][] labelRates,
    boolean[] measureAll,
    int maxCandidates,
    double epsilonC,
    RollInRule[] rollIn) {
  /** 行も写しも作らない（評価）。 */
  public static final Sampling OFF = off();

  public Sampling {
    if (!(holdoutFraction >= 0 && holdoutFraction < 1))
      throw new IllegalArgumentException("holdoutFraction " + holdoutFraction);
    if (labelRates.length != Openness.COUNT
        || measureAll.length != DecisionClass.COUNT
        || rollIn.length != DecisionClass.COUNT)
      throw new IllegalArgumentException("The class tables need one entry per class");
    double[][] rates = new double[Openness.COUNT][];
    for (int hand = 0; hand < Openness.COUNT; hand++) {
      if (labelRates[hand].length != DecisionClass.COUNT)
        throw new IllegalArgumentException("The class tables need one entry per class");
      for (double rate : labelRates[hand]) requireRate(rate);
      rates[hand] = labelRates[hand].clone();
    }
    requireRate(epsilonC);
    for (RollInRule rule : rollIn) Objects.requireNonNull(rule, "rollIn");
    if (maxCandidates < 2 || maxCandidates > 32)
      throw new IllegalArgumentException("Invalid candidate limit " + maxCandidates);
    labelRates = rates;
    measureAll = measureAll.clone();
    rollIn = rollIn.clone();
  }

  /** 率が全て 0 で、候補集合の上限が 2、中立ロールインが全て {@link RollInRule#OFF} の率。 */
  private static Sampling off() {
    RollInRule[] rollIn = new RollInRule[DecisionClass.COUNT];
    Arrays.fill(rollIn, RollInRule.OFF);
    return new Sampling(0, new double[Openness.COUNT][DecisionClass.COUNT], new boolean[DecisionClass.COUNT], 2, 0, rollIn);
  }

  /**
   * 設定から収集の率を作る。主経路のラベル行の率は noise.rates から取り、倍率を掛け、副露している席には open の率を使う。全ての合法手を
   * 測るクラスは noise.rates の allActions から、中立ロールインは labels から取る。根の判断のクラスごとの写しの設定は、書いたキーだけが
   * セクションの値を上書きする。そのルール（四人打ち・三人打ち）に無いクラスでは、ラベル行も写しも作らない。
   */
  public static Sampling of(PrismSettings settings) {
    Variant variant = settings.run().variant();
    LabelSettings labels = settings.labels();
    NoiseSettings noise = settings.noise();
    RateSettings rates = noise.rates();
    double[][] labelRates = new double[Openness.COUNT][DecisionClass.COUNT];
    boolean[] measureAll = new boolean[DecisionClass.COUNT];
    RollInRule[] rollIn = new RollInRule[DecisionClass.COUNT];
    Arrays.fill(rollIn, RollInRule.OFF);
    for (DecisionClass c : ClassTables.classes(variant)) {
      for (Openness hand : Openness.values()) labelRates[hand.ordinal()][c.ordinal()] = rates.rate(c, hand);
      measureAll[c.ordinal()] = rates.allActions().contains(c);
      double[] later = new double[DecisionClass.COUNT];
      for (DecisionClass next : DecisionClass.values()) later[next.ordinal()] = labels.copyLabelRate(c, next);
      rollIn[c.ordinal()] =
          new RollInRule(labels.rollin().get(c), labels.deviationFloor(c), labels.deviation(c), later, labels.maxCopyLabels(c));
    }
    return new Sampling(
        labels.holdout(), labelRates, measureAll, labels.maxCandidates(), labels.uniformCandidate(), rollIn);
  }

  /** クラス {@code c} の判断で全ての合法手を測るか。 */
  public boolean measuresAll(DecisionClass c) {
    return measureAll[c.ordinal()];
  }

  /**
   * 主経路の判断のラベル行の率 ρ。
   *
   * @param c 判断のクラス
   * @param hand 判断する席の手の形
   */
  public double labelRate(DecisionClass c, Openness hand) {
    return labelRates[hand.ordinal()][c.ordinal()];
  }

  /** 根の判断のクラス {@code root} の中立ロールイン。 */
  public RollInRule rollIn(DecisionClass root) {
    return rollIn[root.ordinal()];
  }

  private static void requireRate(double rate) {
    if (!(rate >= 0 && rate <= 1)) throw new IllegalArgumentException("Invalid rate " + rate);
  }
}
