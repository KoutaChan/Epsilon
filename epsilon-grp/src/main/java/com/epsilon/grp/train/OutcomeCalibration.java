package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.grp.HandOutcomes;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeSection;
import com.epsilon.grp.OutcomeTensor;
import com.epsilon.grp.OutcomeTerm;
import com.epsilon.grp.OutcomeWorkspace;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntPredicate;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * 局の結果のネットワーク q(o|s) の較正を、OutcomeRows の行や局末で測る。ネットワークに温度 T を掛けるとは、ロジット z を z/T に
 * することで、最後の層の重みとバイアスを T で割って行う。終わり方の softmax、打点の softmax、立直の σ の全てに同じ T を使う。T が
 * 大きいほど分布は平らになる。NLL を最小にする T が 1 より大きければ、ネットワークは自信を持ちすぎている。
 */
public final class OutcomeCalibration {

  /** 温度を探す範囲の下限と上限。 */
  private static final double LOWEST_TEMPERATURE = 0.25;

  private static final double HIGHEST_TEMPERATURE = 4;

  /** log T の区間がこの幅まで狭まれば、温度の探索を止める。 */
  private static final double LOG_TOLERANCE = 1e-3;

  /** 黄金分割の比 (√5 − 1)/2。 */
  private static final double GOLDEN = (Math.sqrt(5) - 1) / 2;

  /** NLL を並列に求めるとき、1 チャンクに入れる行の数。 */
  private static final int CHUNK_ROWS = 2_048;

  /** 信頼性の表で、終わり方の予測の確率を分ける区間の境。区間は下の境を含み、上の境を含まない（最後の区間は 1 も含む）。 */
  private static final double[] EDGES = {0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1};

  /** 外れとみなす |z| のしきい値。|z| がこれを越えた分類を外れとする。 */
  static final double OUTLIER_Z = 3;

  private OutcomeCalibration() {}

  /** 測る行と、それを測るネットワーク。 */
  public record Part(OutcomeRows rows, OutcomeNetwork network) {}

  /**
   * 局末で測った、局の結果のネットワークの較正。
   *
   * @param handEnds 測った局末の数
   * @param skipped 選んだ局末のうち、遷移を局の結果のネットワークで表せず測らなかった数
   * @param nll 局末あたりの周辺 NLL −log Σ_o q(o|s)（温度 1）
   * @param temperature その NLL を最小にする温度
   * @param outliers 区分ごとのセルで、|z| が 3 を越えた分類（|z| の大きい順。{@code OutcomeClasses.Result.outliers} と同じ）。
   *     全ての区分をまとめたセルと、セルで予測した回数が少ない分類は入れない
   */
  public record Check(
      long handEnds, long skipped, double nll, double temperature, List<OutcomeClasses.Outlier> outliers) {

    public Check {
      outliers = List.copyOf(outliers);
    }
  }

  /**
   * 終わり方の確率の信頼性。実際の終わり方を数えるときは、行の遷移を再現する結果それぞれに q に比例した重みを付け、終わり方ごとに
   * 足す。再現する結果の終わり方が 1 種類だけなら、その終わり方が 1、ほかは 0 になる。
   *
   * @param rows 行の数
   * @param ambiguousRows 遷移を 2 つ以上の終わり方で再現できる行の数
   * @param bins 行と終わり方の組を、予測の確率の区間で分けた表
   * @param sections 区分（ツモ、ロン、ダブロン、荒牌、途中流局）ごとに、予測の確率の和の平均と、実際の率を出した表
   */
  public record Reliability(long rows, long ambiguousRows, List<Bin> bins, List<Bin> sections) {}

  /**
   * 信頼性の表の行 1 つ。
   *
   * @param name 区分の名前。確率の区間の行では空
   * @param low 確率の区間の下の境。区分の行では NaN
   * @param high 確率の区間の上の境。区分の行では NaN
   * @param pairs 数えた組（行と終わり方、または行と区分）の数
   * @param predicted 予測の確率の平均
   * @param realized 実際の率
   * @param standardError 実際の率の、行をクラスタとした標準誤差
   */
  public record Bin(
      String name, double low, double high, long pairs, double predicted, double realized, double standardError) {}

  /**
   * 温度を掛けたネットワークを作る。最後の層の重みとバイアスを T で割り、float に丸める。T が 1 なら重みは変わらない。
   *
   * @throws IllegalArgumentException 温度が正の有限の値でないとき
   */
  public static OutcomeNetwork tempered(OutcomeNetwork network, double temperature) {
    if (!(temperature > 0 && temperature < Double.POSITIVE_INFINITY)) {
      throw new IllegalArgumentException("A temperature must be positive and finite: " + temperature);
    }
    float[] weights = network.weights();
    int[] starts = OutcomeTensor.starts(network.shape());
    for (int index = starts[OutcomeTensor.W3.ordinal()]; index < starts[OutcomeTensor.B3.ordinal() + 1]; index++) {
      weights[index] = (float) (weights[index] / temperature);
    }
    return OutcomeNetwork.of(network.shape(), weights);
  }

  /**
   * 全ての組（行とネットワーク）の行について、重み付きの周辺 NLL Σ w·(−log Σ_o q(o|s)) / Σ w を求める。和と重みは組をまたいで
   * 足す。行ごとの式は {@code OutcomeRows.nll} と同じ。行を決まった数ずつのチャンクに分けて並列に求め、チャンクの順に足すので、
   * 値はスレッドの数に依らない。
   *
   * @param temperature ネットワークに掛ける温度
   * @param workspaces 作業領域を作る関数。計算するスレッドごとに 1 度呼ぶ。作業領域の形は、全ての組のネットワークと同じにする
   * @return 行が無ければ NaN
   * @throws IllegalArgumentException 組のネットワークのルールが、その組の行と違うとき
   */
  public static double nll(List<Part> parts, double temperature, Supplier<OutcomeWorkspace> workspaces) {
    ThreadLocal<OutcomeWorkspace> locals = ThreadLocal.withInitial(workspaces);
    double sum = 0;
    double weight = 0;
    for (Part part : parts) {
      OutcomeRows rows = part.rows();
      if (part.network().variant() != rows.variant()) {
        throw new IllegalArgumentException(
            "A " + part.network().variant() + " network for " + rows.variant() + " rows");
      }
      OutcomeNetwork network = tempered(part.network(), temperature);
      double[] sums = new double[Math.ceilDiv(rows.size(), CHUNK_ROWS)];
      IntStream.range(0, sums.length).parallel().forEach(chunk -> {
        OutcomeWorkspace workspace = locals.get();
        double[] logs = new double[rows.maxOutcomes()];
        int end = Math.min(rows.size(), (chunk + 1) * CHUNK_ROWS);
        double partial = 0;
        for (int row = chunk * CHUNK_ROWS; row < end; row++) {
          network.forward(GrpState.ofKey(rows.variant(), rows.keys[row]), workspace);
          partial -= rows.weightOf(row) * rows.logLikelihood(row, workspace, logs);
        }
        sums[chunk] = partial;
      });
      for (double partial : sums) {
        sum += partial;
      }
      weight += rows.totalWeight();
    }
    return weight > 0 ? sum / weight : Double.NaN;
  }

  /**
   * 全ての組の NLL（{@link #nll(List, double, Supplier)}）を最小にする温度。log T を [log 0.25, log 4] の中で、区間の幅が
   * 0.001 以下になるまで黄金分割探索で探す。
   *
   * @return 行が無ければ NaN
   */
  public static double temperature(List<Part> parts, Supplier<OutcomeWorkspace> workspaces) {
    if (parts.stream().allMatch(part -> part.rows().size() == 0)) {
      return Double.NaN;
    }
    return minimize(temperature -> nll(parts, temperature, workspaces));
  }

  /**
   * 選んだ局末について、周辺 NLL −log Σ_o q(o|s) の局末あたりの平均を求める。o は遷移を再現する結果で、重みは使わない。測るのは、
   * 遷移を再現する結果を持つ局末（NextHands の種類が {@code OUTCOME} か {@code ENDING}）だけ。局末ごとの対数尤度を並列に求めて
   * 局末の順に足すので、値はスレッドの数に依らない。
   *
   * @param selected 測る局末を選ぶ関数（複数のスレッドから同時に呼ぶ）
   * @param temperature ネットワークに掛ける温度
   * @param workspaces 作業領域を作る関数。計算するスレッドごとに 1 度呼ぶ
   * @return 測る局末が無ければ NaN
   * @throws IllegalArgumentException ネットワークのルールが局末と違うとき
   */
  public static double nll(
      NextHands next,
      IntPredicate selected,
      OutcomeNetwork network,
      double temperature,
      Supplier<OutcomeWorkspace> workspaces) {
    double sum = 0;
    long count = 0;
    for (double log : next.logLikelihoods(selected, tempered(network, temperature), workspaces)) {
      if (!Double.isNaN(log)) {
        sum -= log;
        count++;
      }
    }
    return count > 0 ? sum / count : Double.NaN;
  }

  /**
   * 選んだ局末の NLL（{@link #nll(NextHands, IntPredicate, OutcomeNetwork, double, Supplier)}）を最小にする温度。探し方は
   * {@link #temperature(List, Supplier)} と同じ。
   *
   * @return 測る局末が無ければ NaN
   */
  public static double temperature(
      NextHands next, IntPredicate selected, OutcomeNetwork network, Supplier<OutcomeWorkspace> workspaces) {
    HandEnds ends = next.ends();
    boolean measured = false;
    for (int h = 0; h < ends.handEnds() && !measured; h++) {
      measured = next.found(h) > 0 && selected.test(h);
    }
    if (!measured) {
      return Double.NaN;
    }
    return minimize(temperature -> nll(next, selected, network, temperature, workspaces));
  }

  /**
   * 選んだ局末で、局の結果のネットワークの較正を測る。出すのは、周辺 NLL、それを最小にする温度、席から見た分類と親から見た分類の
   * 外れ（OutcomeClasses）。局を数えたファイルから作った局末（{@code NextHands.of(HandEnds, CountedHands)}）では、遷移を
   * 再現する結果は記録した 1 つだけなので、実際の率は記録した結果だけで決まる。
   *
   * @param selected 測る局末を選ぶ関数（複数のスレッドから同時に呼ぶ）
   * @throws IllegalArgumentException ネットワークのルールが局末と違うとき、ネットワークが上の席を抜くのに要る打点のフィールドを
   *     持つとき
   */
  public static Check check(NextHands next, IntPredicate selected, OutcomeNetwork network) {
    Supplier<OutcomeWorkspace> workspaces = () -> OutcomeWorkspace.of(network.shape());
    double nll = nll(next, selected, network, 1, workspaces);
    OutcomeClasses.Result classes = OutcomeClasses.measure(next, h -> network, selected, workspaces);
    return new Check(
        classes.handEnds(),
        classes.skipped(),
        nll,
        temperature(next, selected, network, workspaces),
        classes.outliers(OUTLIER_Z));
  }

  /** NLL を最小にする温度。log T を [log 0.25, log 4] の中で、区間の幅が 0.001 以下になるまで黄金分割探索で探す。 */
  private static double minimize(DoubleUnaryOperator nll) {
    double low = Math.log(LOWEST_TEMPERATURE);
    double high = Math.log(HIGHEST_TEMPERATURE);
    double left = high - GOLDEN * (high - low);
    double right = low + GOLDEN * (high - low);
    double leftNll = nll.applyAsDouble(Math.exp(left));
    double rightNll = nll.applyAsDouble(Math.exp(right));
    while (high - low > LOG_TOLERANCE) {
      if (leftNll <= rightNll) {
        high = right;
        right = left;
        rightNll = leftNll;
        left = high - GOLDEN * (high - low);
        leftNll = nll.applyAsDouble(Math.exp(left));
      } else {
        low = left;
        left = right;
        leftNll = rightNll;
        right = low + GOLDEN * (high - low);
        rightNll = nll.applyAsDouble(Math.exp(right));
      }
    }
    return Math.exp((low + high) / 2);
  }

  /**
   * 終わり方の確率の信頼性を測る。行と終わり方の組を全て、温度 1 での予測の確率で区間に分け、区間ごとに予測の平均と実際の率を
   * 出す。区分ごとには、行ごとにその区分の確率を足した値の平均と、実際の率を出す。行の重みは使わない。
   *
   * @param workspaces 作業領域を作る関数。1 度だけ呼ぶ
   */
  public static Reliability reliability(List<Part> parts, Supplier<OutcomeWorkspace> workspaces) {
    int bins = EDGES.length - 1;
    OutcomeSection[] sections = OutcomeSection.values();
    Clusters realized = new Clusters(bins);
    Clusters sectionRealized = new Clusters(sections.length);
    double[] predicted = new double[bins];
    double[] sectionPredicted = new double[sections.length];
    double[] sectionShares = new double[sections.length];
    OutcomeWorkspace workspace = workspaces.get();
    long count = 0;
    long ambiguous = 0;
    for (Part part : parts) {
      OutcomeRows rows = part.rows();
      HandOutcomes hands = HandOutcomes.of(rows.variant());
      int mainStart = hands.termStart(OutcomeTerm.MAIN);
      double[] shares = new double[hands.mainCount()];
      boolean[] seen = new boolean[hands.mainCount()];
      double[] logs = new double[rows.maxOutcomes()];
      for (int row = 0; row < rows.size(); row++) {
        part.network().forward(GrpState.ofKey(rows.variant(), rows.keys[row]), workspace);
        double likelihood = rows.logLikelihood(row, workspace, logs);
        Arrays.fill(shares, 0);
        Arrays.fill(seen, false);
        int distinct = 0;
        for (int k = 0; k < rows.firstOutcome[row + 1] - rows.firstOutcome[row]; k++) {
          int main = rows.terms[(rows.firstOutcome[row] + k) * rows.stride] - mainStart;
          if (!seen[main]) {
            seen[main] = true;
            distinct++;
          }
          shares[main] += StrictMath.exp(logs[k] - likelihood);
        }
        if (distinct > 1) {
          ambiguous++;
        }
        Arrays.fill(sectionShares, 0);
        for (int main = 0; main < shares.length; main++) {
          double probability = workspace.probability(hands.mainTerm(main));
          int bin = bin(probability);
          predicted[bin] += probability;
          realized.add(bin, shares[main]);
          int section = hands.section(main).ordinal();
          sectionPredicted[section] += probability;
          sectionShares[section] += shares[main];
        }
        for (int section = 0; section < sections.length; section++) {
          sectionRealized.add(section, sectionShares[section]);
        }
        realized.endGame();
        sectionRealized.endGame();
        count++;
      }
    }
    List<Bin> table = new ArrayList<>(bins);
    for (int bin = 0; bin < bins; bin++) {
      table.add(row("", EDGES[bin], EDGES[bin + 1], realized, bin, predicted[bin]));
    }
    List<Bin> bySection = new ArrayList<>(sections.length);
    for (OutcomeSection section : sections) {
      int index = section.ordinal();
      bySection.add(row(section.name(), Double.NaN, Double.NaN, sectionRealized, index, sectionPredicted[index]));
    }
    return new Reliability(count, ambiguous, table, bySection);
  }

  /**
   * 表の行 1 つを作る。
   *
   * @param cell 実際の率を足したセル
   * @param predicted 予測の確率の和
   */
  private static Bin row(String name, double low, double high, Clusters realized, int cell, double predicted) {
    double pairs = realized.count(cell);
    return new Bin(
        name, low, high, (long) pairs, predicted / pairs, realized.mean(cell), Math.sqrt(realized.variance(cell)));
  }

  /** 予測の確率が入る区間の番号。 */
  private static int bin(double probability) {
    int bin = 0;
    while (bin < EDGES.length - 2 && probability >= EDGES[bin + 1]) {
      bin++;
    }
    return bin;
  }
}
