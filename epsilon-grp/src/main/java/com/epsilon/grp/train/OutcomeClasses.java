package com.epsilon.grp.train;

import com.epsilon.grp.HandOutcomes;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeSection;
import com.epsilon.grp.OutcomeWorkspace;
import com.epsilon.grp.Seats;
import com.epsilon.grp.WinMethod;
import com.epsilon.mahjong.rules.BaseClass;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.IntFunction;
import java.util.function.IntPredicate;
import java.util.function.Supplier;

/**
 * 局の結果のネットワーク q(o|s) が較正できているかを、次の局の結果の分類ごとに測る。分類には、席から見た分類（{@link SeatClass}）と
 * 親から見た分類（{@link DealerClass}）がある。予測は、分類に入る結果の q の和。実際の値は、次の局への遷移を再現できる結果
 * （NextHands）に q に比例した重みを割り振り、そのうち分類に入る分を足したもの。測るのは、局の結果のネットワークが表せる結果で
 * 遷移を再現できた局末（NextHands の種類が {@code OUTCOME} か {@code ENDING}）だけ。包・流し満貫や、どの結果でも再現できない
 * 局末は測らず、数だけ数える。
 *
 * <p>席から見た分類は、区分・その席の今の順位・点差の区間ごとのセルで集める。親から見た分類は、区分・親の今の順位ごとのセルで集める。
 * 点差は HandEnds の点差で、区間は 2,000 点未満・2,000〜5,999 点・6,000 点以上。どちらにも、全ての区分をまとめたセル（区分 −1）を
 * 足す。
 *
 * <p>標準誤差には、q が正しいとしたときの「実際 − 予測」の平均の標準誤差 √(Σ p(1 − p)) / n を使う。p は局末ごとの分類の予測、
 * n はセルに入った数。q が正しければ、局末ごとの「実際 − 予測」は前の局末の結果を知っても期待値が 0 なので、同じ半荘の局末どうしでも
 * 相関しない。「実際 − 予測」のばらつきから求めないのは、セルで一度も起きなかった分類では予測のばらつきしか測れず、標準誤差が
 * 小さく出すぎるため。実際の値を遷移を再現する 2 つ以上の結果に割り振る局末では、この標準誤差は大きめに出る（外れを少なめに数える）。
 */
public final class OutcomeClasses {

  /** 点差の区間の境。境の値は上の区間に入る。 */
  private static final int[] GAP_EDGES = {0, 2_000, 6_000, Integer.MAX_VALUE};

  private static final int GAP_BINS = GAP_EDGES.length - 1;

  /**
   * 外れを調べるのに要る、セルで予測した回数の下限。予測の和 Σp と残りの n − Σp のどちらかがこれより小さい分類は、z を正規分布で
   * 近似できないので調べない。
   */
  private static final double MIN_EXPECTED = 5;

  /** 満貫の打点の番号。これ以上の打点を大きい和了とみなす。 */
  private static final int BIG = BaseClass.MANGAN.ordinal();

  private static final SeatClass[] SEAT_CLASSES = SeatClass.values();
  private static final DealerClass[] DEALER_CLASSES = DealerClass.values();

  /**
   * 席から見た次の局の結果の分類。ツモから流局までの 6 つでは、どの結果もちょうど 1 つの分類に入る。満貫以上の 2 つは、それらと
   * 重ねて数える。
   */
  public enum SeatClass {
    /** その席のツモ和了。 */
    TSUMO,
    /** その席のロン和了（ダブロンの 1 人を含む）。 */
    RON,
    /** その席の放銃。 */
    DEAL_IN,
    /** 他家のツモ和了。 */
    OTHER_TSUMO,
    /** その席が和了も放銃もしていないロン。 */
    OTHER_RON,
    /** 荒牌と途中流局。 */
    DRAW,
    /** その席の満貫以上の和了。 */
    BIG_WIN,
    /** その席の放銃のうち、満貫以上の和了者が 1 人以上いるもの。 */
    BIG_DEAL_IN
  }

  /** 親から見た次の局の結果の分類。どの結果もちょうど 1 つの分類に入る。 */
  public enum DealerClass {
    /** 親の和了（ダブロンの 1 人を含む）。 */
    DEALER_WIN,
    /** 子のツモ和了。 */
    CHILD_TSUMO,
    /** 親が和了していないロン。 */
    CHILD_RON,
    /** 親が聴牌の荒牌。 */
    DEALER_TENPAI,
    /** 親が不聴の荒牌。 */
    DEALER_NOTEN,
    /** 途中流局。 */
    ABORTIVE
  }

  /**
   * 分類 1 つの予測と実際の値。
   *
   * @param name 分類の名前
   * @param predicted 予測の確率の平均
   * @param realized 実際の率
   * @param standardError q が正しいとしたときの、実際 − 予測の平均の標準誤差
   */
  public record Rate(String name, double predicted, double realized, double standardError) {

    public double deviation() {
      return realized - predicted;
    }

    /** （実際 − 予測）/ 標準誤差。 */
    public double z() {
      return deviation() / standardError;
    }
  }

  /** 誰から見た分類か。 */
  public enum View {
    /** 席から見た分類（{@link SeatClass}）。 */
    SEAT,
    /** 親から見た分類（{@link DealerClass}）。 */
    DEALER
  }

  /**
   * 較正の外れ 1 つ。セル 1 つの中の、分類 1 つを指す。
   *
   * @param view 誰から見た分類か
   * @param group 区分
   * @param rank 席から見た分類ではその席の今の順位、親から見た分類では親の今の順位
   * @param lowGap 点差の区間の下の境（含む）。親から見た分類では 0
   * @param highGap 点差の区間の上の境（含まない）。上の境が無い区間と、親から見た分類では {@code Integer.MAX_VALUE}
   * @param rate 分類の予測と実際の値
   */
  public record Outlier(View view, int group, int rank, int lowGap, int highGap, Rate rate) {}

  /**
   * 席から見た分類のセル 1 つ。
   *
   * @param group 区分。全ての区分をまとめたセルは −1
   * @param rank その席の今の順位
   * @param lowGap 点差の区間の下の境（含む）
   * @param highGap 点差の区間の上の境（含まない）。上の境が無ければ {@code Integer.MAX_VALUE}
   * @param seatHands 測った（席, 局末）の組の数
   * @param ending そのうち、種類が {@code ENDING} の局末の組の数
   * @param skipped 測らなかった局末のうち、このセルに入る（席, 局末）の組の数
   * @param rates 分類ごとの予測と実際の値（SeatClass の順）
   */
  public record SeatCell(
      int group, int rank, int lowGap, int highGap, long seatHands, long ending, long skipped, List<Rate> rates) {}

  /**
   * 親から見た分類のセル 1 つ。
   *
   * @param group 区分。全ての区分をまとめたセルは −1
   * @param dealerRank 親の今の順位
   * @param handEnds 測った局末の数
   * @param ending そのうち、種類が {@code ENDING} の局末の数
   * @param skipped 測らなかった局末のうち、このセルに入る数
   * @param rates 分類ごとの予測と実際の値（DealerClass の順）
   */
  public record DealerCell(int group, int dealerRank, long handEnds, long ending, long skipped, List<Rate> rates) {}

  /**
   * 測った結果。
   *
   * @param handEnds 測った局末の数
   * @param ending そのうち、種類が {@code ENDING} の局末の数
   * @param skipped 選んだ局末のうち、遷移を表せず測らなかった数（包・流し満貫・どの結果でも再現できない局末）
   * @param seats 席から見た分類のセル（測った局末の無いセルは除く）
   * @param dealers 親から見た分類のセル（測った局末の無いセルは除く）
   */
  public record Result(long handEnds, long ending, long skipped, List<SeatCell> seats, List<DealerCell> dealers) {

    /**
     * 区分ごとのセルで、|z| がしきい値を越えた分類を返す。並びは |z| の大きい順で、同じなら席から見た分類が先、その中はセルの順。
     * 全ての区分をまとめたセル（区分 −1）は入れない。セルでの予測の和と、その残り（セルに入った数 − 予測の和）のどちらかが 5 より
     * 小さい分類は調べない。
     *
     * @param limit |z| のしきい値
     */
    public List<Outlier> outliers(double limit) {
      List<Outlier> outliers = new ArrayList<>();
      for (SeatCell cell : seats) {
        if (cell.group() < 0) {
          continue;
        }
        for (Rate rate : cell.rates()) {
          if (outlying(rate, cell.seatHands(), limit)) {
            outliers.add(new Outlier(View.SEAT, cell.group(), cell.rank(), cell.lowGap(), cell.highGap(), rate));
          }
        }
      }
      for (DealerCell cell : dealers) {
        if (cell.group() < 0) {
          continue;
        }
        for (Rate rate : cell.rates()) {
          if (outlying(rate, cell.handEnds(), limit)) {
            outliers.add(new Outlier(View.DEALER, cell.group(), cell.dealerRank(), 0, Integer.MAX_VALUE, rate));
          }
        }
      }
      outliers.sort(Comparator.comparingDouble((Outlier outlier) -> -Math.abs(outlier.rate().z())));
      return outliers;
    }

    /**
     * 分類が外れなら true。
     *
     * @param count セルに入った数
     */
    private static boolean outlying(Rate rate, long count, double limit) {
      double expected = rate.predicted() * count;
      return expected >= MIN_EXPECTED && count - expected >= MIN_EXPECTED && Math.abs(rate.z()) > limit;
    }
  }

  private OutcomeClasses() {}

  /**
   * 選んだ局末で測る。
   *
   * @param next 局末ごとの、次の局への遷移を再現する結果
   * @param networks 局末の番号から、その局末を測る局の結果のネットワークを返す関数。どのネットワークも同じ形
   * @param selected 測る局末
   * @param workspaces 作業領域を作る関数。1 度だけ呼ぶ
   * @return 測った結果
   */
  public static Result measure(
      NextHands next,
      IntFunction<OutcomeNetwork> networks,
      IntPredicate selected,
      Supplier<OutcomeWorkspace> workspaces) {
    HandEnds ends = next.ends();
    HandOutcomes hands = HandOutcomes.of(ends.variant);
    int seats = ends.seats;
    int groups = ends.groups;
    int seatClasses = SEAT_CLASSES.length;
    int dealerClasses = DEALER_CLASSES.length;
    // セルの区分には区分の番号をそのまま使い、全ての区分をまとめたセルには区分の数を使う。
    int seatCells = (groups + 1) * seats * GAP_BINS;
    int dealerCells = (groups + 1) * seats;
    long[] seatCount = new long[seatCells];
    long[] dealerCount = new long[dealerCells];
    double[] seatPredicted = new double[seatCells * seatClasses];
    double[] seatRealized = new double[seatCells * seatClasses];
    double[] dealerPredicted = new double[dealerCells * dealerClasses];
    double[] dealerRealized = new double[dealerCells * dealerClasses];
    // q が正しいときの「実際 − 予測」の和の分散 Σ p(1 − p)
    double[] seatVariance = new double[seatCells * seatClasses];
    double[] dealerVariance = new double[dealerCells * dealerClasses];
    long[] seatEnding = new long[seatCells];
    long[] seatSkipped = new long[seatCells];
    long[] dealerEnding = new long[dealerCells];
    long[] dealerSkipped = new long[dealerCells];
    double[] predicted = new double[seats * seatClasses];
    double[] realized = new double[seats * seatClasses];
    double[] predictedDealer = new double[dealerClasses];
    double[] realizedDealer = new double[dealerClasses];
    double[] bigTsumo = new double[seats];
    double[] bigRon = new double[seats];
    short[] terms = new short[hands.maxTerms()];
    double[] logs = new double[8];
    OutcomeWorkspace workspace = workspaces.get();
    long measured = 0;
    long ending = 0;
    long skipped = 0;
    for (int game = 0; game < ends.games(); game++) {
      for (int h = ends.firstHand[game]; h < ends.firstHand[game + 1]; h++) {
        if (!selected.test(h)) {
          continue;
        }
        NextHands.Kind transition = next.kind(h);
        if (transition != NextHands.Kind.OUTCOME && transition != NextHands.Kind.ENDING) {
          skipped++;
          countCells(ends, h, seatSkipped, dealerSkipped);
          continue;
        }
        if (transition == NextHands.Kind.ENDING) {
          ending++;
          countCells(ends, h, seatEnding, dealerEnding);
        }
        measured++;
        networks.apply(h).forward(ends.states[h], workspace);
        for (int seat = 0; seat < seats; seat++) {
          bigTsumo[seat] = big(hands, workspace, seat, WinMethod.TSUMO);
          bigRon[seat] = big(hands, workspace, seat, WinMethod.RON);
        }
        Arrays.fill(predicted, 0);
        Arrays.fill(predictedDealer, 0);
        for (int main = 0; main < hands.mainCount(); main++) {
          int winner = hands.winner(main);
          int second = hands.secondWinner(main);
          double firstBig = winner == HandOutcomes.NONE
              ? 0
              : hands.section(main) == OutcomeSection.TSUMO ? bigTsumo[winner] : bigRon[winner];
          double secondBig = second == HandOutcomes.NONE ? 0 : bigRon[second];
          add(hands, main, workspace.probability(hands.mainTerm(main)), firstBig, secondBig, seats, predicted,
              predictedDealer);
        }
        int found = next.found(h);
        if (logs.length < found) {
          logs = new double[2 * found];
        }
        double max = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < found; k++) {
          int count = OutcomeRows.outcomeTerms(
              hands, next.main(h, k), next.base(h, k), next.other(h, k), next.riichi(h, k), terms, 0);
          double log = 0;
          for (int j = 0; j < count; j++) {
            log += workspace.term(terms[j]);
          }
          logs[k] = log;
          max = Math.max(max, log);
        }
        double total = 0;
        for (int k = 0; k < found; k++) {
          total += StrictMath.exp(logs[k] - max);
        }
        Arrays.fill(realized, 0);
        Arrays.fill(realizedDealer, 0);
        for (int k = 0; k < found; k++) {
          int main = next.main(h, k);
          double share = StrictMath.exp(logs[k] - max) / total;
          double firstBig = next.base(h, k) >= BIG ? 1 : 0;
          double secondBig = next.other(h, k) >= BIG ? 1 : 0;
          add(hands, main, share, firstBig, secondBig, seats, realized, realizedDealer);
        }
        int dealer = ends.dealer[h];
        for (int pass = 0; pass < 2; pass++) {
          int cellGroup = pass == 0 ? ends.group[h] : groups;
          for (int relative = 0; relative < seats; relative++) {
            int cell = seatCell(ends, cellGroup, h * seats + Seats.absolute(dealer, relative, seats));
            seatCount[cell]++;
            for (int kind = 0; kind < seatClasses; kind++) {
              int index = cell * seatClasses + kind;
              double p = predicted[relative * seatClasses + kind];
              seatPredicted[index] += p;
              seatRealized[index] += realized[relative * seatClasses + kind];
              seatVariance[index] += p * (1 - p);
            }
          }
          int cell = dealerCell(ends, cellGroup, h);
          dealerCount[cell]++;
          for (int kind = 0; kind < dealerClasses; kind++) {
            int index = cell * dealerClasses + kind;
            double p = predictedDealer[kind];
            dealerPredicted[index] += p;
            dealerRealized[index] += realizedDealer[kind];
            dealerVariance[index] += p * (1 - p);
          }
        }
      }
    }
    List<SeatCell> seatRows = new ArrayList<>();
    for (int cellGroup = 0; cellGroup <= groups; cellGroup++) {
      for (int rank = 0; rank < seats; rank++) {
        for (int bin = 0; bin < GAP_BINS; bin++) {
          int cell = (cellGroup * seats + rank) * GAP_BINS + bin;
          long count = seatCount[cell];
          if (count == 0) {
            continue;
          }
          List<Rate> rates = new ArrayList<>(seatClasses);
          for (SeatClass kind : SEAT_CLASSES) {
            int index = cell * seatClasses + kind.ordinal();
            rates.add(rate(kind.name(), count, index, seatPredicted, seatRealized, seatVariance));
          }
          seatRows.add(new SeatCell(cellGroup == groups ? -1 : cellGroup, rank, GAP_EDGES[bin], GAP_EDGES[bin + 1],
              count, seatEnding[cell], seatSkipped[cell], rates));
        }
      }
    }
    List<DealerCell> dealerRows = new ArrayList<>();
    for (int cellGroup = 0; cellGroup <= groups; cellGroup++) {
      for (int rank = 0; rank < seats; rank++) {
        int cell = cellGroup * seats + rank;
        long count = dealerCount[cell];
        if (count == 0) {
          continue;
        }
        List<Rate> rates = new ArrayList<>(dealerClasses);
        for (DealerClass kind : DEALER_CLASSES) {
          int index = cell * dealerClasses + kind.ordinal();
          rates.add(rate(kind.name(), count, index, dealerPredicted, dealerRealized, dealerVariance));
        }
        dealerRows.add(new DealerCell(cellGroup == groups ? -1 : cellGroup, rank, count, dealerEnding[cell],
            dealerSkipped[cell], rates));
      }
    }
    return new Result(measured, ending, skipped, seatRows, dealerRows);
  }

  /**
   * 局末 1 つを数える。席から見たセルには席ごとに 1 つ、親から見たセルには局末ごとに 1 つ足す。どちらも、その局末の区分のセルと、
   * 全ての区分をまとめたセルの両方に足す。
   *
   * @param bySeat 席から見たセルごとの数（（席, 局末）の組を数える）
   * @param byDealer 親から見たセルごとの数（局末を数える）
   */
  private static void countCells(HandEnds ends, int h, long[] bySeat, long[] byDealer) {
    for (int pass = 0; pass < 2; pass++) {
      int cellGroup = pass == 0 ? ends.group[h] : ends.groups;
      for (int seat = 0; seat < ends.seats; seat++) {
        bySeat[seatCell(ends, cellGroup, h * ends.seats + seat)]++;
      }
      byDealer[dealerCell(ends, cellGroup, h)]++;
    }
  }

  /**
   * 席から見たセルの番号。
   *
   * @param cellGroup セルの区分（全ての区分をまとめたセルは区分の数）
   * @param j （席, 局末）の組の番号（局末 × 席の数 ＋ 席）
   */
  private static int seatCell(HandEnds ends, int cellGroup, int j) {
    return (cellGroup * ends.seats + ends.rank[j]) * GAP_BINS + gapBin(ends.gap[j]);
  }

  /**
   * 親から見たセルの番号。
   *
   * @param cellGroup セルの区分（全ての区分をまとめたセルは区分の数）
   */
  private static int dealerCell(HandEnds ends, int cellGroup, int h) {
    return cellGroup * ends.seats + ends.rank[h * ends.seats + ends.dealer[h]];
  }

  /**
   * 局の終わり方 1 つの重みを、席から見た分類と親から見た分類に足す。
   *
   * @param weight 終わり方の重み（予測なら q、実際なら割り振った重み）
   * @param firstBig 和了者（ダブロンなら {@code HandOutcomes.winner} の席）が満貫以上である確率
   * @param secondBig ダブロンの 2 人目が満貫以上である確率
   * @param bySeat 相対席ごとの分類に足す配列（並びは [相対席 × 分類の数 ＋ 分類]）
   * @param byDealer 親から見た分類に足す配列
   */
  private static void add(
      HandOutcomes hands,
      int main,
      double weight,
      double firstBig,
      double secondBig,
      int seats,
      double[] bySeat,
      double[] byDealer) {
    int classes = SEAT_CLASSES.length;
    int winner = hands.winner(main);
    int second = hands.secondWinner(main);
    int loser = hands.loser(main);
    switch (hands.section(main)) {
      case TSUMO -> {
        for (int seat = 0; seat < seats; seat++) {
          SeatClass kind = seat == winner ? SeatClass.TSUMO : SeatClass.OTHER_TSUMO;
          bySeat[seat * classes + kind.ordinal()] += weight;
        }
        bySeat[winner * classes + SeatClass.BIG_WIN.ordinal()] += weight * firstBig;
        byDealer[(winner == 0 ? DealerClass.DEALER_WIN : DealerClass.CHILD_TSUMO).ordinal()] += weight;
      }
      case RON, DOUBLE_RON -> {
        for (int seat = 0; seat < seats; seat++) {
          SeatClass kind = seat == winner || seat == second
              ? SeatClass.RON
              : seat == loser ? SeatClass.DEAL_IN : SeatClass.OTHER_RON;
          bySeat[seat * classes + kind.ordinal()] += weight;
        }
        bySeat[winner * classes + SeatClass.BIG_WIN.ordinal()] += weight * firstBig;
        if (second != HandOutcomes.NONE) {
          bySeat[second * classes + SeatClass.BIG_WIN.ordinal()] += weight * secondBig;
        }
        bySeat[loser * classes + SeatClass.BIG_DEAL_IN.ordinal()] += weight * (1 - (1 - firstBig) * (1 - secondBig));
        boolean dealerWins = winner == 0 || second == 0;
        byDealer[(dealerWins ? DealerClass.DEALER_WIN : DealerClass.CHILD_RON).ordinal()] += weight;
      }
      case DRAW, ABORTIVE -> {
        for (int seat = 0; seat < seats; seat++) {
          bySeat[seat * classes + SeatClass.DRAW.ordinal()] += weight;
        }
        DealerClass kind = hands.section(main) == OutcomeSection.ABORTIVE
            ? DealerClass.ABORTIVE
            : (hands.tenpai(main) & 1) != 0 ? DealerClass.DEALER_TENPAI : DealerClass.DEALER_NOTEN;
        byDealer[kind.ordinal()] += weight;
      }
    }
  }

  /**
   * その席が指定した和了の仕方で和了したとき、満貫以上になる確率。打点の確率のうち、満貫以上の分を足す。
   *
   * @param seat 和了者の相対席
   */
  private static double big(HandOutcomes hands, OutcomeWorkspace workspace, int seat, WinMethod method) {
    int first = hands.gradeTerm(seat, method, 0);
    double sum = 0;
    for (int base = BIG; base < HandOutcomes.BASES; base++) {
      sum += workspace.probability(first + base);
    }
    return sum;
  }

  /**
   * 分類 1 つの予測と実際の値を作る。
   *
   * @param count セルに入った数
   * @param variance q が正しいときの「実際 − 予測」の和の分散（セルごと・分類ごと）
   */
  private static Rate rate(
      String name, long count, int index, double[] predicted, double[] realized, double[] variance) {
    return new Rate(name, predicted[index] / count, realized[index] / count, Math.sqrt(variance[index]) / count);
  }

  private static int gapBin(int gap) {
    int bin = 0;
    while (gap >= GAP_EDGES[bin + 1]) {
      bin++;
    }
    return bin;
  }
}
