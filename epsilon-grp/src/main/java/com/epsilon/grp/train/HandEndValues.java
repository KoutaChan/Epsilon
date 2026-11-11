package com.epsilon.grp.train;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.FinalRoundWorkspace;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.grp.Phi;
import com.epsilon.grp.RankPermutations;
import com.epsilon.grp.RoundHistory;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * Φ の項目 1 つで、HandEnds の局末ごとに求めた値。局末ごとに、絶対席の Φ（ラベルに入る値）、席ごとの最終順位の周辺確率、実際の
 * 順列の対数確率を持つ。対数確率は、同時分布を出さない項目では NaN。
 *
 * <p>最後の局を数え上げた値は、局末ごとに次の 4 つも持つ。打ち切って葉のネットワークで埋めた確率、局の後に半荘が終わる組の確率の和、
 * 精算した組の数、半荘が続く葉の数。数え上げない局末では、どれも 0。
 */
public final class HandEndValues {

  final HandEnds ends;

  /** 絶対席の Φ。配列の位置は h × S ＋ 席。 */
  final double[] phi;

  /** 周辺確率。配列の位置は (h × S ＋ 席) × S ＋ 順位。 */
  final double[] marginals;

  /** 実際の順列の対数確率。配列の位置は h。 */
  final double[] jointLog;

  /** 打ち切って葉のネットワークで埋めた確率。配列の位置は h。数え上げた値でなければ null。 */
  final double[] filledMass;

  /** 局の後に半荘が終わる組の確率の和。配列の位置は h。数え上げた値でなければ null。 */
  final double[] terminalMass;

  /** 精算して局を進めた組の数。配列の位置は h。数え上げた値でなければ null。 */
  final int[] settled;

  /** 半荘が続く葉の数（葉のネットワークの値を読んだ数で、メモから読んだ分も入る）。配列の位置は h。数え上げた値でなければ null。 */
  final int[] leaves;

  private HandEndValues(HandEnds ends, double[] phi, double[] marginals, double[] jointLog) {
    this(ends, phi, marginals, jointLog, null, null, null, null);
  }

  private HandEndValues(
      HandEnds ends, double[] phi, double[] marginals, double[] jointLog, double[] filledMass, double[] terminalMass,
      int[] settled, int[] leaves) {
    this.ends = ends;
    this.phi = phi;
    this.marginals = marginals;
    this.jointLog = jointLog;
    this.filledMass = filledMass;
    this.terminalMass = terminalMass;
    this.settled = settled;
    this.leaves = leaves;
  }

  /**
   * GrpNetwork の出力から値を作る。Φ と周辺確率はネットワークと同じ式（RankPermutations）で求めるので、Φ は Phi が返す値と
   * ビット単位で同じになる。実際の順列の確率が double で 0 にアンダーフローしたときは、対数確率をロジットから求め直す。
   * ネットワークのルールが局末と違えば IllegalArgumentException を投げる。
   */
  public static HandEndValues of(HandEnds ends, GrpNetwork network, GrpWorkspace workspace) {
    if (network.variant() != ends.variant) {
      throw new IllegalArgumentException("A " + network.variant() + " network for " + ends.variant + " hand ends");
    }
    int seats = ends.seats;
    RankPermutations table = RankPermutations.of(ends.variant);
    double[] probabilities = new double[table.count()];
    double[] seatPhi = new double[seats];
    double[] seatMarginals = new double[seats * seats];
    double[] phi = new double[ends.handEnds() * seats];
    double[] marginals = new double[ends.handEnds() * seats * seats];
    double[] jointLog = new double[ends.handEnds()];
    for (int h = 0; h < ends.handEnds(); h++) {
      jointLog[h] = network(ends, h, network, workspace, probabilities);
      table.phi(probabilities, ends.dealer[h], ends.utility, seatPhi);
      table.marginals(probabilities, ends.dealer[h], seatMarginals);
      System.arraycopy(seatPhi, 0, phi, h * seats, seats);
      System.arraycopy(seatMarginals, 0, marginals, h * seats * seats, seats * seats);
    }
    return new HandEndValues(ends, phi, marginals, jointLog);
  }

  /**
   * 局末 1 つについて、順列ごとの確率をネットワークで求めて書き込み、実際の順列の対数確率を返す。確率が double で 0 に
   * アンダーフローしたときは、対数確率をロジットから求め直す。
   */
  private static double network(
      HandEnds ends, int h, GrpNetwork network, GrpWorkspace workspace, double[] probabilities) {
    GrpState state = ends.states[h];
    network.permutations(state, workspace, probabilities);
    int realized = ends.permutation[h];
    return probabilities[realized] > 0
        ? StrictMath.log(probabilities[realized])
        : network.logProbability(state, realized, workspace);
  }

  /**
   * Φ の項目の値を作る。収集で局末の値を求めるときと同じ計算を通るので、Φ は Phi が返す値とビット単位で同じになる。項目のルールか
   * 効用が局末と違えば IllegalArgumentException を投げる。
   *
   * <p>ネットワークの項目は {@link #of(HandEnds, GrpNetwork, GrpWorkspace)} と同じ値になる。ネットワークと終盤の数え上げ以外の項目は、
   * 実際の順列の対数確率を持たない（NaN）。GruSinkhorn には、局末ごとに半荘の始めからの局の履歴を積んで渡す。
   *
   * <p>終盤の数え上げは、数え上げる局末を {@link #of(HandEnds, FinalRound, Supplier)} と同じ計算で求め、ほかの局末を葉のネットワーク
   * で {@link #of(HandEnds, GrpNetwork, GrpWorkspace)} と同じ計算で求める。どちらの局末も、周辺確率と実際の順列の対数確率まで、
   * それぞれの計算とビット単位で同じになる。
   */
  public static HandEndValues of(HandEnds ends, Phi phi) {
    if (phi.variant() != ends.variant) {
      throw new IllegalArgumentException("A " + phi.variant() + " Φ for " + ends.variant + " hand ends");
    }
    if (phi.utility() != ends.utility) {
      throw new IllegalArgumentException("Φ of " + phi.utility() + " for hand ends of " + ends.utility);
    }
    if (phi.kind() == Phi.Kind.LATE) {
      return enumerated(ends, phi.finalRound(), phi::finalRoundWorkspace, phi::networkWorkspace);
    }
    if (phi.model() instanceof GrpNetwork network) {
      return of(ends, network, phi.networkWorkspace());
    }
    int seats = ends.seats;
    Builder builder = builder(ends);
    double[] values = new double[seats];
    double[] ranks = new double[seats * seats];
    RoundHistory history = phi.needsHistory() ? new RoundHistory(ends.variant) : null;
    for (int h = 0; h < ends.handEnds(); h++) {
      if (history != null) {
        ends.history(h, history);
      }
      phi.evaluate(history, ends.states[h], values, ranks);
      builder.set(h, values, ranks);
    }
    return builder.build();
  }

  /**
   * 最後の局の数え上げ（FinalRound）が出す p(π|s) から値を作る。Φ・周辺確率・実際の順列の対数確率は、GrpNetwork の値と同じ式で
   * 求める。実際の順列の確率が 0 なら、対数確率は −∞。局末ごとに、打ち切って埋めた確率、半荘が終わる組の確率の和、精算した組の数、
   * 半荘が続く葉の数も持つ。局末ごとに独立に並列で求めるので、値も数もスレッドの数や求める順に依らない。
   *
   * @param workspaces 数え上げの作業領域を作る関数。値を求めるスレッドごとに 1 回呼ぶ（呼び出し元のスレッドでも呼ぶ）
   */
  public static HandEndValues of(HandEnds ends, FinalRound round, Supplier<FinalRoundWorkspace> workspaces) {
    return enumerated(ends, round, workspaces, null);
  }

  /**
   * 数え上げた値を作る。数え上げる局末は FinalRound で求める。ほかの局末は、葉のネットワークの作業領域を作る関数があれば
   * {@link #of(HandEnds, GrpNetwork, GrpWorkspace)} と同じ計算で求め、無ければ FinalRound に任せる（FinalRound は葉のネットワークの
   * 値をそのまま書く）。
   *
   * @param workspaces 数え上げの作業領域を作る関数。値を求めるスレッドごとに 1 回呼ぶ
   * @param networks 葉のネットワークの作業領域を作る関数。値を求めるスレッドごとに 1 回呼ぶ。null なら全ての局末を FinalRound で求める
   */
  private static HandEndValues enumerated(
      HandEnds ends, FinalRound round, Supplier<FinalRoundWorkspace> workspaces, Supplier<GrpWorkspace> networks) {
    if (round.variant() != ends.variant) {
      throw new IllegalArgumentException("A " + round.variant() + " final round for " + ends.variant + " hand ends");
    }
    int seats = ends.seats;
    int square = seats * seats;
    RankPermutations table = RankPermutations.of(ends.variant);
    GrpNetwork leaf = round.leaf();
    double[] phi = new double[ends.handEnds() * seats];
    double[] marginals = new double[ends.handEnds() * square];
    double[] jointLog = new double[ends.handEnds()];
    double[] filled = new double[ends.handEnds()];
    double[] terminal = new double[ends.handEnds()];
    int[] settled = new int[ends.handEnds()];
    int[] leaves = new int[ends.handEnds()];
    ThreadLocal<Local> locals = ThreadLocal.withInitial(() -> new Local(
        workspaces.get(), networks == null ? null : networks.get(), new double[table.count()], new double[seats],
        new double[square]));
    // 並列に回す前に、呼び出し元のスレッドで作業領域を 1 つ作る。作れない組み合わせは、ここで例外になる。
    locals.get();
    IntStream.range(0, ends.handEnds()).parallel().forEach(h -> {
      Local local = locals.get();
      double[] probabilities = local.probabilities();
      if (local.network() != null && !FinalRound.covers(ends.states[h])) {
        jointLog[h] = network(ends, h, leaf, local.network(), probabilities);
      } else {
        FinalRoundWorkspace workspace = local.workspace();
        long settledBefore = workspace.settled();
        long leavesBefore = workspace.networkLeaves() + workspace.memoHits();
        filled[h] = round.permutations(ends.states[h], workspace, probabilities);
        terminal[h] = workspace.terminalMass();
        settled[h] = (int) (workspace.settled() - settledBefore);
        leaves[h] = (int) (workspace.networkLeaves() + workspace.memoHits() - leavesBefore);
        jointLog[h] = StrictMath.log(probabilities[ends.permutation[h]]);
      }
      table.phi(probabilities, ends.dealer[h], ends.utility, local.phi());
      table.marginals(probabilities, ends.dealer[h], local.marginals());
      System.arraycopy(local.phi(), 0, phi, h * seats, seats);
      System.arraycopy(local.marginals(), 0, marginals, h * square, square);
    });
    return new HandEndValues(ends, phi, marginals, jointLog, filled, terminal, settled, leaves);
  }

  /**
   * 値を求めるスレッドごとの作業領域と、局末 1 つ分の出力先。
   *
   * @param network 葉のネットワークの作業領域。数え上げない局末を葉のネットワークで求めるときだけ持ち、ほかは null
   */
  private record Local(
      FinalRoundWorkspace workspace, GrpWorkspace network, double[] probabilities, double[] phi, double[] marginals) {}

  /**
   * 局末を選んだ列（{@code HandEnds.select}）の値を作る。選んだ局末の分だけ、この値からコピーする。
   *
   * @param selection この値の局末の列から選んだ列
   * @throws IllegalArgumentException ほかの列から選んだ列のとき
   */
  public HandEndValues select(HandEnds selection) {
    if (selection.origin != ends) {
      throw new IllegalArgumentException("The selection was not made from these hand ends");
    }
    int seats = ends.seats;
    int square = seats * seats;
    int count = selection.handEnds();
    double[] selectedPhi = new double[count * seats];
    double[] selectedMarginals = new double[count * square];
    double[] selectedJointLog = new double[count];
    double[] selectedFilled = filledMass == null ? null : new double[count];
    double[] selectedTerminal = terminalMass == null ? null : new double[count];
    int[] selectedSettled = settled == null ? null : new int[count];
    int[] selectedLeaves = leaves == null ? null : new int[count];
    for (int index = 0; index < count; index++) {
      int h = selection.sources[index];
      System.arraycopy(phi, h * seats, selectedPhi, index * seats, seats);
      System.arraycopy(marginals, h * square, selectedMarginals, index * square, square);
      selectedJointLog[index] = jointLog[h];
      if (selectedFilled != null) {
        selectedFilled[index] = filledMass[h];
        selectedTerminal[index] = terminalMass[h];
        selectedSettled[index] = settled[h];
        selectedLeaves[index] = leaves[h];
      }
    }
    return new HandEndValues(
        selection, selectedPhi, selectedMarginals, selectedJointLog, selectedFilled, selectedTerminal,
        selectedSettled, selectedLeaves);
  }

  /** GrpNetwork 以外の項目（点数の Φ や GRU）の値を、局末ごとに入れて作る。 */
  public static Builder builder(HandEnds ends) {
    return new Builder(ends);
  }

  public HandEnds ends() {
    return ends;
  }

  public double phi(int handEnd, int seat) {
    return phi[handEnd * ends.seats + seat];
  }

  public double marginal(int handEnd, int seat, int place) {
    return marginals[(handEnd * ends.seats + seat) * ends.seats + place];
  }

  /** 実際の順列の対数確率。同時分布を出さない項目では NaN。 */
  public double jointLog(int handEnd) {
    return jointLog[handEnd];
  }

  /** 最後の局を数え上げた値（埋めた確率や、半荘が終わる組の確率の和を持つ）なら true。 */
  public boolean enumerated() {
    return filledMass != null;
  }

  /** 打ち切って葉のネットワークで埋めた確率。数え上げない局末と、数え上げた値でないときは 0。 */
  public double filledMass(int handEnd) {
    return filledMass == null ? 0 : filledMass[handEnd];
  }

  /** 局の後に半荘が終わる組の確率の和（値が厳密に決まる部分）。数え上げない局末と、数え上げた値でないときは 0。 */
  public double terminalMass(int handEnd) {
    return terminalMass == null ? 0 : terminalMass[handEnd];
  }

  /** 精算して局を進めた組の数。数え上げない局末と、数え上げた値でないときは 0。 */
  public int settled(int handEnd) {
    return settled == null ? 0 : settled[handEnd];
  }

  /**
   * 半荘が続く葉の数（葉のネットワークの値を読んだ数で、メモから読んだ分も入る）。精算した組の数からこの数を引くと、半荘が終わる葉の
   * 数になる。数え上げない局末と、数え上げた値でないときは 0。
   */
  public int leaves(int handEnd) {
    return leaves == null ? 0 : leaves[handEnd];
  }

  /** 全ての局末に、実際の順列の対数確率があるか。 */
  public boolean joint() {
    for (double value : jointLog) {
      if (Double.isNaN(value)) {
        return false;
      }
    }
    return true;
  }

  /** Φ と周辺確率が全て有限で、対数確率が全て有限か全て NaN か。 */
  public boolean finite() {
    for (double value : phi) {
      if (!Double.isFinite(value)) {
        return false;
      }
    }
    for (double value : marginals) {
      if (!Double.isFinite(value)) {
        return false;
      }
    }
    boolean joint = joint();
    for (double value : jointLog) {
      if (joint ? !Double.isFinite(value) : !Double.isNaN(value)) {
        return false;
      }
    }
    return true;
  }

  /**
   * 局末ごとに値を入れる。全ての局末に入れてから {@code build} を 1 回だけ呼ぶ。{@code build} は配列をコピーせずに渡す。8 コホート分
   * では項目 1 つの値が約 0.9 GB になり、コピーするとヒープ 4 GB に 2 つの項目を同時に持てないため。
   */
  public static final class Builder {

    private final HandEnds ends;
    private final double[] phi;
    private final double[] marginals;
    private final double[] jointLog;
    private final boolean[] filled;
    private int count;
    private boolean built;

    private Builder(HandEnds ends) {
      this.ends = Objects.requireNonNull(ends, "ends");
      phi = new double[ends.handEnds() * ends.seats];
      marginals = new double[ends.handEnds() * ends.seats * ends.seats];
      jointLog = new double[ends.handEnds()];
      Arrays.fill(jointLog, Double.NaN);
      filled = new boolean[ends.handEnds()];
    }

    /**
     * 同時分布を出さない項目の値を、局末 1 つ分入れる。
     *
     * @param values 絶対席の Φ
     * @param ranks 周辺確率。配列の位置は 席 × S ＋ 順位
     */
    public Builder set(int handEnd, double[] values, double[] ranks) {
      return set(handEnd, values, ranks, Double.NaN);
    }

    /**
     * 局末 1 つの値を入れる。
     *
     * @param values 絶対席の Φ
     * @param ranks 周辺確率。配列の位置は 席 × S ＋ 順位
     * @param jointLog 実際の順列の対数確率
     * @throws IllegalStateException {@code build} の後に呼んだとき
     */
    public Builder set(int handEnd, double[] values, double[] ranks, double jointLog) {
      if (built) {
        throw new IllegalStateException("The values were already built");
      }
      int seats = ends.seats;
      if (values.length != seats || ranks.length != seats * seats) {
        throw new IllegalArgumentException(
            "Expected " + seats + " values and " + seats * seats + " marginals, got " + values.length + " and "
                + ranks.length);
      }
      Objects.checkIndex(handEnd, filled.length);
      if (!filled[handEnd]) {
        filled[handEnd] = true;
        count++;
      }
      System.arraycopy(values, 0, phi, handEnd * seats, seats);
      System.arraycopy(ranks, 0, marginals, handEnd * seats * seats, seats * seats);
      this.jointLog[handEnd] = jointLog;
      return this;
    }

    /** 入れた値から作る。値を入れていない局末があるとき、2 回目の呼び出しのときは IllegalStateException を投げる。 */
    public HandEndValues build() {
      if (built) {
        throw new IllegalStateException("The values were already built");
      }
      if (count != filled.length) {
        throw new IllegalStateException((filled.length - count) + " hand ends have no values");
      }
      built = true;
      return new HandEndValues(ends, phi, marginals, jointLog);
    }
  }
}
