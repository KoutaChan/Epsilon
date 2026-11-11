package com.epsilon.grp;

import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

/**
 * 最後の局の数え上げ。次局が南場の最後の局から先の卓 s では、次局の結果 o を OutcomeNetwork の q(o|s) で数え上げて、
 * p(π|s) = Σ_o q(o|s)·P(π|s′(o)) を出す。o は、局の終わり方、和了ごとの打点、各席が立直したかの組（HandOutcomes）。立直したかを
 * 確率に入れるのは一部の席だけ。次の卓 s′ は、精算（Settlement）と局の進行（Progression）で作る。その局で半荘が終われば、P は
 * 終局の点で決まる順列に確率 1 を置く分布で、厳密に求まる。半荘が続けば、葉のネットワーク（GrpNetwork）の p(π|s′) を使う。
 * 数え上げるのは 1 局先までで、数え上げない卓には葉のネットワークの値をそのまま出す。局の始めの点が 1,000 点未満の席は立直できない
 * ので、q がその席の立直に置く確率は使わず、立直していないものとして数える。
 *
 * <p>終わり方、打点、立直の順に確率を掛け、積が打ち切りの値より小さくなった組はその先を足さない。打ち切った確率の和 m は、葉の
 * ネットワークの p(π|s) で埋める（p ← 足した分 ＋ m·p_ネットワーク(π|s)）。残りの組で割り直さないので、打ち切られやすい組
 * （稀な大物手で順位が入れ替わる組）の分だけネットワークの値に寄る。足す順は番号の順に決まっていて、葉のネットワークの値も卓だけで
 * 決まる。そのため、メモの有無に依らず値はビット単位で再現する。作業領域はスレッドごとに持つ。
 */
public final class FinalRound {

  /** 数え上げられるルール。局の結果のネットワークの形も、このルールでだけ作れる。精算と局の進行は epsilon-mahjong のルールに従う。 */
  public static final Set<Variant> VARIANTS =
      Collections.unmodifiableSet(EnumSet.of(Variant.TENHOU_4P, Variant.TENHOU_3P));

  private final OutcomeNetwork outcomes;
  private final GrpNetwork leaf;
  private final double cutoff;
  private final Variant variant;
  private final int seats;
  private final HandOutcomes hands;
  private final RankPermutations permutations;
  private final int count;

  private FinalRound(OutcomeNetwork outcomes, GrpNetwork leaf, double cutoff) {
    this.outcomes = outcomes;
    this.leaf = leaf;
    this.cutoff = cutoff;
    variant = leaf.variant();
    seats = variant.seats();
    hands = HandOutcomes.of(variant);
    permutations = RankPermutations.of(variant);
    count = permutations.count();
  }

  /**
   * @param outcomes 次局の結果の確率を出すネットワーク
   * @param leaf 葉の卓の p(π|s′) を出すネットワーク。数え上げない卓と、打ち切った確率を埋めるのにも使う
   * @param cutoff 打ち切りの値。確率の積がこれより小さい組は足さない。0 なら全ての組を足す
   * @throws IllegalArgumentException 2 つのネットワークのルールが違うか、打ち切りの値が [0, 1) の外のとき
   */
  public static FinalRound of(OutcomeNetwork outcomes, GrpNetwork leaf, double cutoff) {
    if (outcomes.variant() != leaf.variant()) {
      throw new IllegalArgumentException("Outcome network of " + outcomes.variant() + " with a leaf of " + leaf.variant());
    }
    if (!(cutoff >= 0 && cutoff < 1)) {
      throw new IllegalArgumentException("The cutoff must be in [0, 1): " + cutoff);
    }
    return new FinalRound(outcomes, leaf, cutoff);
  }

  public OutcomeNetwork outcomes() {
    return outcomes;
  }

  public GrpNetwork leaf() {
    return leaf;
  }

  public double cutoff() {
    return cutoff;
  }

  public Variant variant() {
    return variant;
  }

  /**
   * 数え上げる卓なら true を返す。数え上げるのは、ルールが {@link #VARIANTS} に入り、半荘が続き、次局が南場の最後の局から先の卓
   * （四人打ちは南 4 局から、三人打ちは南 3 局から）。
   */
  public static boolean covers(TableState table) {
    Variant tableVariant = table.variant();
    return VARIANTS.contains(tableVariant) && !table.gameOver() && table.round() >= tableVariant.lastRound();
  }

  public static boolean covers(GrpState state) {
    return VARIANTS.contains(state.variant()) && state.round() >= state.variant().lastRound();
  }

  /**
   * 卓の p(π|s) を書く。
   *
   * @param out 出力先（長さ S!）
   * @return 葉のネットワークで埋めた確率。数え上げない卓は 0
   */
  public double permutations(GrpState state, FinalRoundWorkspace workspace, double[] out) {
    return permutations(check(workspace).load(state), workspace, out);
  }

  /**
   * 半荘が続く卓の p(π|s) を書く。席は次局の親を 0 とする相対席で、順列の番号は RankPermutations に従う。数え上げない卓には
   * 葉のネットワークの値を書く。卓のルールが違うときと、出力先の長さが違うときは IllegalArgumentException を投げる。
   *
   * @param out 出力先（長さ S!）
   * @return 葉のネットワークで埋めた確率（足さなかった組の確率の和）。数え上げない卓は 0
   */
  public double permutations(TableState start, FinalRoundWorkspace workspace, double[] out) {
    check(workspace);
    if (start.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: final round " + variant + ", table " + start.variant());
    }
    if (out.length != count) {
      throw new IllegalArgumentException("Output needs " + count + " values, got " + out.length);
    }
    workspace.terminalMass = 0;
    if (!covers(start)) {
      leaf.permutations(start, workspace.leaf, out);
      return 0;
    }
    outcomes.forward(start, workspace.outcomes);
    Arrays.fill(out, 0);
    double dropped = enumerate(start, workspace, out);
    if (dropped > 0) {
      leaf.permutations(start, workspace.leaf, workspace.leafOut);
      for (int k = 0; k < count; k++) {
        out[k] += dropped * workspace.leafOut[k];
      }
    }
    return dropped;
  }

  /**
   * 絶対席ごとの Φ を書く。
   *
   * @param out 出力先（長さ S）
   * @return 葉のネットワークで埋めた確率。数え上げない卓は 0
   */
  public double phi(GrpState state, RankUtility utility, FinalRoundWorkspace workspace, double[] out) {
    return phi(check(workspace).load(state), utility, workspace, out);
  }

  /**
   * 半荘が続く卓の、絶対席ごとの Φ_i = Σ_π p(π|s)·u_{π(i)} を書く。足し方は {@code RankPermutations.phi} と同じ。
   *
   * @param out 出力先（長さ S）
   * @return 葉のネットワークで埋めた確率。数え上げない卓は 0
   */
  public double phi(TableState start, RankUtility utility, FinalRoundWorkspace workspace, double[] out) {
    if (out.length != seats) {
      throw new IllegalArgumentException(variant + " needs " + seats + " values, got " + out.length);
    }
    double filled = permutations(start, workspace, workspace.mixture);
    permutations.phi(workspace.mixture, start.dealer(), utility, out);
    return filled;
  }

  /**
   * 終わり方・打点・立直の組を数え上げ、打ち切らなかった組の葉の値を足す。
   *
   * @return 打ち切った組の確率の和
   */
  private double enumerate(TableState start, FinalRoundWorkspace workspace, double[] out) {
    double[] probability = workspace.outcomes.probabilities;
    double dropped = 0;
    for (int main = 0; main < hands.mainCount(); main++) {
      double p = probability[hands.mainTerm(main)];
      if (p < cutoff) {
        dropped += p;
        continue;
      }
      prepareRiichi(start, workspace, main, probability);
      switch (hands.section(main)) {
        case TSUMO, RON -> {
          WinMethod method = hands.section(main) == OutcomeSection.TSUMO ? WinMethod.TSUMO : WinMethod.RON;
          int grades = hands.gradeTerm(hands.winner(main), method, 0);
          for (int base = 0; base < HandOutcomes.BASES; base++) {
            double q = p * probability[grades + base];
            if (q < cutoff) {
              dropped += q;
              continue;
            }
            dropped += riichi(start, workspace, out, main, q, base, HandOutcomes.NONE);
          }
        }
        case DOUBLE_RON -> {
          int grades = hands.gradeTerm(hands.winner(main), WinMethod.RON, 0);
          int others = hands.gradeTerm(hands.secondWinner(main), WinMethod.RON, 0);
          for (int base = 0; base < HandOutcomes.BASES; base++) {
            double q = p * probability[grades + base];
            if (q < cutoff) {
              dropped += q;
              continue;
            }
            for (int other = 0; other < HandOutcomes.BASES; other++) {
              double r = q * probability[others + other];
              if (r < cutoff) {
                dropped += r;
                continue;
              }
              dropped += riichi(start, workspace, out, main, r, base, other);
            }
          }
        }
        case DRAW, ABORTIVE -> dropped += riichi(start, workspace, out, main, p, HandOutcomes.NONE, HandOutcomes.NONE);
      }
    }
    return dropped;
  }

  /**
   * 立直の有無を確率に入れる席ごとに、立直した確率、立直していない確率、絶対席のビットを作業領域に置く。局の始めの点が 1,000 点
   * 未満の席は立直できないので置かず、立直していないものとして数える。
   *
   * @param main 終わり方の番号
   */
  private void prepareRiichi(TableState start, FinalRoundWorkspace workspace, int main, double[] probability) {
    int dealer = start.dealer();
    int factors = 0;
    for (int index = 0; index < hands.factorCount(main); index++) {
      int seat = hands.factorSeat(main, index);
      int absolute = Seats.absolute(dealer, seat, seats);
      if (start.score(absolute) < Variant.DEPOSIT) {
        continue;
      }
      RiichiRole role = hands.role(main, seat);
      workspace.declared[factors] = probability[hands.riichiTerm(seat, role, true)];
      workspace.quiet[factors] = probability[hands.riichiTerm(seat, role, false)];
      workspace.riichiBits[factors] = 1 << absolute;
      factors++;
    }
    workspace.factors = factors;
  }

  /**
   * 立直した席の組み合わせを数え上げ、打ち切らなかった組の葉の値を足す。
   *
   * @return 打ち切った組の確率の和
   */
  private double riichi(
      TableState start, FinalRoundWorkspace workspace, double[] out, int main, double p, int base, int other) {
    int factors = workspace.factors;
    double dropped = 0;
    for (int mask = 0; mask < 1 << factors; mask++) {
      double q = p;
      int riichi = 0;
      for (int index = 0; index < factors; index++) {
        if ((mask & 1 << index) != 0) {
          q *= workspace.declared[index];
          riichi |= workspace.riichiBits[index];
        } else {
          q *= workspace.quiet[index];
        }
      }
      if (q < cutoff) {
        dropped += q;
        continue;
      }
      settle(start, workspace, main, base, other, riichi);
      add(start, workspace, out, q);
    }
    return dropped;
  }

  /** 組が表す局の結果を RoundOutcome に置いて精算し、局を進めた卓を作業領域に作る。 */
  private void settle(TableState start, FinalRoundWorkspace workspace, int main, int base, int other, int riichi) {
    RoundOutcome outcome = workspace.outcome;
    hands.place(main, base, other, riichi, start.dealer(), outcome);
    Settlement.apply(variant, start, outcome, workspace.delta);
    Progression.next(start, outcome, workspace.delta, workspace.after);
    workspace.settled++;
  }

  /**
   * 局を進めた卓の葉の値に確率 q を掛けて足す。順列は元の卓の相対席に直して足す。半荘が終わった卓では、終局の点で決まる順列に
   * q を足す。
   */
  private void add(TableState start, FinalRoundWorkspace workspace, double[] out, double q) {
    TableState after = workspace.after;
    if (after.gameOver()) {
      int[] scores = workspace.scores;
      for (int seat = 0; seat < seats; seat++) {
        scores[seat] = after.score(seat);
      }
      out[permutations.realized(scores, start.dealer())] += q;
      workspace.terminalMass += q;
      workspace.terminalLeaves++;
      return;
    }
    int at = leafValues(after, workspace);
    double[] values = at < 0 ? workspace.leafOut : workspace.memoValues;
    int from = Math.max(at, 0);
    int shift = Math.floorMod(after.dealer() - start.dealer(), seats);
    for (int k = 0; k < count; k++) {
      out[permutations.rotate(k, shift)] += q * values[from + k];
    }
  }

  /**
   * 半荘が続く卓について、葉のネットワークの p(π|s′) を用意する。
   *
   * @return メモの値の配列（{@code memoValues}）の中で、値が始まる位置。キーに収まらない卓では −1 で、値は {@code leafOut} にある
   */
  private int leafValues(TableState after, FinalRoundWorkspace workspace) {
    long key = GrpKey.of(after);
    if (key == GrpKey.NONE) {
      leaf.permutations(after, workspace.leaf, workspace.leafOut);
      workspace.networkLeaves++;
      return -1;
    }
    int slot = (int) SeedMixer.mix64(key) & workspace.memoMask;
    int at = slot * count;
    if (workspace.memoKeys[slot] == key) {
      workspace.memoHits++;
      return at;
    }
    leaf.permutations(after, workspace.leaf, workspace.leafOut);
    System.arraycopy(workspace.leafOut, 0, workspace.memoValues, at, count);
    workspace.memoKeys[slot] = key;
    workspace.networkLeaves++;
    return at;
  }

  private FinalRoundWorkspace check(FinalRoundWorkspace workspace) {
    if (workspace.finalRound != this) {
      throw new IllegalArgumentException("A workspace of another final round");
    }
    return workspace;
  }

  @Override
  public String toString() {
    return "FinalRound[" + outcomes + " leaf=" + leaf + " cutoff=" + cutoff + "]";
  }
}
