package com.epsilon.grp;

import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.Arrays;

/**
 * 最後の局の数え上げの、参照用の実装。葉の値は外から渡した関数で求める。ベンチマークで、葉の値を定数に替えて測るときと、葉の卓を
 * 記録するときに使う。組を数え上げる順に、その場で葉の値を求めて足す。渡した関数が葉のネットワークの値を書くなら、結果は FinalRound と
 * ビット単位で同じになる。葉の値は、GrpKey で引く 1,024 スロットのメモに覚える。同時に使えるのは 1 スレッドだけ。
 */
public final class FinalRoundReference {

  /** 葉の卓の p(π|s′) を書く関数。 */
  @FunctionalInterface
  public interface Leaf {

    /**
     * @param table 半荘が続く卓（次の呼び出しまで有効）
     * @param out 出力先（長さ S!）
     */
    void permutations(TableState table, double[] out);
  }

  private static final int MEMO_SLOTS = 1 << 10;

  private final FinalRound round;
  private final Leaf leaf;
  private final double cutoff;
  private final Variant variant;
  private final int seats;
  private final HandOutcomes hands;
  private final RankPermutations permutations;
  private final int count;
  private final OutcomeWorkspace outcomes;
  private final TableState start;
  private final TableState after;
  private final RoundOutcome outcome;
  private final int[] delta;
  private final int[] scores;
  private final double[] leafOut;
  private final long[] memoKeys;
  private final double[] memoValues;
  private final double[] declared;
  private final double[] quiet;
  private final int[] riichiBits;
  private int factors;
  private long settled;
  private long terminalLeaves;
  private long networkLeaves;
  private long memoHits;

  private FinalRoundReference(FinalRound round, Leaf leaf) {
    this.round = round;
    this.leaf = leaf;
    cutoff = round.cutoff();
    variant = round.variant();
    seats = variant.seats();
    hands = HandOutcomes.of(variant);
    permutations = RankPermutations.of(variant);
    count = permutations.count();
    outcomes = OutcomeWorkspace.of(round.outcomes().shape());
    start = new TableState(variant);
    after = new TableState(variant);
    outcome = new RoundOutcome(variant);
    delta = new int[seats];
    scores = new int[seats];
    leafOut = new double[count];
    memoKeys = new long[MEMO_SLOTS];
    Arrays.fill(memoKeys, GrpKey.NONE);
    memoValues = new double[MEMO_SLOTS * count];
    declared = new double[seats];
    quiet = new double[seats];
    riichiBits = new int[seats];
  }

  /** 葉の値を、渡した関数で求める実装を作る。 */
  public static FinalRoundReference of(FinalRound round, Leaf leaf) {
    return new FinalRoundReference(round, leaf);
  }

  /** 葉の値を、数え上げが持つ葉のネットワークで 1 卓ずつ求める実装を作る。 */
  public static FinalRoundReference of(FinalRound round) {
    GrpWorkspace workspace = GrpWorkspace.of(round.leaf().shape());
    return new FinalRoundReference(round, (table, out) -> round.leaf().permutations(table, workspace, out));
  }

  public long settled() {
    return settled;
  }

  public long terminalLeaves() {
    return terminalLeaves;
  }

  /** 数え上げの中で葉の関数を呼んだ回数。打ち切った確率を埋める呼び出しと、数え上げない卓での呼び出しは数えない。 */
  public long networkLeaves() {
    return networkLeaves;
  }

  public long memoHits() {
    return memoHits;
  }

  /**
   * 卓の p(π|s) を書く。
   *
   * @param out 出力先（長さ S!）
   * @return 打ち切った確率の和（その分は葉の値で埋める）。数え上げない卓では 0
   */
  public double permutations(GrpState state, double[] out) {
    state.copyInto(start);
    if (!FinalRound.covers(start)) {
      leaf.permutations(start, out);
      return 0;
    }
    round.outcomes().forward(start, outcomes);
    Arrays.fill(out, 0);
    double dropped = enumerate(out);
    if (dropped > 0) {
      leaf.permutations(start, leafOut);
      for (int k = 0; k < count; k++) {
        out[k] += dropped * leafOut[k];
      }
    }
    return dropped;
  }

  private double enumerate(double[] out) {
    double dropped = 0;
    for (int main = 0; main < hands.mainCount(); main++) {
      double p = outcomes.probability(hands.mainTerm(main));
      if (p < cutoff) {
        dropped += p;
        continue;
      }
      prepareRiichi(main);
      switch (hands.section(main)) {
        case TSUMO, RON -> {
          WinMethod method = hands.section(main) == OutcomeSection.TSUMO ? WinMethod.TSUMO : WinMethod.RON;
          int grades = hands.gradeTerm(hands.winner(main), method, 0);
          for (int base = 0; base < HandOutcomes.BASES; base++) {
            double q = p * outcomes.probability(grades + base);
            if (q < cutoff) {
              dropped += q;
              continue;
            }
            dropped += riichi(out, main, q, base, HandOutcomes.NONE);
          }
        }
        case DOUBLE_RON -> {
          int grades = hands.gradeTerm(hands.winner(main), WinMethod.RON, 0);
          int others = hands.gradeTerm(hands.secondWinner(main), WinMethod.RON, 0);
          for (int base = 0; base < HandOutcomes.BASES; base++) {
            double q = p * outcomes.probability(grades + base);
            if (q < cutoff) {
              dropped += q;
              continue;
            }
            for (int other = 0; other < HandOutcomes.BASES; other++) {
              double r = q * outcomes.probability(others + other);
              if (r < cutoff) {
                dropped += r;
                continue;
              }
              dropped += riichi(out, main, r, base, other);
            }
          }
        }
        case DRAW, ABORTIVE -> dropped += riichi(out, main, p, HandOutcomes.NONE, HandOutcomes.NONE);
      }
    }
    return dropped;
  }

  private void prepareRiichi(int main) {
    int dealer = start.dealer();
    factors = 0;
    for (int index = 0; index < hands.factorCount(main); index++) {
      int seat = hands.factorSeat(main, index);
      int absolute = Seats.absolute(dealer, seat, seats);
      if (start.score(absolute) < Variant.DEPOSIT) {
        continue;
      }
      RiichiRole role = hands.role(main, seat);
      declared[factors] = outcomes.probability(hands.riichiTerm(seat, role, true));
      quiet[factors] = outcomes.probability(hands.riichiTerm(seat, role, false));
      riichiBits[factors] = 1 << absolute;
      factors++;
    }
  }

  private double riichi(double[] out, int main, double p, int base, int other) {
    double dropped = 0;
    for (int mask = 0; mask < 1 << factors; mask++) {
      double q = p;
      int riichi = 0;
      for (int index = 0; index < factors; index++) {
        if ((mask & 1 << index) != 0) {
          q *= declared[index];
          riichi |= riichiBits[index];
        } else {
          q *= quiet[index];
        }
      }
      if (q < cutoff) {
        dropped += q;
        continue;
      }
      hands.place(main, base, other, riichi, start.dealer(), outcome);
      Settlement.apply(variant, start, outcome, delta);
      Progression.next(start, outcome, delta, after);
      settled++;
      add(out, q);
    }
    return dropped;
  }

  private void add(double[] out, double q) {
    if (after.gameOver()) {
      for (int seat = 0; seat < seats; seat++) {
        scores[seat] = after.score(seat);
      }
      out[permutations.realized(scores, start.dealer())] += q;
      terminalLeaves++;
      return;
    }
    int at = leafValues();
    double[] values = at < 0 ? leafOut : memoValues;
    int from = Math.max(at, 0);
    int shift = Math.floorMod(after.dealer() - start.dealer(), seats);
    for (int k = 0; k < count; k++) {
      out[permutations.rotate(k, shift)] += q * values[from + k];
    }
  }

  /**
   * 局を進めた卓の葉の値を用意する。
   *
   * @return メモの値の配列の中で、その卓の値が始まる位置。キーに収まらない卓では −1 で、値は {@code leafOut} にある
   */
  private int leafValues() {
    long key = GrpKey.of(after);
    if (key == GrpKey.NONE) {
      leaf.permutations(after, leafOut);
      networkLeaves++;
      return -1;
    }
    int slot = (int) SeedMixer.mix64(key) & MEMO_SLOTS - 1;
    int at = slot * count;
    if (memoKeys[slot] == key) {
      memoHits++;
      return at;
    }
    leaf.permutations(after, leafOut);
    System.arraycopy(leafOut, 0, memoValues, at, count);
    memoKeys[slot] = key;
    networkLeaves++;
    return at;
  }
}
