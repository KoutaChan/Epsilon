package com.epsilon.prism.luck;

import com.epsilon.grp.Phi;
import com.epsilon.grp.RoundHistory;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.ChanceKind;
import com.epsilon.mahjong.engine.ChanceLog;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.HypotheticalEnd;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.noise.UraSettings;
import java.util.Arrays;
import java.util.Objects;

/**
 * 局末の席ごとの値を求める。実際の値は、精算した後の卓の Φ（終局なら Φ を呼ばず、最終順位の効用そのもの）。期待値は、立直の和了で終わった
 * 局では裏ドラの数で平均した値 Σ_k P(k)·V(k) で、ほかの局では実際の値と同じ。
 *
 * <p>V(k) は、立直した和了者の基本点を裏ドラ k 枚で計算し直し、局の始まりの卓から精算した次局の卓の Φ。計算し直すときは実際の採点の翻と
 * 符を使い、翻だけを裏ドラの分ずらす。役満はそのまま。基本点が同じになる k はまとめ、精算と Φ の計算を 1 回で済ませる。ダブロンで 2 人とも
 * 立直していれば、同じ裏ドラ表示牌で数える 2 人の同時分布で平均する。
 *
 * <p>裏ドラ表示牌の枚数は、和了の時点でめくってあったドラ表示牌の枚数と同じ。プールは、局末に誰にも見えていない牌に、めくった裏ドラ表示牌を
 * 戻したもの。まだめくっていない保留の槓ドラの表示牌はプールに残る（{@link UraDistribution}）。
 *
 * <p>1 つのスレッドだけで使う作業領域で、値は次の呼び出しまで有効。メモリを確保しない。
 */
public final class UraExpectation {

  private static final int SIDE = UraDistribution.MAX_URA + 1;

  private final Phi phi;
  private final RankUtility utility;
  private final Variant variant;
  private final int seats;
  private final LedgerTables tables;

  private final UraDistribution distribution = new UraDistribution();
  private final HypotheticalEnd hypothetical;
  private final HandShape hand = new HandShape();
  private final int[] pool = new int[Tiles.KINDS];
  private final int[] owned = new int[Tiles.KINDS];

  /** [立直した和了者の何人目か][表示牌の種類] → その和了者が数える裏ドラの数。 */
  private final int[][] counts = new int[2][Tiles.KINDS];

  /** [立直した和了者の何人目か][裏ドラの数] → 基本点。 */
  private final int[][] rebased = new int[2][SIDE];

  /** 立直していて役満でない和了者が、局の結果の何番目の和了者か。 */
  private final int[] varying = new int[2];

  private final int[] bases = new int[2];
  private final int[] groupFirst = new int[SIDE * SIDE];
  private final int[] groupSecond = new int[SIDE * SIDE];
  private final double[] groupProbability = new double[SIDE * SIDE];

  private final double[] actual;
  private final double[] expected;
  private final double[] variance;
  private final double[] scratch;

  private int varyingCount;
  private boolean riichiWin;
  private boolean averaged;
  private boolean kan;
  private int groups;
  private long nanos;

  /** @param phi 局末の卓の値を求める Φ。この作業領域は Φ のルールを使う */
  public UraExpectation(Phi phi) {
    this.phi = Objects.requireNonNull(phi, "phi");
    utility = phi.utility();
    variant = phi.variant();
    seats = variant.seats();
    tables = LedgerTables.of(variant);
    hypothetical = new HypotheticalEnd(variant);
    actual = new double[seats];
    expected = new double[seats];
    variance = new double[seats];
    scratch = new double[seats];
  }

  public Variant variant() {
    return variant;
  }

  /**
   * 局末の値を求める。実際の値はいつも求める。期待値は、立直の和了で終わった局で、求めるよう指定したときだけ求める。
   *
   * @param full 局末の局面
   * @param end 局の終わり方と結果
   * @param start 局の始まりの卓（{@code Engine.handStart()}）
   * @param history 終わった局までの各局の始まりの卓。Φ が履歴を読まなければ null でよい
   * @param average 期待値を求めるなら true
   * @return 期待値を求めたら true
   * @throws IllegalStateException 局末の偶然のイベントの記録の末尾に、めくってあるドラ表示牌と同じ枚数の裏ドラ表示牌が無いとき、
   *     またはめくった裏ドラ表示牌で数えた裏ドラの数が和了の採点と食い違うとき
   */
  public boolean evaluate(FullView full, RoundEnd end, TableState start, RoundHistory history, boolean average) {
    long started = System.nanoTime();
    tableValues(history, end.next(), actual);
    System.arraycopy(actual, 0, expected, 0, seats);
    Arrays.fill(variance, 0);
    averaged = false;
    groups = 0;
    RoundOutcome outcome = end.outcome();
    riichiWin = scanWinners(full, end, outcome);
    kan = riichiWin && kan(full);
    if (riichiWin && average) {
      averaged = true;
      groups = 1;
      if (varyingCount > 0) average(full, end, outcome, start, history);
    }
    nanos = System.nanoTime() - started;
    return averaged;
  }

  /** 実際の値。 */
  public double actual(int seat) {
    return actual[seat];
  }

  /** 期待値。期待値を求めなかった局末では実際の値と同じ。 */
  public double expected(int seat) {
    return expected[seat];
  }

  /** 裏ドラの数で決まる値の分散 Σ_k P(k)·V(k)² − 期待値²。期待値を求めなかった局末では 0。 */
  public double variance(int seat) {
    return variance[seat];
  }

  /**
   * 指定した作り方で局末の値を書く。
   *
   * @param ura 局末の値の作り方。期待値で学ぶなら期待値を、実際の値で学ぶなら実際の値を書く
   * @param out 書き込み先（長さは席の数）
   * @throws IllegalStateException 期待値で学ぶ指定なのに、立直の和了の局末で期待値を求めていなかったとき
   */
  public void values(UraSettings ura, double[] out) {
    boolean expectation = ura instanceof UraSettings.Expected;
    if (expectation && riichiWin && !averaged) {
      throw new IllegalStateException("The expected ura value was not computed for this riichi win");
    }
    System.arraycopy(expectation ? expected : actual, 0, out, 0, seats);
  }

  /** 立直した和了者がいる局末なら true。 */
  public boolean riichiWin() {
    return riichiWin;
  }

  /** 期待値を求めたなら true。 */
  public boolean averaged() {
    return averaged;
  }

  /** 立直の和了で終わった局に、槓（槍槓で奪われた加槓を含む）があったなら true。 */
  public boolean kan() {
    return kan;
  }

  /** 期待値を求めたときの基本点の組の数（立直した和了者が全員役満なら 1）。求めなかったときは 0。 */
  public int groups() {
    return groups;
  }

  /** 直前の {@link #evaluate} にかかった時間（ナノ秒）。 */
  public long nanos() {
    return nanos;
  }

  /** 立直した和了者がいれば true を返す。そのうち役満でない和了者を、基本点を計算し直す側に入れる。 */
  private boolean scanWinners(FullView full, RoundEnd end, RoundOutcome outcome) {
    varyingCount = 0;
    if (!outcome.kind().win()) return false;
    boolean riichi = false;
    for (int i = 0; i < outcome.winnerCount(); i++) {
      int winner = outcome.winner(i);
      if (full.seat(winner).riichi(winner) == RiichiState.NONE) continue;
      riichi = true;
      if (end.score(i).yakuman() == 0) varying[varyingCount++] = i;
    }
    return riichi;
  }

  private void average(FullView full, RoundEnd end, RoundOutcome outcome, TableState start, RoundHistory history) {
    int draws = loadPool(full);
    for (int v = 0; v < varyingCount; v++) {
      loadCounts(full, outcome.winner(varying[v]), counts[v]);
      requireRealized(full.chance(), draws, counts[v], end.score(varying[v]));
    }
    if (varyingCount == 1) {
      distribution.single(pool, counts[0], draws);
    } else {
      distribution.joint(pool, counts[0], counts[1], draws);
    }
    int maxFirst = distribution.maxFirst();
    int maxSecond = distribution.maxSecond();
    rebase(end.score(varying[0]), maxFirst, rebased[0]);
    if (varyingCount == 2) rebase(end.score(varying[1]), maxSecond, rebased[1]);
    groups = 0;
    for (int k1 = 0; k1 <= maxFirst; k1++) {
      for (int k2 = 0; k2 <= maxSecond; k2++) {
        double p = distribution.probability(k1, k2);
        if (p == 0) continue;
        int first = rebased[0][k1];
        int second = varyingCount == 2 ? rebased[1][k2] : 0;
        int group = 0;
        while (group < groups && (groupFirst[group] != first || groupSecond[group] != second)) group++;
        if (group == groups) {
          groupFirst[group] = first;
          groupSecond[group] = second;
          groupProbability[group] = 0;
          groups++;
        }
        groupProbability[group] += p;
      }
    }
    Arrays.fill(expected, 0);
    for (int group = 0; group < groups; group++) {
      double p = groupProbability[group];
      tableValues(history, settle(outcome, start, groupFirst[group], groupSecond[group]), scratch);
      for (int seat = 0; seat < seats; seat++) {
        double value = scratch[seat];
        expected[seat] += p * value;
        variance[seat] += p * value * value;
      }
    }
    for (int seat = 0; seat < seats; seat++)
      variance[seat] = Math.max(0, variance[seat] - expected[seat] * expected[seat]);
  }

  /** 卓の値を書く。終局の卓には Φ を呼ばず、順位の効用をそのまま書く（Φ が終局の卓に返す値と同じ）。 */
  private void tableValues(RoundHistory history, TableState table, double[] out) {
    if (!table.gameOver()) {
      phi.values(history, table, out);
      return;
    }
    for (int seat = 0; seat < seats; seat++) out[seat] = utility.value(variant, table.rank(seat));
  }

  /** 局の結果を写して基本点を差し替え、精算した次局の卓を返す（次の呼び出しまで有効）。 */
  private TableState settle(RoundOutcome outcome, TableState start, int first, int second) {
    int winners = outcome.winnerCount();
    for (int i = 0; i < winners; i++) bases[i] = outcome.base(i);
    bases[varying[0]] = first;
    if (varyingCount == 2) bases[varying[1]] = second;
    RoundOutcome copy = hypothetical.outcome();
    int mask = outcome.riichiMask();
    if (outcome.kind() == EndKind.TSUMO) {
      copy.setTsumo(bases[0], outcome.winner(0), outcome.pao(0), mask);
    } else {
      copy.setRon(bases[0], outcome.winner(0), outcome.loser(), outcome.pao(0), mask);
      for (int i = 1; i < winners; i++) copy.addRonWinner(bases[i], outcome.winner(i), outcome.pao(i));
    }
    return hypothetical.settle(start);
  }

  /** 裏ドラの数 k ごとの基本点を書く。翻は、実際の採点の翻から裏ドラの分を引いて k を足したもの。 */
  private static void rebase(WinScore score, int max, int[] out) {
    int han = score.han() - score.ura();
    for (int k = 0; k <= max; k++) out[k] = ScoreMath.base(han + k, score.fu(), 0);
  }

  /**
   * 局末のプールを書き、裏ドラ表示牌の枚数を返す。
   *
   * @throws IllegalStateException 偶然のイベントの記録の末尾にある表示牌の枚数分のイベントが、裏ドラ表示牌でないとき（イベントが足りない
   *     ときを含む）
   */
  private int loadPool(FullView full) {
    tables.pool(full, pool);
    int draws = full.seat(0).indicatorCount();
    ChanceLog log = full.chance();
    int size = log.size();
    if (size < draws) throw new IllegalStateException("Only " + size + " chance events for " + draws + " ura indicators");
    for (int index = size - draws; index < size; index++) {
      if (log.kind(index) != ChanceKind.URA)
        throw new IllegalStateException("Chance event " + index + " is " + log.kind(index) + ", not an ura indicator");
      pool[Tiles.kind(log.tile(index))]++;
    }
    return draws;
  }

  /**
   * めくった裏ドラ表示牌で数えた裏ドラの数が、実際の採点と同じことを確かめる。
   *
   * @throws IllegalStateException 食い違うとき
   */
  private static void requireRealized(ChanceLog log, int draws, int[] count, WinScore score) {
    int realized = 0;
    for (int index = log.size() - draws; index < log.size(); index++) realized += count[Tiles.kind(log.tile(index))];
    if (realized != score.ura())
      throw new IllegalStateException("The ura indicators count " + realized + " ura but the win scored " + score);
  }

  /** 和了者が数える裏ドラの数を、表示牌の種類ごとに書く（和了の手・面子・抜いた北のうち、その表示牌が指す種類の枚数）。 */
  private void loadCounts(FullView full, int winner, int[] out) {
    tables.winningHand(full, winner, hand, owned);
    Arrays.fill(out, 0);
    int[] doraOf = tables.doraOf;
    for (long bits = tables.kinds; bits != 0; bits &= bits - 1) {
      int kind = Long.numberOfTrailingZeros(bits);
      out[kind] = owned[doraOf[kind]];
    }
  }

  /** 局に槓があったなら true。槍槓の局では、奪われた加槓が最後のイベントとして残る。 */
  private boolean kan(FullView full) {
    SeatView view = full.seat(0);
    for (int seat = 0; seat < seats; seat++) {
      for (int m = 0; m < view.meldCount(seat); m++) {
        if (Meld.of(view.meld(seat, m)).type().kan()) return true;
      }
    }
    return view.last().kind() == TurnEvent.Kind.KAN;
  }
}
