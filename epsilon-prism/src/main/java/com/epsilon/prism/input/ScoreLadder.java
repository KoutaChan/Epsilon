package com.epsilon.prism.input;

import com.epsilon.agent.facts.WaitScores.Method;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;

/**
 * 追い越しに要る和了の格（{@link PassGrade}）を求める。2 席の組ごとに、和了者が 1 回の和了で相手の上に出るのに要る最小の格を、
 * 本場と供託を含めた点の移動から求める。点の移動は、{@link Settlement} の払い（三人打ちはツモ損）に、場の供託（{@link Variant#DEPOSIT} ×
 * 本数）を和了者の分として足したもの。上か下かは点で決め、同点なら絶対席の若い方を上とする（今の順位と同じ決め方）。方策の入力は判断の記録
 * （自分と他家）から求める。
 *
 * <p>本場と供託を除いた格ごとの点の移動は、{@link Settlement} から一度だけ表に作っておく。読むたびに、本場と供託の分を同じ式で求めた差
 * として足す。こうすると点数計算を {@link Settlement} の 1 か所だけに置ける。包は扱わない。表は、席 0 が和了し、ロンなら席 1 が放銃する
 * 局で作る。子が和了する局の親は、ロンなら席 2、ツモなら席 1。
 */
public final class ScoreLadder {

  /** 追い越し方。 */
  public enum Pass {
    /** 自分がこの席からの直撃のロンで、この席を抜く。 */
    BY_DIRECT_RON(true, Method.RON, true),
    /** 自分がほかの席からのロンで、この席を抜く。 */
    BY_OTHER_RON(true, Method.RON, false),
    /** 自分がツモで、この席を抜く。 */
    BY_TSUMO(true, Method.TSUMO, true),
    /** この席が自分からの直撃のロンで、自分を抜く。 */
    PASSED_BY_RON(false, Method.RON, true),
    /** この席がツモで、自分を抜く。 */
    PASSED_BY_TSUMO(false, Method.TSUMO, true);

    private final boolean selfWins;
    private final Method method;
    private final boolean opponentPays;

    Pass(boolean selfWins, Method method, boolean opponentPays) {
      this.selfWins = selfWins;
      this.method = method;
      this.opponentPays = opponentPays;
    }
  }

  private static final Pass[] PASSES = Pass.values();
  private static final PassGrade[] SCORED = scoredGrades();
  private static final int CHILD = 0;
  private static final int DEALER = 1;

  private final Variant variant;
  private final int seats;

  /** ロン和了で和了者が受け取る点。添字は [和了者が親か][格]。 */
  private final int[][] ronGain = new int[2][SCORED.length];

  /** ツモ和了で和了者が受け取る点。添字は [和了者が親か][格]。 */
  private final int[][] tsumoGain = new int[2][SCORED.length];

  /** 子のツモ和了で親が払う点。添字は [格]。 */
  private final int[] tsumoFromDealer = new int[SCORED.length];

  /** ツモ和了で子が払う点。添字は [和了者が親か][格]。 */
  private final int[][] tsumoFromChild = new int[2][SCORED.length];

  private final PassGrade[] grades;
  private final int[] scores;
  private final int[] absolute;
  private final int[] deltas;
  private int dealer;
  private int ronGainExtra;
  private int ronPayExtra;
  private int tsumoGainExtra;
  private int tsumoPayExtra;

  /**
   * そのルールの格の表を作る。
   *
   * @param variant ルール（四人打ちか三人打ち）
   */
  public ScoreLadder(Variant variant) {
    this.variant = variant;
    seats = variant.seats();
    grades = new PassGrade[seats * PASSES.length];
    scores = new int[seats];
    absolute = new int[seats];
    deltas = new int[seats];
    for (int grade = 0; grade < SCORED.length; grade++) {
      int base = SCORED[grade].basePoints();
      ron(base, 2, 0, 0, deltas);
      ronGain[CHILD][grade] = deltas[0];
      ron(base, 0, 0, 0, deltas);
      ronGain[DEALER][grade] = deltas[0];
      tsumo(base, 1, 0, 0, deltas);
      tsumoGain[CHILD][grade] = deltas[0];
      tsumoFromDealer[grade] = -deltas[1];
      tsumoFromChild[CHILD][grade] = -deltas[2];
      tsumo(base, 0, 0, 0, deltas);
      tsumoGain[DEALER][grade] = deltas[0];
      tsumoFromChild[DEALER][grade] = -deltas[1];
    }
  }

  /**
   * 判断の記録の点数と本場・供託を読み、他家の全ての席について格を求める。
   *
   * @param o 判断の記録
   * @throws IllegalArgumentException 判断の記録のルールが表のルールと違うとき
   */
  public void load(Observation o) {
    if (o.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + o.variant());
    }
    settle(o.honba(), o.kyotaku(), o.dealer());
    for (int seat = 0; seat < seats; seat++) {
      scores[seat] = o.score(seat);
      absolute[seat] = o.absoluteSeat(seat);
    }
    for (int seat = 1; seat < seats; seat++) {
      for (Pass pass : PASSES) {
        grades[seat * PASSES.length + pass.ordinal()] =
            pass.selfWins ? between(pass, 0, seat) : between(pass, seat, 0);
      }
    }
  }

  /**
   * 格を返す。
   *
   * @param seat 他家の相対席（1 から）
   * @param pass 追い越し方
   * @return 格
   */
  public PassGrade grade(int seat, Pass pass) {
    if (seat < 1 || seat >= seats) {
      throw new IndexOutOfBoundsException("pass grades are for other seats: " + seat);
    }
    return grades[seat * PASSES.length + pass.ordinal()];
  }

  /**
   * 読んだ卓で、和了する席が 1 回の和了で抜かれる席の上に出るのに要る最小の格を返す。追い越し方からは、ロンかツモかと、相手が払うか
   * だけを使う。
   *
   * @param pass 追い越し方
   * @param winner 和了する席（読んだ卓での席の番号）
   * @param loser 抜かれる席
   * @return 格。既に上にいれば NOT_NEEDED、どの格でも届かなければ UNREACHABLE
   */
  public PassGrade between(Pass pass, int winner, int loser) {
    if (above(scores[winner], absolute[winner], scores[loser], absolute[loser])) {
      return PassGrade.NOT_NEEDED;
    }
    int role = winner == dealer ? DEALER : CHILD;
    for (int grade = 0; grade < SCORED.length; grade++) {
      int gain;
      int loss;
      if (pass.method == Method.RON) {
        gain = ronGain[role][grade] + ronGainExtra;
        loss = pass.opponentPays ? ronGain[role][grade] + ronPayExtra : 0;
      } else {
        gain = tsumoGain[role][grade] + tsumoGainExtra;
        int share =
            role == DEALER
                ? tsumoFromChild[DEALER][grade]
                : loser == dealer ? tsumoFromDealer[grade] : tsumoFromChild[CHILD][grade];
        loss = share + tsumoPayExtra;
      }
      if (above(
          scores[winner] + gain, absolute[winner], scores[loser] - loss, absolute[loser])) {
        return SCORED[grade];
      }
    }
    return PassGrade.UNREACHABLE;
  }

  /** 本場と供託による点の増減（格に依らない）を求め、親の席を覚える。 */
  private void settle(int honba, int kyotaku, int dealer) {
    int base = SCORED[0].basePoints();
    ron(base, 2, honba, kyotaku, deltas);
    ronGainExtra = deltas[0] - ronGain[CHILD][0];
    ronPayExtra = -deltas[1] - ronGain[CHILD][0];
    tsumo(base, 1, honba, kyotaku, deltas);
    tsumoGainExtra = deltas[0] - tsumoGain[CHILD][0];
    tsumoPayExtra = -deltas[2] - tsumoFromChild[CHILD][0];
    this.dealer = dealer;
  }

  /** 席 0 が席 1 からロン和了したときの点の移動を {@code deltas} に書く。基本点は {@code base}、親は席 {@code dealer}。供託は席 0 が受け取る。 */
  private void ron(int base, int dealer, int honba, int kyotaku, int[] deltas) {
    Arrays.fill(deltas, 0);
    Settlement.ron(variant, base, 0, 1, dealer, honba, RoundOutcome.NO_SEAT, deltas);
    deltas[0] += kyotaku * Variant.DEPOSIT;
  }

  /** 席 0 がツモ和了したときの点の移動を {@code deltas} に書く。基本点は {@code base}、親は席 {@code dealer}。供託は席 0 が受け取る。 */
  private void tsumo(int base, int dealer, int honba, int kyotaku, int[] deltas) {
    Arrays.fill(deltas, 0);
    Settlement.tsumo(variant, base, 0, dealer, honba, RoundOutcome.NO_SEAT, deltas);
    deltas[0] += kyotaku * Variant.DEPOSIT;
  }

  /** 席 A が席 B より上にいるか。点が多いか、同点で絶対席が若ければ上。 */
  private static boolean above(int scoreA, int seatA, int scoreB, int seatB) {
    return scoreA > scoreB || scoreA == scoreB && seatA < seatB;
  }

  private static PassGrade[] scoredGrades() {
    int count = 0;
    for (PassGrade grade : PassGrade.values()) {
      count += grade.scored() ? 1 : 0;
    }
    PassGrade[] scored = new PassGrade[count];
    int index = 0;
    for (PassGrade grade : PassGrade.values()) {
      if (grade.scored()) {
        scored[index++] = grade;
      }
    }
    return scored;
  }
}
