package com.epsilon.prism.input;

import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.core.GameState;
import com.epsilon.engine.WinPointDeltas;
import com.epsilon.prism.input.InputSchema.PassGrade;
import com.epsilon.prism.input.InputSchema.SeatField;
import com.epsilon.prism.observe.Observation;

/**
 * 追い越しの格（{@link PassGrade}）。自分と他家の 1 席の組ごとに、和了 1 つで和了者が相手の上に出るのに要る最小の格を、本場・供託込みの点の移動
 * （{@link WinPointDeltas}）で求める。上か下かは点、同点は絶対席の順で決める（今の順位と同じ決まり）。
 *
 * <p>格ごとの点の移動は、本場・供託なしの表を {@link WinPointDeltas} から一度だけ作り、判断ごとに本場・供託の分を同じ口の差から足す。だから点の
 * 決まりは {@link WinPointDeltas} の 1 か所だけにある。包は扱わない。
 */
final class ScoreLadder {

  /** 追い越しの組（席の欄と 1 対 1）。 */
  enum Pass {
    /** 自分がこの席からの直撃のロンで、この席を抜く。 */
    BY_DIRECT_RON(SeatField.PASS_BY_DIRECT_RON, true, WinMethod.RON, true),
    /** 自分がほかの席からのロンで、この席を抜く。 */
    BY_OTHER_RON(SeatField.PASS_BY_OTHER_RON, true, WinMethod.RON, false),
    /** 自分がツモで、この席を抜く。 */
    BY_TSUMO(SeatField.PASS_BY_TSUMO, true, WinMethod.TSUMO, true),
    /** この席が自分からの直撃のロンで、自分を抜く。 */
    PASSED_BY_RON(SeatField.PASSED_BY_RON, false, WinMethod.RON, true),
    /** この席がツモで、自分を抜く。 */
    PASSED_BY_TSUMO(SeatField.PASSED_BY_TSUMO, false, WinMethod.TSUMO, true);

    private final SeatField field;
    private final boolean selfWins;
    private final WinMethod method;
    private final boolean opponentPays;

    Pass(SeatField field, boolean selfWins, WinMethod method, boolean opponentPays) {
      this.field = field;
      this.selfWins = selfWins;
      this.method = method;
      this.opponentPays = opponentPays;
    }

    /** 書く席の欄。 */
    SeatField field() {
      return field;
    }
  }

  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final Pass[] PASSES = Pass.values();
  private static final PassGrade[] SCORED = scoredGrades();
  private static final int CHILD = 0;
  private static final int DEALER = 1;

  /** 和了者の受け取り（ロン）。[和了者が親か][格]。 */
  private static final int[][] RON_GAIN = new int[2][SCORED.length];

  /** 和了者の受け取り（ツモ）。[和了者が親か][格]。 */
  private static final int[][] TSUMO_GAIN = new int[2][SCORED.length];

  /** 子のツモで親が払う額。[格]。 */
  private static final int[] TSUMO_FROM_DEALER = new int[SCORED.length];

  /** ツモで子が払う額。[和了者が親か][格]。 */
  private static final int[][] TSUMO_FROM_CHILD = new int[2][SCORED.length];

  static {
    int[] deltas = new int[SEATS];
    for (int grade = 0; grade < SCORED.length; grade++) {
      int base = SCORED[grade].basePoints();
      WinPointDeltas.settleInto(base, 0, 1, 2, 0, 0, deltas);
      RON_GAIN[CHILD][grade] = deltas[0];
      WinPointDeltas.settleInto(base, 0, 1, 0, 0, 0, deltas);
      RON_GAIN[DEALER][grade] = deltas[0];
      WinPointDeltas.settleInto(base, 0, 0, 1, 0, 0, deltas);
      TSUMO_GAIN[CHILD][grade] = deltas[0];
      TSUMO_FROM_DEALER[grade] = -deltas[1];
      TSUMO_FROM_CHILD[CHILD][grade] = -deltas[2];
      WinPointDeltas.settleInto(base, 0, 0, 0, 0, 0, deltas);
      TSUMO_GAIN[DEALER][grade] = deltas[0];
      TSUMO_FROM_CHILD[DEALER][grade] = -deltas[1];
    }
  }

  private final PassGrade[] grades = new PassGrade[SEATS * PASSES.length];
  private final int[] scores = new int[SEATS];
  private final int[] absolute = new int[SEATS];
  private final int[] deltas = new int[SEATS];
  private int dealer;
  private int ronGainExtra;
  private int ronPayExtra;
  private int tsumoGainExtra;
  private int tsumoPayExtra;

  /**
   * 判断の点と本場・供託から、他家の全ての席の格を求める。
   *
   * @param o 判断の記録
   */
  void load(Observation o) {
    int honba = o.honba();
    int kyotaku = o.kyotaku();
    int base = SCORED[0].basePoints();
    WinPointDeltas.settleInto(base, 0, 1, 2, honba, kyotaku, deltas);
    ronGainExtra = deltas[0] - RON_GAIN[CHILD][0];
    ronPayExtra = -deltas[1] - RON_GAIN[CHILD][0];
    WinPointDeltas.settleInto(base, 0, 0, 1, honba, kyotaku, deltas);
    tsumoGainExtra = deltas[0] - TSUMO_GAIN[CHILD][0];
    tsumoPayExtra = -deltas[2] - TSUMO_FROM_CHILD[CHILD][0];
    dealer = o.dealer();
    for (int seat = 0; seat < SEATS; seat++) {
      scores[seat] = o.score(seat);
      absolute[seat] = o.absoluteSeat(seat);
    }
    for (int seat = 1; seat < SEATS; seat++) {
      for (Pass pass : PASSES) {
        grades[seat * PASSES.length + pass.ordinal()] = grade(pass, seat);
      }
    }
  }

  /**
   * 格を返す。
   *
   * @param seat 他家の相対席（1〜3）
   * @param pass 組
   * @return 格
   */
  PassGrade grade(int seat, Pass pass) {
    if (seat < 1 || seat >= SEATS) {
      throw new IndexOutOfBoundsException("pass grades are for other seats: " + seat);
    }
    return grades[seat * PASSES.length + pass.ordinal()];
  }

  private PassGrade grade(Pass pass, int seat) {
    int winner = pass.selfWins ? 0 : seat;
    int loser = pass.selfWins ? seat : 0;
    if (above(scores[winner], absolute[winner], scores[loser], absolute[loser])) {
      return PassGrade.NOT_NEEDED;
    }
    int role = winner == dealer ? DEALER : CHILD;
    for (int grade = 0; grade < SCORED.length; grade++) {
      int gain;
      int loss;
      if (pass.method == WinMethod.RON) {
        gain = RON_GAIN[role][grade] + ronGainExtra;
        loss = pass.opponentPays ? RON_GAIN[role][grade] + ronPayExtra : 0;
      } else {
        gain = TSUMO_GAIN[role][grade] + tsumoGainExtra;
        int share =
            role == DEALER
                ? TSUMO_FROM_CHILD[DEALER][grade]
                : loser == dealer ? TSUMO_FROM_DEALER[grade] : TSUMO_FROM_CHILD[CHILD][grade];
        loss = share + tsumoPayExtra;
      }
      if (above(
          scores[winner] + gain, absolute[winner], scores[loser] - loss, absolute[loser])) {
        return SCORED[grade];
      }
    }
    return PassGrade.UNREACHABLE;
  }

  /** a が b の上にいる（点が多いか、同点で絶対席が若い）。 */
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
