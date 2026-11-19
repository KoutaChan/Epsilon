package com.epsilon.grp;

import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;

/**
 * 次局の結果の確率 q(o|s) を出すネットワーク（純 Java）。h1 = ReLU(W1·x ＋ b1)、h2 = ReLU(W2·h1 ＋ b2)、z = W3·h2 ＋ b3 で、z は
 * ロジット（並びは HandOutcomes と OutcomeShape が決める）。終わり方のロジットの softmax が終わり方の確率になる。打点の確率は、
 * 和了者が 1 人の終わり方ごとに、和了者と和了の形の打点のロジットの softmax を取ったもの。「立直した」の確率は、終わり方と席
 * ごとに、相対席と立場のロジットの σ を取ったもの。形が OutcomeLink のロジットを持てば、softmax と σ を取る前に足す。順位の
 * ロジットは打点ごとに、その打点で和了者が就く順位の分を足す。和了者ごとのロジットは、和了者でない席に足す。1 つの結果の q は、
 * 終わり方の確率に、和了者ごとの打点の確率と、立直の有無を確率に入れる席ごとの立直の確率を掛けたもの。どの項を掛けるかは
 * HandOutcomes が決める。
 *
 * <p>層の計算には Dense を使う。隠れ層は float、ロジットから先は double で求め、指数と対数は StrictMath で求める。値は卓だけで
 * 決まり、ビット単位で再現する。重みは書き換えず、作業領域はスレッドごとに持つ。
 */
public final class OutcomeNetwork {

  private final OutcomeShape shape;
  private final float[] weights;

  /** テンソルごとの、重みの配列の中での開始位置（OutcomeTensor の宣言の順）。 */
  private final int[] starts;

  private final HandOutcomes hands;
  private final int seats;
  private final int width;
  private final int hidden;
  private final int logits;
  private final int mains;

  /** 順位のロジットと、和了者ごとのロジットの開始位置。形が持たなければ −1。 */
  private final int rankStart;

  private final int winnerStart;

  private OutcomeNetwork(OutcomeShape shape, float[] weights) {
    this.shape = shape;
    this.weights = weights;
    starts = OutcomeTensor.starts(shape);
    hands = shape.outcomes();
    seats = shape.variant().seats();
    width = shape.width();
    hidden = shape.hidden();
    logits = shape.logits();
    mains = hands.mainCount();
    rankStart = shape.linkStart(OutcomeLink.RANK);
    winnerStart = shape.linkStart(OutcomeLink.WINNER);
  }

  /**
   * 重みのコピーを持つネットワークを作る。重みの数が形に合わないか、有限でない値があれば IllegalArgumentException を投げる。
   *
   * @param weights OutcomeTensor の宣言の順に並べた重み
   */
  public static OutcomeNetwork of(OutcomeShape shape, float[] weights) {
    int expected = OutcomeTensor.starts(shape)[OutcomeTensor.values().length];
    if (weights.length != expected) {
      throw new IllegalArgumentException(
          shape + " needs " + expected + " weights, got " + weights.length);
    }
    for (int index = 0; index < weights.length; index++) {
      if (!Float.isFinite(weights[index])) {
        throw new IllegalArgumentException("Weight " + index + " is not finite: " + weights[index]);
      }
    }
    return new OutcomeNetwork(shape, weights.clone());
  }

  public OutcomeShape shape() {
    return shape;
  }

  public Variant variant() {
    return shape.variant();
  }

  /** 重みのコピーを返す。並びは OutcomeTensor の宣言の順。 */
  public float[] weights() {
    return weights.clone();
  }

  /** 半荘が続く卓の、各項の対数確率と確率を作業領域に書く。 */
  public void forward(GrpState state, OutcomeWorkspace workspace) {
    forward(check(workspace).encoder.load(state), workspace);
  }

  /** 半荘が続く卓の、各項の対数確率と確率を作業領域に書く。立直の有無を確率に入れない席の立直の項には書かない。 */
  public void forward(TableState table, OutcomeWorkspace workspace) {
    check(workspace).encoder.encode(table, workspace.input);
    Dense.relu(
        weights,
        starts[OutcomeTensor.W1.ordinal()],
        starts[OutcomeTensor.B1.ordinal()],
        workspace.input,
        width,
        workspace.hidden1,
        hidden);
    Dense.relu(
        weights,
        starts[OutcomeTensor.W2.ordinal()],
        starts[OutcomeTensor.B2.ordinal()],
        workspace.hidden1,
        hidden,
        workspace.hidden2,
        hidden);
    Dense.logits(
        weights,
        starts[OutcomeTensor.W3.ordinal()],
        starts[OutcomeTensor.B3.ordinal()],
        workspace.hidden2,
        hidden,
        workspace.logits,
        logits);
    logSoftmax(workspace.logits, hands.mainLogit(0), mains, workspace, hands.mainTerm(0));
    grades(table, workspace);
    riichi(workspace);
  }

  /** 和了者が 1 人の終わり方ごとに、打点の対数確率と確率を書く。 */
  private void grades(TableState table, OutcomeWorkspace workspace) {
    double[] z = workspace.logits;
    int methods = WinMethod.values().length;
    for (int main = 0; main < hands.wins(); main++) {
      int winner = hands.winner(main);
      WinMethod method =
          hands.section(main) == OutcomeSection.TSUMO ? WinMethod.TSUMO : WinMethod.RON;
      int from = hands.gradeLogit(winner, method, 0);
      int to = hands.gradeTerm(main, 0);
      if (rankStart < 0) {
        logSoftmax(z, from, HandOutcomes.BASES, workspace, to);
        continue;
      }
      int ranks = rankStart + (winner * methods + method.ordinal()) * seats;
      double[] grades = workspace.grades;
      workspace.ranks.ranks(table, main, workspace.winRanks);
      for (int base = 0; base < HandOutcomes.BASES; base++) {
        grades[base] = z[from + base] + z[ranks + workspace.winRanks[base]];
      }
      logSoftmax(grades, 0, HandOutcomes.BASES, workspace, to);
    }
  }

  /** 終わり方と、立直の有無を確率に入れる席ごとに、「立直した」と「立直していない」の対数確率と確率を書く。 */
  private void riichi(OutcomeWorkspace workspace) {
    double[] z = workspace.logits;
    double[] terms = workspace.terms;
    double[] probabilities = workspace.probabilities;
    int roles = RiichiRole.values().length;
    for (int main = 0; main < mains; main++) {
      int winner = hands.winner(main);
      int second = hands.secondWinner(main);
      for (int index = 0; index < hands.factorCount(main); index++) {
        int seat = hands.factorSeat(main, index);
        RiichiRole role = hands.role(main, seat);
        double value = z[hands.riichiLogit(seat, role)];
        if (winnerStart >= 0 && role != RiichiRole.WINNER) {
          int winners = winnerStart + (seat * roles + role.ordinal()) * seats;
          if (winner != HandOutcomes.NONE) {
            value += z[winners + winner];
          }
          if (second != HandOutcomes.NONE) {
            value += z[winners + second];
          }
        }
        int declared = hands.riichiTerm(main, seat, true);
        int quiet = hands.riichiTerm(main, seat, false);
        double small = StrictMath.exp(-Math.abs(value));
        double log1p = StrictMath.log1p(small);
        double large = 1 / (1 + small);
        if (value >= 0) {
          terms[declared] = -log1p;
          terms[quiet] = -value - log1p;
          probabilities[declared] = large;
          probabilities[quiet] = small * large;
        } else {
          terms[declared] = value - log1p;
          terms[quiet] = -log1p;
          probabilities[declared] = small * large;
          probabilities[quiet] = large;
        }
      }
    }
  }

  private OutcomeWorkspace check(OutcomeWorkspace workspace) {
    if (workspace.shape != shape && !workspace.shape.equals(shape)) {
      throw new IllegalArgumentException(
          "Workspace for " + workspace.shape + " used with " + shape);
    }
    return workspace;
  }

  /**
   * z の 1 つの区間について log softmax と softmax を求め、作業領域の対数確率と確率に書く。
   *
   * @param from 区間の開始位置
   * @param count 区間の長さ
   * @param to 書き先の項の開始位置
   */
  private static void logSoftmax(
      double[] z, int from, int count, OutcomeWorkspace workspace, int to) {
    double max = z[from];
    for (int k = 1; k < count; k++) {
      max = Math.max(max, z[from + k]);
    }
    double sum = 0;
    for (int k = 0; k < count; k++) {
      double exponent = StrictMath.exp(z[from + k] - max);
      workspace.probabilities[to + k] = exponent;
      sum += exponent;
    }
    double log = StrictMath.log(sum);
    for (int k = 0; k < count; k++) {
      workspace.terms[to + k] = z[from + k] - max - log;
      workspace.probabilities[to + k] /= sum;
    }
  }

  @Override
  public String toString() {
    return "OutcomeNetwork[" + shape + "]";
  }
}
