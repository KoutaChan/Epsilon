package com.epsilon.grp.train;

import com.epsilon.grp.GrpEncoder;
import com.epsilon.grp.GrpLayout;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.HandOutcomes;
import com.epsilon.grp.OutcomeLink;
import com.epsilon.grp.OutcomeSection;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.OutcomeTerm;
import com.epsilon.grp.RiichiRole;
import com.epsilon.grp.WinMethod;
import com.epsilon.grp.WinRanks;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;

/**
 * OutcomeNetwork の学習に使う行を、入力のバッチに詰める。配列は作るときに 1 度だけ確保し、バッチごとに使い回す。行を足すときに、
 * その行の値を全て書き直す。行 i の入力は {@code inputs[i × width ...]} に、重みは {@code weights[i]} に置く。行 i の結果 k の項の
 * 番号は {@code terms[(i × outcomes + k) × stride ...]} に置く。項の無い所には項の総数を入れる（ネットワークは値が 0 の列を 1 つ
 * 足して読む）。結果の無い k では {@code mask[i × outcomes + k]} を EMPTY に、ほかは 0 にする。
 *
 * <p>ロジットから項を作るのに使うロジットの位置も、行ごとに持つ。打点の項ごとの打点のロジットの位置と、立直の項ごとの相対席と
 * 立場のロジットの位置は、どの行でも同じ。順位のロジットの位置は、行の卓で決まる（その打点で和了者が就く順位の分）。和了者ごとの
 * ロジットの位置は、和了者が無い所ではロジットの総数にする（ネットワークは値が 0 の列を 1 つ足して読む）。
 */
final class OutcomeBatch implements TrainingLoop.Batch<OutcomeRows> {

  /** 結果の無い所の対数確率に足す値。exp すると 0 になり、float32 に収まる。 */
  static final float EMPTY = -1e30f;

  final int width;
  final int outcomes;
  final int stride;
  final int padding;
  final float[] inputs;
  final long[] terms;
  final float[] mask;
  final float[] weights;

  /** 打点の項ごとの、打点のロジットの位置。並びは [行 × 打点の項の数 ＋ 打点の項]。 */
  final long[] gradeLogits;

  /** 打点の項ごとの、順位のロジットの位置。形が順位のロジットを持たなければ null。 */
  final long[] rankLogits;

  /** 立直の項ごとの、相対席と立場のロジットの位置。並びは [行 × 「立直した」の項の数 ＋ 項]。項を持たない席では 0。 */
  final long[] riichiLogits;

  /** 立直の項ごとの、1 人目と 2 人目の和了者のロジットの位置。形が和了者ごとのロジットを持たなければ null。 */
  final long[] winnerLogits;

  final long[] secondLogits;

  int size;

  private final int capacity;
  private final GrpEncoder encoder;
  private final float[] row;
  private final HandOutcomes hands;
  private final int grades;
  private final int rankStart;
  private final WinRanks ranks;
  private final int[] winRanks = new int[HandOutcomes.BASES];
  private final TableState table;

  /**
   * バッチの配列を確保する。
   *
   * @param outcomes 1 行の結果の数の上限
   */
  OutcomeBatch(OutcomeShape shape, int capacity, int outcomes) {
    GrpLayout layout = shape.layout();
    encoder = GrpEncoder.of(layout);
    this.capacity = capacity;
    this.outcomes = outcomes;
    stride = shape.outcomes().maxTerms();
    width = layout.width();
    padding = shape.outcomes().terms();
    inputs = new float[capacity * width];
    terms = new long[capacity * outcomes * stride];
    mask = new float[capacity * outcomes];
    weights = new float[capacity];
    row = new float[width];
    Variant variant = shape.variant();
    hands = shape.outcomes();
    int seats = variant.seats();
    grades = OutcomeTerm.GRADE.width(variant);
    gradeLogits = new long[capacity * grades];
    for (int main = 0; main < hands.wins(); main++) {
      WinMethod method =
          hands.section(main) == OutcomeSection.TSUMO ? WinMethod.TSUMO : WinMethod.RON;
      for (int base = 0; base < HandOutcomes.BASES; base++) {
        gradeLogits[main * HandOutcomes.BASES + base] =
            hands.gradeLogit(hands.winner(main), method, base);
      }
    }
    repeat(gradeLogits, grades, capacity);
    rankStart = shape.linkStart(OutcomeLink.RANK);
    rankLogits = rankStart < 0 ? null : new long[capacity * grades];
    ranks = rankStart < 0 ? null : new WinRanks(variant);
    table = new TableState(variant);
    int riichi = OutcomeTerm.RIICHI.width(variant);
    riichiLogits = new long[capacity * riichi];
    int winnerStart = shape.linkStart(OutcomeLink.WINNER);
    winnerLogits = winnerStart < 0 ? null : new long[capacity * riichi];
    secondLogits = winnerStart < 0 ? null : new long[capacity * riichi];
    int none = shape.logits();
    for (int main = 0; main < hands.mainCount(); main++) {
      for (int seat = 0; seat < seats; seat++) {
        RiichiRole role = hands.role(main, seat);
        int at = main * seats + seat;
        riichiLogits[at] = role == null ? 0 : hands.riichiLogit(seat, role);
        if (winnerStart < 0) {
          continue;
        }
        boolean linked = role != null && role != RiichiRole.WINNER;
        int winner = hands.winner(main);
        int second = hands.secondWinner(main);
        int winners =
            linked
                ? winnerStart + (seat * RiichiRole.values().length + role.ordinal()) * seats
                : none;
        winnerLogits[at] = linked && winner != HandOutcomes.NONE ? winners + winner : none;
        secondLogits[at] = linked && second != HandOutcomes.NONE ? winners + second : none;
      }
    }
    repeat(riichiLogits, riichi, capacity);
    if (winnerStart >= 0) {
      repeat(winnerLogits, riichi, capacity);
      repeat(secondLogits, riichi, capacity);
    }
  }

  /** 配列の先頭の 1 行分を、残りの行に写す。 */
  private static void repeat(long[] values, int width, int rows) {
    for (int row = 1; row < rows; row++) {
      System.arraycopy(values, 0, values, row * width, width);
    }
  }

  @Override
  public int capacity() {
    return capacity;
  }

  @Override
  public void clear() {
    size = 0;
  }

  boolean full() {
    return size == capacity;
  }

  /** 行を 1 つ足す。行の重みには倍率を掛ける。 */
  @Override
  public void add(OutcomeRows rows, int index, float scale) {
    if (full()) {
      throw new IllegalStateException("The batch already holds " + capacity + " rows");
    }
    int count = rows.outcomes(index);
    if (count > outcomes) {
      throw new IllegalArgumentException(
          "Row " + index + " has " + count + " outcomes, the batch holds " + outcomes);
    }
    GrpState state = GrpState.ofKey(rows.variant(), rows.keys[index]);
    encoder.encode(state, row);
    System.arraycopy(row, 0, inputs, size * width, width);
    if (rankLogits != null) {
      state.copyInto(table);
      int seats = table.variant().seats();
      int methods = WinMethod.values().length;
      for (int main = 0; main < hands.wins(); main++) {
        WinMethod method =
            hands.section(main) == OutcomeSection.TSUMO ? WinMethod.TSUMO : WinMethod.RON;
        int first = rankStart + (hands.winner(main) * methods + method.ordinal()) * seats;
        ranks.ranks(table, main, winRanks);
        for (int base = 0; base < HandOutcomes.BASES; base++) {
          rankLogits[size * grades + main * HandOutcomes.BASES + base] = first + winRanks[base];
        }
      }
    }
    int first = rows.firstOutcome[index];
    for (int k = 0; k < outcomes; k++) {
      int at = (size * outcomes + k) * stride;
      for (int j = 0; j < stride; j++) {
        short term = k < count ? rows.terms[(first + k) * stride + j] : -1;
        terms[at + j] = term >= 0 ? term : padding;
      }
      mask[size * outcomes + k] = k < count ? 0f : EMPTY;
    }
    weights[size] = rows.weightOf(index) * scale;
    size++;
  }
}
