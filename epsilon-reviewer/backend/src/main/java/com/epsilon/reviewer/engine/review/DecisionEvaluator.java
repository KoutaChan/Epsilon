package com.epsilon.reviewer.engine.review;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.reviewer.engine.view.CandidateTiles;
import com.epsilon.spi.ReviewEvaluation;
import java.util.List;

/** 方策の唯一の呼び出し。解析ともしもが同じ評価を使う。 */
public final class DecisionEvaluator {
  /** 損失から導いた表示の確率と方策の確率の差がこれを超える判断だけ確率を持つ。 */
  private static final double PROBABILITY_TOLERANCE = 1e-6;

  private DecisionEvaluator() {}

  /**
   * 局面の公開観測だけを方策へ渡し、候補を表示の順に並べる。価値があれば損失の昇順・同じなら確率の降順、無ければ確率の降順で、同じ値は
   * 合法手の順のまま。
   *
   * @param chosenSlot 合法手の中の牌譜の手。無ければ -1
   * @throws IllegalStateException 方策の評価の数が合法手の数と違う
   */
  public static EvaluatedDecision evaluate(
      ReviewModel model, GameState state, int seat, List<Action> legal, int chosenSlot) {
    ReviewEvaluation evaluation = model.evaluate(state.publicObservation(seat), legal);
    int count = legal.size();
    if (evaluation.actions() != count)
      throw new IllegalStateException(
          "Policy evaluated " + evaluation.actions() + " actions for " + count + " legal actions.");
    float[] probabilities = evaluation.probabilities();
    float[] losses = evaluation.valued() ? evaluation.losses() : null;
    int[] order = order(probabilities, losses);
    int[] actionIds = new int[count];
    float[] values = new float[count];
    float[] shown = new float[count];
    String[][] tiles = new String[count][];
    int chosen = -1;
    for (int i = 0; i < count; i++) {
      int slot = order[i];
      Action action = legal.get(slot);
      actionIds[i] = action.toIndex();
      values[i] = losses == null ? probabilities[slot] : losses[slot];
      shown[i] = probabilities[slot];
      tiles[i] = CandidateTiles.of(action, state, seat);
      if (slot == chosenSlot) chosen = i;
    }
    boolean derivable = losses == null || derivable(values, shown, model.temperature());
    return new EvaluatedDecision(
        actionIds, values, tiles, chosen, derivable ? null : shown, losses != null);
  }

  /** 表示の順の合法手の位置（安定な挿入の並べ替え。候補は数十まで）。 */
  private static int[] order(float[] probabilities, float[] losses) {
    int[] order = new int[probabilities.length];
    for (int slot = 0; slot < order.length; slot++) {
      int position = slot;
      while (position > 0 && after(order[position - 1], slot, probabilities, losses)) {
        order[position] = order[position - 1];
        position--;
      }
      order[position] = slot;
    }
    return order;
  }

  /** 合法手 a が合法手 b より後に並ぶか。 */
  private static boolean after(int a, int b, float[] probabilities, float[] losses) {
    if (losses != null) {
      int byLoss = Float.compare(losses[a], losses[b]);
      if (byLoss != 0) return byLoss > 0;
    }
    return Float.compare(probabilities[a], probabilities[b]) < 0;
  }

  /**
   * protocol/result.ts の displayProbabilities（exp(-loss/temperature) を binary64 で候補の順に足し、各値を
   * float32 に丸める）が方策の確率と許容の内で合うか。
   */
  private static boolean derivable(float[] losses, float[] probabilities, double temperature) {
    double[] weights = new double[losses.length];
    double total = 0;
    for (int i = 0; i < losses.length; i++) {
      weights[i] = Math.exp(-losses[i] / temperature);
      total += weights[i];
    }
    for (int i = 0; i < losses.length; i++)
      if (Math.abs((float) (weights[i] / total) - probabilities[i]) > PROBABILITY_TOLERANCE)
        return false;
    return true;
  }
}
