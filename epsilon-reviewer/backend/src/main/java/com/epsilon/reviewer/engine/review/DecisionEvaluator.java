package com.epsilon.reviewer.engine.review;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.reviewer.engine.view.CandidateTiles;

/** 判断を評価する。方策を呼び出すのはこのクラスだけで、解析ももしもも同じ評価を使う。 */
public final class DecisionEvaluator {
  private static final int PASS_CODE = Action.code(Action.Type.PASS);

  private DecisionEvaluator() {}

  /**
   * 判断する席の O だけを方策に渡して評価し、候補を表示の順に並べる。価値を返すモデルでは、損失（最大の価値 − 価値）の小さい順に
   * 並べ、損失が同じなら確率の大きい順にする。価値を返さないモデルでは、確率の大きい順に並べる。どちらでも、同じ値の候補はモデルが
   * 貪欲に選ぶ手を先にし（PRISM の同値な打牌では、切る方の手が先に来る）、ほかは合法手の順を保つ。価値を返すモデルの表示する確率は、
   * 損失とモデルの温度から作る。
   *
   * @param model 評価に使うモデル
   * @param observation 判断する席の O（呼び出しの間だけ読む）
   * @param view 判断する席から見える卓（候補の鳴き・槓に使う牌を読む）
   * @param decisions この時点で判断する席と、席ごとの合法手
   * @param slot 判断する席が {@code decisions} の何番目か
   * @param chosen 牌譜の手が合法手の何番目か。牌譜に無ければ -1
   * @return 評価した判断
   * @throws IllegalStateException 方策が返した評価の数が合法手の数と違う
   */
  public static EvaluatedDecision evaluate(
      ReviewModel model,
      Observation observation,
      SeatView view,
      Decisions decisions,
      int slot,
      int chosen) {
    Evaluation evaluation = model.evaluate(observation);
    int count = decisions.legalCount(slot);
    if (evaluation.probabilities().length != count)
      throw new IllegalStateException(
          "Policy evaluated "
              + evaluation.probabilities().length
              + " actions for "
              + count
              + " legal actions.");
    float[] losses = evaluation.values() == null ? null : losses(evaluation.values());
    float[] probabilities =
        losses == null ? evaluation.probabilities() : display(losses, model.temperature());
    int[] order = order(probabilities, losses, evaluation.greedy());
    int[] actionIds = new int[count];
    float[] values = new float[count];
    String[][] tiles = new String[count][];
    int position = -1;
    for (int i = 0; i < count; i++) {
      int index = order[i];
      int code = decisions.legal(slot, index);
      actionIds[i] = code;
      values[i] = losses == null ? probabilities[index] : losses[index];
      tiles[i] = CandidateTiles.of(Action.of(code), view);
      if (index == chosen) position = i;
    }
    return new EvaluatedDecision(actionIds, values, tiles, position, losses != null);
  }

  /**
   * 見送りしか選べない応答の評価を、方策を呼ばずに作る。もしもの変更で鳴きもロンもできなくなった判断に使う。
   *
   * @param model 評価に使うモデル
   * @param recorded 牌譜の手の行動の番号。牌譜に無ければ -1
   * @return 見送りだけを候補にした評価
   */
  public static EvaluatedDecision passOnly(ReviewModel model, int recorded) {
    boolean valued = model.valued();
    return new EvaluatedDecision(
        new int[] {PASS_CODE},
        new float[] {valued ? 0f : 1f},
        new String[][] {null},
        recorded == PASS_CODE ? 0 : -1,
        valued);
  }

  /** 手ごとの損失（最大の価値 − 価値）を返す。 */
  private static float[] losses(float[] values) {
    float best = Float.NEGATIVE_INFINITY;
    for (float value : values) best = Math.max(best, value);
    float[] losses = new float[values.length];
    for (int i = 0; i < values.length; i++) losses[i] = best - values[i];
    return losses;
  }

  /** 損失と温度から表示する確率を作る。足し方は protocol/result.ts の displayProbabilities と同じ。 */
  private static float[] display(float[] losses, double temperature) {
    double[] weights = new double[losses.length];
    double total = 0;
    for (int i = 0; i < losses.length; i++) {
      weights[i] = Math.exp(-losses[i] / temperature);
      total += weights[i];
    }
    float[] probabilities = new float[losses.length];
    for (int i = 0; i < losses.length; i++) probabilities[i] = (float) (weights[i] / total);
    return probabilities;
  }

  /**
   * 合法手の位置を表示の順に並べて返す。候補は多くても数十なので、安定な挿入ソートで並べる。
   *
   * @param greedy モデルが貪欲に選ぶ手（同じ値の候補の中で先にする）
   */
  private static int[] order(float[] probabilities, float[] losses, int greedy) {
    int[] order = new int[probabilities.length];
    for (int slot = 0; slot < order.length; slot++) {
      int position = slot;
      while (position > 0 && after(order[position - 1], slot, probabilities, losses, greedy)) {
        order[position] = order[position - 1];
        position--;
      }
      order[position] = slot;
    }
    return order;
  }

  /** 1 つ目の合法手が、2 つ目の合法手より後に並ぶか。 */
  private static boolean after(int a, int b, float[] probabilities, float[] losses, int greedy) {
    if (losses != null) {
      int byLoss = Float.compare(losses[a], losses[b]);
      if (byLoss != 0) return byLoss > 0;
    }
    int byProbability = Float.compare(probabilities[a], probabilities[b]);
    if (byProbability != 0) return byProbability < 0;
    return b == greedy;
  }
}
