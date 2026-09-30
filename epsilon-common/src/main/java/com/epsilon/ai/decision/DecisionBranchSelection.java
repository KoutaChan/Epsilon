package com.epsilon.ai.decision;

import com.epsilon.core.Action;
import java.util.List;

/** 探索を含む最終分布を変更せず、終了ゲートの探索成分への帰属だけを求める。 */
public final class DecisionBranchSelection {
  private DecisionBranchSelection() {}

  /** 対象ゲートに属する行動で条件付けした宣言確率を返す。 */
  public static float acceptanceProbability(
      DecisionBranchGate gate, List<Action> actions, float[] policy) {
    double accepted = 0;
    double total = 0;
    for (int i = 0; i < actions.size(); i++) {
      if (gate.contains(actions.get(i))) {
        total += policy[i];
      }
      if (gate.accepts(actions.get(i))) {
        accepted += policy[i];
      }
    }
    return total == 0 ? 0 : (float) (accepted / total);
  }

  /** RON/TSUMOのdeclineを選んだ条件下で、leaf floorを除いた探索成分の事後確率。 */
  public static double explorationPosterior(
      float acceptance,
      float selectedBehavior,
      float explorationMass,
      float minimumLeafProbability) {
    double declined = (1.0 - explorationMass) * (1.0 - acceptance) + explorationMass * 0.5;
    if (declined == 0 || selectedBehavior <= minimumLeafProbability) {
      return 0;
    }
    return Math.min(
        1,
        (selectedBehavior - minimumLeafProbability)
            / selectedBehavior
            * (explorationMass * 0.5)
            / declined);
  }

  /** 宣言を選んだ主枝に対する続行手。下位の探索分布を条件付けして抽選する。 */
  public static int sampleDecline(
      DecisionBranchGate gate, List<Action> actions, float[] behavior, double uniform) {
    double mass = 0;
    int last = -1;
    for (int i = 0; i < actions.size(); i++) {
      if (gate.contains(actions.get(i)) && !gate.accepts(actions.get(i))) {
        mass += behavior[i];
      }
    }
    double threshold = uniform * mass;
    for (int i = 0; i < actions.size(); i++) {
      if (!gate.contains(actions.get(i)) || gate.accepts(actions.get(i))) {
        continue;
      }
      last = i;
      threshold -= behavior[i];
      if (threshold < 0) {
        return i;
      }
    }
    return last;
  }
}
