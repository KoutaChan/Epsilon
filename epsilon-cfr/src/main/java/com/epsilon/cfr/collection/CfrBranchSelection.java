package com.epsilon.cfr.collection;

import com.epsilon.ai.decision.DecisionBranchGate;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.core.Action;
import java.util.List;

/** 行動選択前に終局分岐の計画を立て、各行動が収集対象に含まれる周辺確率を求める。 */
public final class CfrBranchSelection {
  private CfrBranchSelection() {}

  /** 代替行動がない場合は、スロットを-1、収集対象に含まれる確率を0とする。 */
  public record Choice(
      double mainInclusionProbability,
      int alternativeSlot,
      double alternativeInclusionProbability) {}

  /** 本線の行動を抽選する前に採用する分岐計画。収集対象に含まれる確率は、実際の抽選結果を条件とせず、本線の抽選と、それに応じた継続行動の抽選を合わせて求める。 */
  public record Plan(
      DecisionBranchGate gate,
      int acceptanceSlot,
      int excludedSlot,
      double declineMass,
      double gateMass,
      double[] behavior) {
    public Plan {
      behavior = behavior.clone();
    }

    @Override
    public double[] behavior() {
      return behavior.clone();
    }

    public Choice afterSelected(int slot, double uniform) {
      if (slot == excludedSlot) return new Choice(behavior[slot], -1, 0);
      if (gate == DecisionBranchGate.RON) {
        return slot == acceptanceSlot
            ? new Choice(1, -1, 0)
            : new Choice(behavior[slot], acceptanceSlot, 1);
      }
      if (slot != acceptanceSlot)
        return new Choice(continuationInclusion(slot), acceptanceSlot, gateMass);
      double threshold = uniform * declineMass;
      int alternative = -1;
      for (int candidate = 0; candidate < behavior.length; candidate++) {
        if (candidate == acceptanceSlot || candidate == excludedSlot || behavior[candidate] == 0)
          continue;
        alternative = candidate;
        threshold -= behavior[candidate];
        if (threshold < 0) break;
      }
      return new Choice(gateMass, alternative, continuationInclusion(alternative));
    }

    private double continuationInclusion(int slot) {
      return Math.min(1, Math.max(behavior[slot], behavior[slot] * gateMass / declineMass));
    }
  }

  /** 有効な終局分岐のいずれにも正の選択確率を持つ継続行動がなければ、{@code null}を返す。 */
  public static Plan plan(
      List<Action> actions,
      double[] targetPolicy,
      double epsilon,
      DecisionBranchComparisonSettings settings) {
    if (!settings.enabled()) return null;
    for (DecisionBranchGate gate : DecisionBranchGate.PRIORITY) {
      if (gate == DecisionBranchGate.TSUMO) continue;
      if (gate == DecisionBranchGate.KYUSHU ? !settings.kyushu() : !settings.winDecline()) continue;
      int accepted = -1;
      int excluded = -1;
      for (int slot = 0; slot < actions.size(); slot++) {
        Action action = actions.get(slot);
        if (gate.accepts(action)) accepted = slot;
        else if (!gate.contains(action)) excluded = slot;
      }
      if (accepted < 0) continue;
      double declineMass = 0;
      double[] behavior = new double[actions.size()];
      for (int slot = 0; slot < actions.size(); slot++) {
        behavior[slot] = (1 - epsilon) * targetPolicy[slot] + epsilon / actions.size();
        if (slot != accepted && slot != excluded) declineMass += behavior[slot];
      }
      if (declineMass > 0) {
        return new Plan(
            gate,
            accepted,
            excluded,
            declineMass,
            Math.min(1, declineMass + behavior[accepted]),
            behavior);
      }
    }
    return null;
  }
}
