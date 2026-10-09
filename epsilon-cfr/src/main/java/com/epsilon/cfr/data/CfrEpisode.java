package com.epsilon.cfr.data;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

/** 収集を完了した対局の経路を判断順に保持する。効用は全4席分を記録する。 */
public record CfrEpisode(
    long episodeId,
    List<CfrDecision> decisions,
    double[] utilities,
    List<CfrBranchOutcome> branchOutcomes) {
  public CfrEpisode(long episodeId, List<CfrDecision> decisions, double[] utilities) {
    this(episodeId, decisions, utilities, List.of());
  }

  public CfrEpisode {
    if (episodeId < 0 || decisions == null || utilities == null || utilities.length != 4) {
      throw new IllegalArgumentException("Invalid CFR episode");
    }
    decisions = List.copyOf(decisions);
    branchOutcomes = List.copyOf(branchOutcomes);
    utilities = utilities.clone();
    for (double utility : utilities) {
      if (!Double.isFinite(utility)) throw new IllegalArgumentException("Nonfinite utility");
    }
    long previous = -1;
    for (CfrDecision decision : decisions) {
      if (decision.decisionId() <= previous) {
        throw new IllegalArgumentException("Decision IDs must be strictly increasing");
      }
      previous = decision.decisionId();
    }
    if (!branchOutcomes.isEmpty()) {
      var byId = new HashMap<Long, CfrDecision>();
      for (CfrDecision decision : decisions) byId.put(decision.decisionId(), decision);
      var compared = new HashSet<Long>();
      for (CfrBranchOutcome branch : branchOutcomes) {
        CfrDecision decision = byId.get(branch.decisionId());
        if (decision == null
            || !compared.add(branch.decisionId())
            || branch.alternativeSlot() >= decision.actionCount()
            || branch.alternativeSlot() == decision.selectedSlot()
            || branch.mainInclusionProbability() + 1e-12 < decision.behaviorProbability()) {
          throw new IllegalArgumentException(
              "CFR branch must reference a distinct legal alternative");
        }
      }
    }
  }

  @Override
  public double[] utilities() {
    return utilities.clone();
  }

  public double utility(int seat) {
    return utilities[seat];
  }
}
