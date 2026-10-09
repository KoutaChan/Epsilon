package com.epsilon.cfr.collection;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import java.util.List;

/** 借用した盤面観測と、その判断で選べる合法手。実局序数は方策抽選用のメタデータであり、モデル入力には含めない。 */
public record DecisionContext(
    long episodeId,
    long decisionId,
    int seat,
    PublicObservation observation,
    List<Action> legalActions,
    int roundOrdinal) {
  public DecisionContext(
      long episodeId,
      long decisionId,
      int seat,
      PublicObservation observation,
      List<Action> legalActions) {
    this(episodeId, decisionId, seat, observation, legalActions, 0);
  }

  public DecisionContext {
    legalActions = List.copyOf(legalActions);
  }
}
