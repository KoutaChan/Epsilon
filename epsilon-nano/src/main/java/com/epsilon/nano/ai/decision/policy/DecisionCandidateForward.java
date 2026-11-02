package com.epsilon.nano.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/** 候補表現だけが使う推論演算。方策グラフや出力ヘッドには依存しない。 */
public interface DecisionCandidateForward {
  NDArray candidate(
      ParameterStore parameters,
      PairList<String, Object> runtimeParameters,
      NDArray actionFeatures,
      NDArray primaryTiles,
      NDArray transitions,
      NDArray playerContexts,
      NDArray stateContext);

  /** 合法候補だけを処理し、盤面の射影結果を対応する候補行へ集める。 */
  NDArray compactCandidate(
      ParameterStore parameters,
      PairList<String, Object> runtimeParameters,
      NDArray actionFeatures,
      NDArray primaryTiles,
      NDArray transitions,
      NDArray playerContexts,
      NDArray states,
      NDArray stateRows);

  /** 盤面単位の状態と元盤面への対応を受け取り、実在する遷移だけを射影する。 */
  NDArray transition(
      ParameterStore parameters,
      PairList<String, Object> runtimeParameters,
      NDArray features,
      NDArray tileContexts,
      NDArray discardTiles,
      NDArray discardSafety,
      NDArray states,
      NDArray stateRows);

  NDArray mixPlayerTileContexts(ParameterStore parameters, NDArray byPlayer);
}
