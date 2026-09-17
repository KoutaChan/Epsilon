package com.epsilon.ai.belief;

import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.ArrayList;
import java.util.List;

/** 型付き公開局面を入力順に評価する同期Belief推論サーバー。モデルの所有権は呼び出し元にある。 */
public final class EpsilonBeliefInferenceServer<I> implements EpsilonBeliefEvaluator<I> {

  private final int maxBatch;
  private final NDManager manager;
  private final Block network;
  private final BeliefBatchFactory<I> inputs;

  /** 指定モデルと系列固有の入力転送処理を使う、同期バッチ推論サーバーを作る。 */
  public EpsilonBeliefInferenceServer(Model model, int maxBatch, BeliefBatchFactory<I> inputs) {
    if (maxBatch <= 0) {
      throw new IllegalArgumentException("maxBatch must be positive");
    }
    network = model.getBlock();
    manager = model.getNDManager();
    this.maxBatch = maxBatch;
    this.inputs = inputs;
  }

  @Override
  public synchronized EpsilonBeliefPrior evaluate(I input) {
    return evaluateBatch(List.of(input)).getFirst();
  }

  @Override
  public synchronized List<EpsilonBeliefPrior> evaluateBatch(List<I> states) {
    if (states.isEmpty()) {
      return List.of();
    }
    ArrayList<EpsilonBeliefPrior> result = new ArrayList<>(states.size());
    for (int start = 0; start < states.size(); ) {
      int end = start + 1;
      while (end < states.size()
          && end - start < maxBatch
          && inputs.sameBucket(states.get(start), states.get(end))) {
        end++;
      }
      evaluateContiguous(states.subList(start, end), result);
      start = end;
    }
    return result;
  }

  private void evaluateContiguous(List<I> states, List<EpsilonBeliefPrior> result) {
    try (NDManager sub = manager.newSubManager()) {
      NDList batch = inputs.transfer(sub, states);
      float[] flat =
          network
              .forward(new ParameterStore(sub, false), batch, false, new PairList<>())
              .singletonOrThrow()
              .toFloatArray();
      for (int offset = 0; offset < flat.length; offset += EpsilonBeliefLayout.OUTPUT_SIZE) {
        result.add(EpsilonBeliefPrior.fromFlatOutput(flat, offset));
      }
    }
  }
}
