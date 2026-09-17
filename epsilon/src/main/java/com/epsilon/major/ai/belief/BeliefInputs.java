package com.epsilon.major.ai.belief;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import com.epsilon.ai.belief.BeliefBatchFactory;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.major.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.ai.decision.input.DecisionStateInputs;
import java.util.List;

/** 系列固有のホスト側バッチを、共通の Belief 学習・推論処理へ同期転送する。 */
public final class BeliefInputs implements BeliefBatchFactory<DecisionHostBatch> {

  @Override
  public NDList transfer(NDManager manager, List<DecisionHostBatch> inputs) {
    DecisionHostBatch host = DecisionHostBatch.concatenate(inputs);
    DecisionStateInputs state =
        DecisionBatchTransfer.transferStateToDevice(
            manager, host.sliceRows(0, host.size()), null, DecisionTensorTransfer.DIRECT_BUFFER);
    return new NDList(state.stateCategories(), state.stateNumerics());
  }

  @Override
  public boolean sameBucket(DecisionHostBatch first, DecisionHostBatch second) {
    return first.bucket().equals(second.bucket());
  }
}
