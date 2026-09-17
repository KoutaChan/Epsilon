package com.epsilon.nano.ai.decision.training;

import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import java.io.IOException;

/** 変換済みの事前学習データを、学習器へバッチ単位で順に供給する。 */
interface EpsilonDecisionPretrainBatchCursor extends AutoCloseable {

  DecisionHostBatch next() throws IOException;

  /** 未確定なら-1を返す。 */
  int totalBatches();

  /** 未確定なら-1を返す。 */
  long totalRows();

  @Override
  void close();
}
