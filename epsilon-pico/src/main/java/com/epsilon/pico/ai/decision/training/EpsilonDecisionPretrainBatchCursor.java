package com.epsilon.pico.ai.decision.training;

import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import java.io.IOException;

/** 事前学習用に変換されたバッチを、学習処理へ順次供給する。 */
interface EpsilonDecisionPretrainBatchCursor extends AutoCloseable {

  DecisionHostBatch next() throws IOException;

  /** 未確定なら-1を返す。 */
  int totalBatches();

  /** 未確定なら-1を返す。 */
  long totalRows();

  @Override
  void close();
}
