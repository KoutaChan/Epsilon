package com.epsilon.prism.policy;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.Observation;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism4p.input.InputSchema;
import com.epsilon.prism4p.input.InputWriter;
import com.epsilon.prism4p.input.Tokenizer;
import java.nio.ByteBuffer;
import java.util.List;

/** 方策を通さずに、要求が運ぶ O を要求の順に 1 つのバッチにする。直に順伝播するときの入力に使い、バッチは使い回す。 */
final class RequestBatch {

  private final Observation observation = new Observation(TENHOU_4P);
  private final Tokenizer tokenizer = new Tokenizer();
  private final InputWriter writer = new InputWriter();
  private final HostInputs batch =
      HostInputs.allocate(InputSchema.FORMAT, BatchPlan.Capacity.standard(InputSchema.FORMAT), ByteBuffer::allocateDirect);

  /**
   * 要求をバッチに書く。
   *
   * @param requests O を取り込んだ要求。行 r に要求 r を書く
   * @return 確定したバッチ。次の呼び出しまで有効
   */
  HostInputs encode(List<PolicyRequest> requests) {
    batch.plan().clear();
    for (PolicyRequest request : requests) {
      assertTrue(batch.plan().add(bind(request)) >= 0);
    }
    batch.plan().seal();
    for (int row = 0; row < requests.size(); row++) {
      tokenizer.write(bind(requests.get(row)), writer.bind(batch.input(), row));
    }
    return batch;
  }

  private Observation bind(PolicyRequest request) {
    return observation.bind(request.recordWords(), request.recordBase());
  }
}
