package com.epsilon.prism.policy;

import static org.testng.Assert.assertTrue;

import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.FrontierFeatures;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputWriter;
import com.epsilon.prism.input.Tokenizer;
import com.epsilon.prism.observe.Observation;
import java.util.List;

/** 方策を通さずに、問い合わせが運ぶ O を問い合わせの順の 1 つの塊にする（直の順伝播の材料。塊は使い回す）。 */
final class RequestBatch {

  private final Observation observation = new Observation();
  private final Tokenizer tokenizer = new Tokenizer();
  private final InputWriter writer = new InputWriter();
  private final FrontierFeatures frontier;
  private final HostInputs batch;

  /**
   * @param frontier 前線の腕の網なら true（前線の塊も書く）
   */
  RequestBatch(boolean frontier) {
    this.frontier = frontier ? new FrontierFeatures() : null;
    batch = HostInputs.allocate(BatchPlan.Capacity.DEFAULT, frontier);
  }

  /**
   * 問い合わせを書いた塊（次の呼び出しで書き換わる）。
   *
   * @param requests O を捕捉した問い合わせ（行 r は問い合わせ r）
   * @return 封じた塊
   */
  HostInputs encode(List<PolicyRequest> requests) {
    batch.plan().clear();
    for (PolicyRequest request : requests) {
      assertTrue(batch.plan().add(bind(request)) >= 0);
    }
    batch.plan().seal();
    for (int row = 0; row < requests.size(); row++) {
      Observation o = bind(requests.get(row));
      tokenizer.write(o, writer.bind(batch.input(), row));
      if (frontier != null) {
        frontier.write(o, batch.frontier(), row);
      }
    }
    return batch;
  }

  private Observation bind(PolicyRequest request) {
    return observation.bind(request.recordWords(), request.recordBase());
  }
}
