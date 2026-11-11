package com.epsilon.prism4p.input;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import java.nio.ByteBuffer;

/** テスト用のヘルパー。判断する席の観測 O を agent のライターで取り込み、{@link Tokenizer} で 1 行のバッチに書く。バッチは使い回す。 */
final class RowEncoder {

  static final BatchPlan.Capacity ONE_ROW =
      new BatchPlan.Capacity(1, Limits.MAX_EVENT_TOKENS, Limits.MAX_TRANSITIONS, Limits.MAX_LEGAL_ACTIONS);

  final ObservationWriter writer = new ObservationWriter(TENHOU_4P);
  final ObservationBatch observations = new ObservationBatch(TENHOU_4P, 1 << 12);
  final Observation read = new Observation(TENHOU_4P);
  final Tokenizer tokenizer = new Tokenizer();
  final InputWriter input = new InputWriter();
  final HostBatch batch = new HostBatch(InputSchema.FORMAT, ONE_ROW, ByteBuffer.allocate(BatchPlan.maxBytes(InputSchema.FORMAT, ONE_ROW)));

  /** 判断する席の観測 O を取り込み、リーダーに結び付けて返す。バッチには書かない。 */
  Observation capture(Engine engine, int slot) {
    Decisions decisions = engine.decisions();
    observations.clear();
    int index = writer.capture(engine.view(decisions.seat(slot)), decisions, slot, observations);
    return read.bind(observations, index);
  }

  /** 判断する席の観測 O を取り込み、1 行のバッチに書いて返す。書くときにライターの確かめも通す。 */
  Observation encode(Engine engine, int slot) {
    capture(engine, slot);
    BatchPlan plan = batch.plan();
    plan.clear();
    if (plan.add(read) != 0) {
      throw new IllegalStateException("one observation does not fit a one-row batch");
    }
    plan.seal();
    tokenizer.write(read, input.bind(batch, 0));
    return read;
  }

  /** テンソルの要素 [from, from + count) のバイト列を返す。 */
  byte[] bytes(Tensor tensor, int from, int count) {
    int row = InputSchema.FORMAT.width(tensor) * tensor.element().bytes();
    byte[] out = new byte[count * row];
    batch.blob().get(batch.plan().offset(tensor) + from * row, out);
    return out;
  }

  /** 合法手 {@code action} の、手のインデックスの列の値を返す。 */
  int column(int action, ActionColumn column) {
    int at =
        batch.plan().offset(Tensor.ACTION_COLUMN)
            + (action * InputSchema.FORMAT.width(Tensor.ACTION_COLUMN) + column.ordinal()) * Integer.BYTES;
    return batch.blob().getInt(at);
  }
}
