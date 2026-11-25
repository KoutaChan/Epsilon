package com.epsilon.prism.input;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.FieldSpec.BoundedField;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputSchema.DiscardExact;
import com.epsilon.prism.input.InputSchema.EventField;
import com.epsilon.prism.input.InputSchema.RoundField;
import com.epsilon.prism.input.InputSchema.SeatField;
import com.epsilon.prism.input.InputSchema.TileField;
import com.epsilon.prism.input.InputSchema.TransitionField;
import com.epsilon.prism.input.InputSchema.WinExact;
import java.nio.ByteBuffer;

/** テスト用のヘルパー。判断する席の O を agent のライターで取り込み、{@link Tokenizer} で 1 行のバッチに書く（バッチは使い回す）。書いた行のフィールドを読むアクセサーも持つ。 */
public final class RowEncoder {

  static final BatchPlan.Capacity ONE_ROW =
      new BatchPlan.Capacity(
          1, Limits.MAX_EVENT_TOKENS, Limits.MAX_TRANSITIONS, Limits.MAX_LEGAL_ACTIONS);

  final InputFormat format;
  private final ObservationWriter writer;
  private final ObservationBatch observations;
  private final Observation read;
  public final InputFormat.RowWriter tokenizer;
  public final HostBatch batch;
  private final HostInputs host;

  public RowEncoder(InputFormat format) {
    this.format = format;
    var variant = format.variant();
    writer = new ObservationWriter(variant);
    observations = new ObservationBatch(variant, 1 << 12);
    read = new Observation(variant);
    tokenizer = format.writer();
    host = HostInputs.allocate(format, ONE_ROW, ByteBuffer::allocate);
    batch = host.input();
  }

  /** 判断する席の O を取り込み、読み取れる形にして返す（バッチには書かない）。 */
  public Observation capture(Engine engine, int slot) {
    Decisions decisions = engine.decisions();
    observations.clear();
    int index = writer.capture(engine.view(decisions.seat(slot)), decisions, slot, observations);
    return read.bind(observations, index);
  }

  /** 判断する席の O を取り込み、1 行のバッチに書く（ライターの確かめを通す）。 */
  public Observation encode(Engine engine, int slot) {
    capture(engine, slot);
    BatchPlan plan = batch.plan();
    plan.clear();
    if (plan.add(read) != 0) {
      throw new IllegalStateException("one observation does not fit a one-row batch");
    }
    plan.seal();
    tokenizer.write(read, host, 0);
    return read;
  }

  /** テンソルの要素 [from, from + count) のバイトを返す。 */
  byte[] bytes(Tensor tensor, int from, int count) {
    int row = format.width(tensor) * tensor.element().bytes();
    byte[] out = new byte[count * row];
    batch.blob().get(batch.plan().offset(tensor) + from * row, out);
    return out;
  }

  /** 手ごとの列の値。 */
  int column(int action, ActionColumn column) {
    int at =
        batch.plan().offset(Tensor.ACTION_COLUMN)
            + (action * format.width(Tensor.ACTION_COLUMN) + column.ordinal()) * Integer.BYTES;
    return batch.blob().getInt(at);
  }

  private int category(Tensor tensor, int element, int column) {
    int at = batch.plan().offset(tensor) + element * format.width(tensor) + column;
    return batch.blob().get(at) & 0xFF;
  }

  /** 局のトークンのフィールドの値。 */
  int round(RoundField field) {
    return category(Tensor.ROUND_CATEGORY, 0, format.round().column(field));
  }

  /** 席のトークンのフィールドの値。 */
  int seat(int relativeSeat, SeatField field) {
    return category(Tensor.SEAT_CATEGORY, relativeSeat, format.seat().column(field));
  }

  /** 牌のトークンのフィールドの値（繰り返さないフィールド）。 */
  int tile(int token, TileField field) {
    return category(Tensor.TILE_CATEGORY, token, format.tile().column(field));
  }

  /** 牌のトークンのフィールドの値（繰り返すフィールド）。 */
  int tile(int token, TileField field, int repeat) {
    return category(Tensor.TILE_CATEGORY, token, format.tile().column(field, repeat));
  }

  /** イベントのトークンのフィールドの値。 */
  int event(int position, EventField field) {
    return category(Tensor.EVENT_CATEGORY, position, format.event().column(field));
  }

  /** 遷移のフィールドの値。 */
  int transition(int transition, TransitionField field) {
    return category(Tensor.TRANSITION_CATEGORY, transition, format.transition().column(field));
  }

  /** B に直接渡す遷移ごとの値（正規化の後）。 */
  float discardExact(int transition, DiscardExact field) {
    int at =
        batch.plan().offset(Tensor.DISCARD_EXACT)
            + (transition * format.width(Tensor.DISCARD_EXACT) + field.ordinal()) * Float.BYTES;
    return batch.blob().getFloat(at);
  }

  /** ロンとツモのゲートに直接渡す手ごとの値（正規化の後）。 */
  float winExact(int action, WinExact field) {
    int at =
        batch.plan().offset(Tensor.ACTION_EXACT)
            + (action * format.width(Tensor.ACTION_EXACT) + format.winFields().index(field))
                * Float.BYTES;
    return batch.blob().getFloat(at);
  }

  /** enum のフィールドに書かれる値（語彙の中の位置 + 1）。 */
  int id(BoundedField field, Enum<?> value) {
    return Ids.of(field.spec(format.variant()).values(), value);
  }
}
