package com.epsilon.prism4p.input;

import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.RowInputWriter;
import com.epsilon.prism4p.input.InputSchema.EventField;
import com.epsilon.prism4p.input.InputSchema.KeyBlock;
import com.epsilon.prism4p.input.InputSchema.RelationAttribute;
import com.epsilon.prism4p.input.InputSchema.RoundField;
import com.epsilon.prism4p.input.InputSchema.SeatField;
import com.epsilon.prism4p.input.InputSchema.TileField;
import com.epsilon.prism4p.input.InputSchema.TransitionField;
import java.util.List;

/** 四人打ちの 1 行分のライター（フィールドの型を {@link InputSchema} の宣言にした {@link RowInputWriter}）。{@link Tokenizer} と対で使い、符号化のスレッドごとに 1 つ持つ。 */
public final class InputWriter
    extends RowInputWriter<RoundField, SeatField, TileField, EventField, TransitionField, RelationAttribute> {

  public InputWriter() {
    super(
        InputSchema.FORMAT,
        InputSchema.ROUND,
        InputSchema.SEAT,
        InputSchema.TILE,
        InputSchema.EVENT,
        InputSchema.TRANSITION,
        InputSchema.RELATION,
        List.of(KeyBlock.values()));
  }

  @Override
  public InputWriter bind(HostBatch batch, int row) {
    super.bind(batch, row);
    return this;
  }
}
