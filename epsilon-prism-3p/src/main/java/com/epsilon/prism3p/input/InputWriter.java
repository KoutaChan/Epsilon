package com.epsilon.prism3p.input;

import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.RowInputWriter;
import com.epsilon.prism3p.input.InputSchema.EventField;
import com.epsilon.prism3p.input.InputSchema.KeyBlock;
import com.epsilon.prism3p.input.InputSchema.RelationAttribute;
import com.epsilon.prism3p.input.InputSchema.RoundField;
import com.epsilon.prism3p.input.InputSchema.SeatField;
import com.epsilon.prism3p.input.InputSchema.TileField;
import com.epsilon.prism3p.input.InputSchema.TransitionField;
import java.util.List;

/**
 * 三人打ちの入力を 1 行ずつ書くライター（フィールドの型を {@link InputSchema} の宣言にした {@link RowInputWriter}）。{@link Tokenizer}
 * と組にして、符号化のスレッドごとに 1 つ持つ。
 */
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
