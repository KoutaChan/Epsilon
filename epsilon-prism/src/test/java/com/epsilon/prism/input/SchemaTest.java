package com.epsilon.prism.input;

import com.epsilon.calculate.scoring.ScoringYaku;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Condition;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Reduction;
import com.epsilon.prism.input.InputSchema.BoundedField;
import com.epsilon.prism.input.InputSchema.KeyBlock;
import com.epsilon.prism.input.InputSchema.WaitClass;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 入力・補助の頭・記録 O の schema。列と名前空間に隙間・重なりがなく語彙が u8 に入る、鍵の符号が区画を埋めて u8 に入る、候補の頭の区画が
 * 隙間なく 58 列を覆う、役の列がエンジンの役の順と同じ、条件が手で書いた表のとおり。
 */
public class SchemaTest {

  @Test
  public void columnsAndNamespacesHaveNoGapsOrOverlaps() {
    assertPartition(InputSchema.ROUND, InputSchema.RoundField.class);
    assertPartition(InputSchema.SEAT, InputSchema.SeatField.class);
    assertPartition(InputSchema.TILE, InputSchema.TileField.class);
    assertPartition(InputSchema.EVENT, InputSchema.EventField.class);
    assertPartition(InputSchema.TRANSITION, InputSchema.TransitionField.class);
    assertPartition(InputSchema.RELATION, InputSchema.RelationAttribute.class);
  }

  @Test
  public void keyCodesFillTheirBlocksWithinOneByte() {
    int key = 0;
    int code = 0;
    for (KeyBlock block : KeyBlock.values()) {
      Assert.assertEquals(block.firstKey(), key, block.name());
      Assert.assertEquals(block.firstCode(), code, block.name());
      key += block.keys();
      code += block.codes();
    }
    Assert.assertEquals(KeyBlock.KEY_COUNT, key);
    Assert.assertEquals(KeyBlock.CODE_COUNT, code);
    Assert.assertTrue(code <= InputSchema.MAX_VOCABULARY);
    Assert.assertEquals(KeyBlock.roundCode(), KeyBlock.ROUND.firstCode());
    Assert.assertEquals(KeyBlock.seatCode(0, false), KeyBlock.SEAT.firstCode());
    Assert.assertEquals(
        KeyBlock.seatCode(GameState.NUM_PLAYERS - 1, true),
        KeyBlock.SEAT.firstCode() + KeyBlock.SEAT.codes() - 1);
    Assert.assertEquals(KeyBlock.tileCode(0, false, null, false), KeyBlock.TILE.firstCode());
    Assert.assertEquals(
        KeyBlock.tileCode(Tile.TILES_PER_TYPE, true, WaitClass.HAN_6_PLUS, true),
        KeyBlock.TILE.firstCode() + KeyBlock.TILE.codes() - 1);
  }

  @Test
  public void candidateHeadsSpanFiftyEightColumns() {
    int offset = 0;
    for (CandidateTarget target : CandidateTarget.values()) {
      Assert.assertEquals(target.offset(), offset, target.name());
      offset += target.width();
    }
    Assert.assertEquals(offset, 58);
    Assert.assertEquals(CandidateTarget.WIDTH, 58);
    int reading = 0;
    for (ReadingTarget target : ReadingTarget.values()) {
      Assert.assertEquals(target.offset(), reading, target.name());
      reading += target.width();
    }
    Assert.assertEquals(ReadingTarget.WIDTH, reading);
  }

  /** 役は束ねない: 役の頭の列はエンジンの役 42 種で、列 k は役のビットの k 番目。損失は列の和にする。 */
  @Test
  public void yakuColumnsFollowTheEngine() {
    Assert.assertEquals(CandidateTarget.YAKU.width(), 42);
    Assert.assertEquals(CandidateTarget.YAKU.width(), ScoringYaku.values().length);
    for (ScoringYaku yaku : ScoringYaku.values()) {
      Assert.assertEquals(yaku.bit(), 1L << yaku.ordinal(), yaku.name());
    }
    Assert.assertEquals(CandidateTarget.YAKU.reduction(), Reduction.SUM);
    Assert.assertEquals(ReadingTarget.DANGER.reduction(), Reduction.MEAN);
  }

  @Test
  public void conditionsSelectOutcomes() {
    Assert.assertEquals(outcomes(Condition.ALWAYS), Outcome.values().length);
    Assert.assertEquals(outcomes(Condition.SELF_WIN), 2);
    Assert.assertEquals(outcomes(Condition.DEAL_IN), 1);
    Assert.assertEquals(outcomes(Condition.OTHER_WIN), 3);
    Assert.assertTrue(Condition.OTHER_WIN.holds(Outcome.DEAL_IN));
  }

  /** 列が隙間なく 1 回ずつ使われ、名前空間が表の行を隙間なく 1 回ずつ覆う。語彙は u8 に入る。 */
  private static <E extends Enum<E> & BoundedField> void assertPartition(
      TokenLayout<E> layout, Class<E> type) {
    int[] vocabularies = layout.columnVocabularies();
    boolean[] columns = new boolean[layout.width()];
    boolean[] rows = new boolean[layout.tableSize()];
    for (E field : type.getEnumConstants()) {
      FieldSpec spec = field.spec();
      Assert.assertTrue(spec.vocabulary() <= InputSchema.MAX_VOCABULARY, field.name());
      for (int index = 0; index < spec.repeats(); index++) {
        int column = layout.column(field, index);
        Assert.assertFalse(columns[column], field.name() + " overlaps column " + column);
        columns[column] = true;
        Assert.assertEquals(vocabularies[column], spec.vocabulary(), field.name());
        int offset = layout.offset(field, index);
        for (int row = offset; row < offset + spec.vocabulary(); row++) {
          Assert.assertFalse(rows[row], field.name() + " overlaps row " + row);
          rows[row] = true;
        }
      }
    }
    for (int column = 0; column < columns.length; column++) {
      Assert.assertTrue(columns[column], type.getSimpleName() + " gap at column " + column);
    }
    for (int row = 0; row < rows.length; row++) {
      Assert.assertTrue(rows[row], type.getSimpleName() + " gap at row " + row);
    }
  }

  private static int outcomes(Condition condition) {
    int count = 0;
    for (Outcome outcome : Outcome.values()) {
      if (condition.holds(outcome)) {
        count++;
      }
    }
    return count;
  }
}
