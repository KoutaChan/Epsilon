package com.epsilon.prism.learn;

import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import org.testng.annotations.Test;

/**
 * 学習に使う行の選び方のテスト。どちらの選び方も、学習用（TRAIN）のラベルの行のうち |C| ≥ 2 の行を選ぶ。根の行を足すのは
 * {@code LABELS_AND_ROOTS} だけで、足すのは学習用の根のうち 2 手の値がある行（ホールドアウトの根と、d = played の根は入らない）。
 */
public class SelectionTest {
  private static RowCodec.Index index(Row.Purpose purpose, Row.Source source, int measured) {
    return new RowCodec.Index(40, purpose, source, DecisionClass.CALL, measured);
  }

  @Test
  public void trainingRowsAreMeasuredTrainRows() {
    var label = index(Row.Purpose.TRAIN, Row.Source.ROLLIN, 2);
    var single = index(Row.Purpose.TRAIN, Row.Source.MAIN, 1);
    var holdout = index(Row.Purpose.HOLDOUT, Row.Source.MAIN, 3);
    var trainRoot = index(Row.Purpose.TRAIN, Row.Source.ROOT, 2);
    var sameRoot = index(Row.Purpose.TRAIN, Row.Source.ROOT, 1);
    var holdoutRoot = index(Row.Purpose.HOLDOUT, Row.Source.ROOT, 2);
    for (Selection selection : Selection.values()) {
      assertTrue(selection.test(label), selection.name());
      assertFalse(selection.test(single), selection.name());
      assertFalse(selection.test(holdout), selection.name());
      assertFalse(selection.test(sameRoot), selection.name());
      assertFalse(selection.test(holdoutRoot), selection.name());
    }
    assertFalse(Selection.LABELS.test(trainRoot));
    assertTrue(Selection.LABELS_AND_ROOTS.test(trainRoot));
  }
}
