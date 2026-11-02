package com.epsilon.prism.learn;

import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import org.testng.annotations.Test;

/**
 * 学習の行の選び: どちらも TRAIN のラベル行で |C| ≥ 2 の行を選ぶ。根は LABELS_AND_ROOTS だけが足し、足すのは TRAIN の根で 2 手の値がある行
 * （保留の根と d = played の根は入らない）。
 */
public class SelectionTest {
  private static RowCodec.Index index(Row.Purpose purpose, Row.Source source, int measured) {
    return new RowCodec.Index(40, purpose, Row.Group.A, source, DecisionClass.CALL, measured);
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
