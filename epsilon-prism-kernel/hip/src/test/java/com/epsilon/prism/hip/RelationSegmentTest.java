package com.epsilon.prism.hip;

import org.testng.Assert;
import org.testng.annotations.Test;

public class RelationSegmentTest {
  @Test
  public void segmentsOwnSixteenRowsInPackedCodeOrder() {
    RelationSegment[] order = {
      RelationSegment.TYPE_PAIR,
      RelationSegment.TILE_FIRST,
      RelationSegment.TILE_SECOND,
      RelationSegment.SEAT,
      RelationSegment.RIICHI,
      RelationSegment.TIME,
      RelationSegment.LINK_SAFETY
    };
    for (int i = 0; i < order.length; i++) {
      Assert.assertEquals(order[i].firstRow(), 16 * i);
    }
  }
}
