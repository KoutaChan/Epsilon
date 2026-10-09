package com.epsilon.cfr.model;

import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.cfr.data.ObservationPayload;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CfrSampleMultiplicityTest {
  @Test
  public void aggregatesOnlyEqualIdentitiesAndRetainsTheirDrawCounts() {
    var first = row(1, 1);
    var equalCopy = row(1, 1);
    var differentId = row(2, 1);
    var grouped = CfrSampleMultiplicity.aggregate(List.of(first, differentId, first, equalCopy));
    Assert.assertEquals(grouped.size(), 2);
    Assert.assertSame(grouped.getFirst().row(), first);
    Assert.assertEquals(grouped.getFirst().count(), 3);
    Assert.assertSame(grouped.get(1).row(), differentId);
    Assert.assertEquals(grouped.get(1).count(), 1);
    Assert.assertEquals(grouped.stream().mapToInt(CfrSampleMultiplicity.Entry::count).sum(), 4);
  }

  @Test
  public void rejectsDifferentTargetsUnderTheSameIdentity() {
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> CfrSampleMultiplicity.aggregate(List.of(row(1, 1), row(1, -1))));
  }

  private static CfrTrainingRow row(long episode, double utility) {
    return new CfrTrainingRow(
        new CfrSampleId(1, episode, 0, 0),
        new int[] {0, 1},
        0,
        new double[] {.25, .75},
        utility,
        80,
        -40,
        ObservationPayload.copyOf(new byte[] {7}));
  }
}
