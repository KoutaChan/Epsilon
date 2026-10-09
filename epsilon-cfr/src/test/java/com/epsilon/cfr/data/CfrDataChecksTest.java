package com.epsilon.cfr.data;

import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class CfrDataChecksTest {
  private static final ObservationPayload EMPTY = ObservationPayload.copyOf(new byte[0]);

  @DataProvider
  public Object[][] actionCounts() {
    return new Object[][] {{1}, {16}, {64}, {65}, {4096}};
  }

  @Test(dataProvider = "actionCounts")
  public void everySizePreservesUniqueNonnegativeActionIdsAndRejectsMalformedRows(int count) {
    int[] actions = new int[count];
    for (int i = 0; i < count; i++) actions[i] = Integer.MAX_VALUE - i;
    double[] policy = new double[count];
    Arrays.fill(policy, 1d / count);
    CfrDataChecks.decision(actions, count - 1, policy, EMPTY);

    actions[count - 1] = -1;
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> CfrDataChecks.decision(actions, count - 1, policy, EMPTY));
    actions[count - 1] = Integer.MAX_VALUE - count + 1;
    if (count > 1) {
      actions[count - 1] = actions[0];
      Assert.expectThrows(
          IllegalArgumentException.class,
          () -> CfrDataChecks.decision(actions, count - 1, policy, EMPTY));
      actions[count - 1] = Integer.MAX_VALUE - count + 1;
    }
    policy[count - 1] = Double.NaN;
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> CfrDataChecks.decision(actions, count - 1, policy, EMPTY));
  }
}
