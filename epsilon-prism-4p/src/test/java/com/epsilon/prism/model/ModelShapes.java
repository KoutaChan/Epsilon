package com.epsilon.prism.model;

import org.testng.annotations.DataProvider;

/** 形・保存形式・順伝播の契約を CPU で確かめるための形と期待値。 */
public final class ModelShapes {
  private ModelShapes() {}

  @DataProvider
  public static Object[][] eagerArms() {
    return new Object[][] {{new PrismShape.V4(512)}};
  }

  public static PrismShape.V4 bidirectional(int hidden) {
    return new PrismShape.V4(hidden, 2 * hidden, 2, PrismShape.EventAttention.BIDIRECTIONAL);
  }

  @DataProvider
  public static Object[][] savedShapes() {
    var cases = new java.util.ArrayList<>(java.util.Arrays.asList(smallShapes()));
    cases.add(new Object[] {new PrismShape.V4()});
    return cases.toArray(Object[][]::new);
  }

  @DataProvider
  public static Object[][] smallShapes() {
    return new Object[][] {
      {Batches.small(2)}, {Batches.small(3)}, {new PrismShape.V3(64, 128, 64, 128, 256, 128, true)}
    };
  }
}
