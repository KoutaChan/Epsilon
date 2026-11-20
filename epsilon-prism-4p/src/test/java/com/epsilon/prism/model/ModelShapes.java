package com.epsilon.prism.model;

import java.util.List;
import org.testng.annotations.DataProvider;

/** 形・保存形式・順伝播の契約を CPU で確かめるための形と期待値。 */
public final class ModelShapes {
  private ModelShapes() {}

  /** 既定の形の手番と応答の行に対し、各計算精度で得るスコアのビット列（行優先）。 */
  public record DefaultShape(ShapeArm arm, List<String> omitted, int[] fp32, int[] bf16) {}

  @DataProvider
  public static Object[][] defaultShapes() {
    int[] v2 = {
      0xbe325a7e, 0x3f1ebd40, 0x3eb63485, 0x3f85a21a, 0x3f8bbe2c, 0x3def7c90, 0x3bf8b300,
          0x3f1ebd40,
      0, 0x3fe89bb2, 0x3fdf3867, 0x3fd26146, 0x3fa4c1e0, 0, 0, 0
    };
    int[] v2Bf16 = {
      0xbe2fb4f2, 0x3f21d626, 0x3ebb5975, 0x3f867b8f, 0x3f8cb09c, 0x3dd69f08, 0x3c56d600,
          0x3f21d626,
      0, 0x3feaa90c, 0x3fdf83cf, 0x3fd2716e, 0x3fa4e9f3, 0, 0, 0
    };
    return new Object[][] {{new DefaultShape(ShapeArm.V2, List.of(), v2, v2Bf16)}};
  }

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
