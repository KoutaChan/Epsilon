package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * {@link Staging} と {@link Uploads}: 送った区画が DJL のホストの配列から作る配列とバイトで同じこと（型・形・同じ型の区画の位置）、置き場を
 * 使い回しても前に送った配列が変わらないこと（足りないバッファの取り直しを含む）、誤った区画を拒むこと。
 */
public class StagingTest {

  @Test(timeOut = 120_000)
  public void sectionsMatchArraysMadeFromHostValues() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      Uploads uploads = Uploads.on(manager);
      long[] longs = {Long.MIN_VALUE, -1, 0, 7, Long.MAX_VALUE, 1L << 40};
      int[] ints = {Integer.MIN_VALUE, -3, 0, 5, Integer.MAX_VALUE, 123_456};
      float[] floats = {Float.NaN, -0f, Float.MIN_VALUE, -1.5f, Float.MAX_VALUE, 3.25f};
      boolean[] flags = {true, false, false, true, true, false};
      long[] moreLongs = {11, 12, 13, 14};
      Staging staging = new Staging();
      Staging.Section first = staging.add(longs, new Shape(2, 3));
      Staging.Section second = staging.add(ints, new Shape(6));
      Staging.Section third = staging.add(floats, new Shape(3, 2));
      Staging.Section fourth = staging.add(flags, new Shape(6, 1));
      Staging.Section fifth = staging.add(moreLongs, new Shape(1, 4));
      staging.send(uploads, manager);
      assertSameBytes(first.array(), manager.create(longs).reshape(2, 3));
      assertSameBytes(second.array(), manager.create(ints));
      assertSameBytes(third.array(), manager.create(floats).reshape(3, 2));
      assertSameBytes(fourth.array(), manager.create(flags).reshape(6, 1));
      assertSameBytes(fifth.array(), manager.create(moreLongs).reshape(1, 4));
    }
  }

  @Test(timeOut = 120_000)
  public void reusedBuffersKeepEarlierArrays() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      Uploads uploads = Uploads.on(manager);
      float[] small = {1f, 2f, 3f};
      float[] large = new float[4096];
      for (int i = 0; i < large.length; i++) {
        large[i] = -i;
      }
      float[] again = {9f, 8f, 7f};
      NDArray first = send(uploads, manager, small, new Shape(3));
      // 長い区画はバッファを取り直し、次の短い区画は取り直したバッファの先頭を書き直す
      NDArray second = send(uploads, manager, large, new Shape(64, 64));
      NDArray third = send(uploads, manager, again, new Shape(3));
      Assert.assertEquals(first.toFloatArray(), small);
      Assert.assertEquals(second.toFloatArray(), large);
      Assert.assertEquals(third.toFloatArray(), again);
    }
  }

  @Test
  public void invalidSectionsAreRejected() {
    Staging staging = new Staging();
    Staging.Section unsent = staging.add(new int[] {1}, new Shape(1));
    Assert.assertThrows(IllegalStateException.class, unsent::array);
    Assert.assertThrows(IllegalStateException.class, () -> staging.add(new long[0], new Shape(0)));
    Assert.assertThrows(
        IllegalStateException.class, () -> staging.add(new float[] {1, 2}, new Shape(3)));
    Assert.assertNull(staging.addLongs(List.of()));
  }

  private static NDArray send(Uploads uploads, NDManager manager, float[] values, Shape shape) {
    Staging staging = new Staging();
    Staging.Section section = staging.add(values, shape);
    staging.send(uploads, manager);
    return section.array();
  }

  private static void assertSameBytes(NDArray actual, NDArray expected) {
    Assert.assertEquals(actual.getShape(), expected.getShape());
    Assert.assertEquals(actual.getDataType(), expected.getDataType());
    Assert.assertEquals(actual.toByteBuffer(), expected.toByteBuffer());
  }
}
