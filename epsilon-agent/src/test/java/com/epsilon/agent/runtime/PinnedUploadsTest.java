package com.epsilon.agent.runtime;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtPinnedBuffer;
import ai.djl.pytorch.engine.PtStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.Map;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * ページ固定のバッファ（{@link PinnedUploads}）。型ごとのバッファを使い回し、足りなければ大きく取り直すが、小さい写しでは縮めない。型どうしは混ざらない。ほかに、CPU での写し、
 * 誤った順の呼び出しを拒むこと、閉じても写した配列が残ることを確かめる。GPU のテストのグループ（rocm）では、写しが終わる前にバッファを書き換えないことを、バッファの末尾から書き換えて
 * 確かめる（待たずに書けば、写しがまだ読んでいない末尾の値が変わる）。写しの途中でバッファを取り直すときと閉じるときは、写した値が変わらないことだけを確かめる。ページ固定のメモリの
 * 解放はデバイスの処理が終わるのを待つので、{@link PinnedUploads} が待たなくても値は変わらない。待つこと自体は、ここでは見えない。
 */
public class PinnedUploadsTest {

  /** 写しに数ミリ秒かかる大きさ（FLOAT32 で 64 MB）。 */
  private static final int LONG_COPY = 16 << 20;

  private static final float REWRITTEN = -123.5f;

  @Test
  public void aBufferIsReusedAcrossSends() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PinnedUploads uploads = PinnedUploads.on(manager)) {
      float[] first = {1f, 2f, 3f, 4f};
      float[] second = {-5f, 6f, -7f, 8f};
      ByteBuffer[] views = new ByteBuffer[2];
      NDArray firstArray = send(uploads, manager, first, new Shape(2, 2), views, 0);
      NDArray secondArray = send(uploads, manager, second, new Shape(4), views, 1);
      assertEquals(views[0].capacity(), first.length * Float.BYTES);
      views[0].putFloat(0, REWRITTEN);
      assertEquals(views[1].getFloat(0), REWRITTEN, "the second send wrote to other memory");
      assertEquals(firstArray.getShape(), new Shape(2, 2));
      assertEquals(firstArray.toFloatArray(), first);
      assertEquals(secondArray.toFloatArray(), second);
    }
  }

  @Test
  public void aLargerSendGrowsTheBufferAndASmallerOneKeepsIt() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PinnedUploads uploads = PinnedUploads.on(manager)) {
      float[] small = {1f, 2f, 3f};
      float[] large = sequence(4096, 0.5f);
      float[] again = {9f, 8f, 7f};
      ByteBuffer[] views = new ByteBuffer[3];
      NDArray smallArray = send(uploads, manager, small, new Shape(3), views, 0);
      NDArray largeArray = send(uploads, manager, large, new Shape(64, 64), views, 1);
      NDArray againArray = send(uploads, manager, again, new Shape(3), views, 2);
      assertEquals(views[0].capacity(), small.length * Float.BYTES);
      assertEquals(views[1].capacity(), large.length * Float.BYTES);
      assertEquals(views[2].capacity(), large.length * Float.BYTES, "the buffer shrank");
      assertEquals(smallArray.toFloatArray(), small);
      assertEquals(largeArray.toFloatArray(), large);
      assertEquals(againArray.toFloatArray(), again);
      assertEquals(againArray.getShape(), new Shape(3));
    }
  }

  @Test
  public void eachDataTypeHasItsOwnBuffer() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PinnedUploads uploads = PinnedUploads.on(manager)) {
      Map<DataType, ByteBuffer> raw = new EnumMap<>(DataType.class);
      raw.put(DataType.BOOLEAN, bytes(1, 0, 0, 1, 1, 0));
      raw.put(DataType.UINT8, bytes(0, 1, 127, 128, 200, 255));
      ByteBuffer int16 = allocate(6 * Short.BYTES);
      int16.asShortBuffer().put(new short[] {Short.MIN_VALUE, -1, 0, 7, 300, Short.MAX_VALUE});
      raw.put(DataType.INT16, int16);
      ByteBuffer bfloat16 = allocate(6 * Short.BYTES);
      bfloat16.asShortBuffer().put(new short[] {0x3f80, (short) 0xc000, 0x7fc0, 0, 1, 0x4000});
      raw.put(DataType.BFLOAT16, bfloat16);
      ByteBuffer int32 = allocate(6 * Integer.BYTES);
      int32.asIntBuffer().put(new int[] {Integer.MIN_VALUE, -3, 0, 5, 123_456, Integer.MAX_VALUE});
      raw.put(DataType.INT32, int32);
      ByteBuffer int64 = allocate(6 * Long.BYTES);
      int64.asLongBuffer().put(new long[] {Long.MIN_VALUE, -1, 0, 7, 1L << 40, Long.MAX_VALUE});
      raw.put(DataType.INT64, int64);
      ByteBuffer float32 = allocate(6 * Float.BYTES);
      float32.asFloatBuffer().put(new float[] {Float.NaN, -0f, Float.MIN_VALUE, -1.5f, Float.MAX_VALUE, 3.25f});
      raw.put(DataType.FLOAT32, float32);
      Shape shape = new Shape(2, 3);
      Map<DataType, NDArray> arrays = new EnumMap<>(DataType.class);
      Map<DataType, ByteBuffer> views = new EnumMap<>(DataType.class);
      for (Map.Entry<DataType, ByteBuffer> entry : raw.entrySet()) {
        arrays.put(
            entry.getKey(),
            uploads.send(
                manager,
                entry.getKey(),
                shape,
                buffer -> {
                  views.put(entry.getKey(), buffer);
                  buffer.put(entry.getValue().duplicate());
                }));
      }
      byte marker = 1;
      for (ByteBuffer view : views.values()) {
        view.put(0, marker++);
      }
      marker = 1;
      for (ByteBuffer view : views.values()) {
        assertEquals(view.get(0), marker++, "two data types share memory");
      }
      for (Map.Entry<DataType, ByteBuffer> entry : raw.entrySet()) {
        assertContents(arrays.get(entry.getKey()), shape, entry.getKey(), entry.getValue());
      }
    }
  }

  @Test
  public void enqueueOnTheCpuCopiesEveryWrittenType() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PinnedUploads region = PinnedUploads.on(manager)) {
      short[] shorts = {-1, 2, -3};
      float[] floats = {0.25f, -8f};
      region.write(DataType.INT16, shorts.length).asShortBuffer().put(shorts);
      region.write(DataType.FLOAT32, floats.length).asFloatBuffer().put(floats);
      PinnedUploads.Transfer transfer = region.enqueue(manager, null);
      transfer.waitOnStream();
      assertEquals(transfer.array(DataType.INT16).getDataType(), DataType.INT16);
      assertEquals(transfer.array(DataType.INT16).getShape(), new Shape(3));
      assertEquals(transfer.array(DataType.INT16).toShortArray(), shorts);
      assertEquals(transfer.array(DataType.FLOAT32).toFloatArray(), floats);
      assertThrows(IllegalArgumentException.class, () -> transfer.array(DataType.INT64));
    }
  }

  @Test
  public void aTypeNotWrittenAgainIsNotCopiedAgain() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PinnedUploads region = PinnedUploads.on(manager)) {
      region.write(DataType.INT16, 2).asShortBuffer().put(new short[] {1, 2});
      region.write(DataType.FLOAT32, 2).asFloatBuffer().put(new float[] {3f, 4f});
      PinnedUploads.Transfer first = region.enqueue(manager, null);
      region.write(DataType.FLOAT32, 1).asFloatBuffer().put(5f);
      PinnedUploads.Transfer second = region.enqueue(manager, null);
      assertEquals(second.array(DataType.FLOAT32).toFloatArray(), new float[] {5f});
      assertThrows(IllegalArgumentException.class, () -> second.array(DataType.INT16));
      assertEquals(first.array(DataType.FLOAT32).toFloatArray(), new float[] {3f, 4f});
      assertEquals(first.array(DataType.INT16).toShortArray(), new short[] {1, 2});
    }
  }

  @Test
  public void theLastWriteOfATypeDecidesWhatIsCopied() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PinnedUploads region = PinnedUploads.on(manager)) {
      region.write(DataType.FLOAT32, 5).asFloatBuffer().put(new float[] {1f, 2f, 3f, 4f, 5f});
      region.write(DataType.FLOAT32, 2).asFloatBuffer().put(new float[] {6f, 7f});
      NDArray array = region.enqueue(manager, null).array(DataType.FLOAT32);
      assertEquals(array.getShape(), new Shape(2));
      assertEquals(array.toFloatArray(), new float[] {6f, 7f});
    }
  }

  @Test
  public void aSendIsRefusedWhileWrittenBuffersWaitToBeEnqueued() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PinnedUploads region = PinnedUploads.on(manager)) {
      float[] written = {1f, 2f};
      region.write(DataType.FLOAT32, written.length).asFloatBuffer().put(written);
      boolean[] called = {false};
      assertThrows(
          IllegalStateException.class,
          () -> region.send(manager, DataType.FLOAT32, new Shape(2), buffer -> called[0] = true));
      assertFalse(called[0], "the refused send touched the buffer");
      assertEquals(region.enqueue(manager, null).array(DataType.FLOAT32).toFloatArray(), written);
      float[] sent = {3f, 4f};
      NDArray array =
          region.send(manager, DataType.FLOAT32, new Shape(2), buffer -> buffer.asFloatBuffer().put(sent));
      assertEquals(array.toFloatArray(), sent);
    }
  }

  @Test
  public void enqueueIsRefusedUntilSomethingIsWritten() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PinnedUploads region = PinnedUploads.on(manager)) {
      assertThrows(IllegalStateException.class, () -> region.enqueue(manager, null));
      region.write(DataType.FLOAT32, 1).asFloatBuffer().put(1f);
      region.enqueue(manager, null);
      assertThrows(IllegalStateException.class, () -> region.enqueue(manager, null));
    }
  }

  @Test
  public void aFailedEnqueueLeavesThePlaceUsable() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PinnedUploads region = PinnedUploads.on(manager)) {
      NDManager closed = manager.newSubManager();
      closed.close();
      region.write(DataType.FLOAT32, 2).asFloatBuffer().put(new float[] {1f, 2f});
      assertThrows(IllegalStateException.class, () -> region.enqueue(closed, null));
      assertThrows(IllegalStateException.class, () -> region.enqueue(manager, null));
      region.write(DataType.INT16, 1).asShortBuffer().put((short) 9);
      PinnedUploads.Transfer transfer = region.enqueue(manager, null);
      assertEquals(transfer.array(DataType.INT16).toShortArray(), new short[] {9});
      assertThrows(IllegalArgumentException.class, () -> transfer.array(DataType.FLOAT32));
    }
  }

  @Test
  public void copiedArraysOutliveThePlace() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      float[] values = {1f, 2f};
      PinnedUploads uploads = PinnedUploads.on(manager);
      NDArray sent =
          uploads.send(manager, DataType.FLOAT32, new Shape(2), buffer -> buffer.asFloatBuffer().put(values));
      uploads.write(DataType.FLOAT32, values.length).asFloatBuffer().put(values);
      NDArray enqueued = uploads.enqueue(manager, null).array(DataType.FLOAT32);
      uploads.close();
      uploads.close();
      assertEquals(sent.toFloatArray(), values);
      assertEquals(enqueued.toFloatArray(), values);
    }
  }

  @Test
  public void closingTheOwnerFreesThePlace() {
    NDManager owner = NDManager.newBaseManager(Device.cpu());
    PinnedUploads uploads = PinnedUploads.on(owner);
    uploads.send(owner, DataType.FLOAT32, new Shape(1), buffer -> buffer.asFloatBuffer().put(1f));
    owner.close();
    uploads.close();
  }

  @Test(groups = "rocm")
  public void sendsOnTheGpuComeFromPinnedMemoryAndKeepEarlierArrays() {
    requireGpu();
    try (NDManager manager = NDManager.newBaseManager(Device.gpu(0));
        PinnedUploads uploads = PinnedUploads.on(manager)) {
      try (PtPinnedBuffer probe = ((PtNDManager) manager).allocatePinned(8, DataType.FLOAT32)) {
        assertTrue(probe.isPinned(), "the GPU build gave unpinned host memory");
      }
      float[] small = {1f, 2f, 3f};
      float[] large = sequence(1 << 16, -0.5f);
      short[] shorts = {7, -8};
      ByteBuffer[] views = new ByteBuffer[3];
      NDArray smallArray = send(uploads, manager, small, new Shape(3), views, 0);
      NDArray largeArray = send(uploads, manager, large, new Shape(256, 256), views, 1);
      NDArray againArray = send(uploads, manager, small, new Shape(3), views, 2);
      NDArray shortArray =
          uploads.send(manager, DataType.INT16, new Shape(2), buffer -> buffer.asShortBuffer().put(shorts));
      assertEquals(views[2].capacity(), large.length * Float.BYTES);
      assertEquals(smallArray.getDevice(), manager.getDevice());
      assertEquals(smallArray.toFloatArray(), small);
      assertEquals(largeArray.toFloatArray(), large);
      assertEquals(againArray.toFloatArray(), small);
      assertEquals(shortArray.toShortArray(), shorts);
    }
  }

  @Test(groups = "rocm")
  public void enqueueOnTheGpuCopiesEveryWrittenType() {
    requireGpu();
    PtStream stream = transferStream();
    try (NDManager manager = NDManager.newBaseManager(Device.gpu(0));
        PinnedUploads region = PinnedUploads.on(manager)) {
      short[] shorts = {-1, 2, -3};
      float[] floats = sequence(1 << 16, 0.25f);
      region.write(DataType.INT16, shorts.length).asShortBuffer().put(shorts);
      region.write(DataType.FLOAT32, floats.length).asFloatBuffer().put(floats);
      PinnedUploads.Transfer transfer = region.enqueue(manager, stream);
      transfer.waitOnStream();
      assertEquals(transfer.array(DataType.INT16).getDevice(), manager.getDevice());
      assertEquals(transfer.array(DataType.INT16).toShortArray(), shorts);
      assertEquals(transfer.array(DataType.FLOAT32).toFloatArray(), floats);
      assertThrows(IllegalArgumentException.class, () -> transfer.array(DataType.INT64));
    } finally {
      stream.close();
    }
  }

  @Test(groups = "rocm")
  public void aRewriteWaitsForThePendingCopy() {
    rewriteDuringCopy(LONG_COPY);
  }

  @Test(groups = "rocm")
  public void growingDuringACopyKeepsTheCopiedValues() {
    rewriteDuringCopy(2 * LONG_COPY);
  }

  @Test(groups = "rocm")
  public void aSendWaitsForThePendingCopy() {
    requireGpu();
    PtStream stream = transferStream();
    try (NDManager manager = NDManager.newBaseManager(Device.gpu(0));
        PinnedUploads region = PinnedUploads.on(manager)) {
      float[] values = sequence(LONG_COPY, 0.5f);
      region.write(DataType.FLOAT32, values.length).asFloatBuffer().put(values);
      NDArray copied = region.enqueue(manager, stream).array(DataType.FLOAT32);
      NDArray sent =
          region.send(
              manager,
              DataType.FLOAT32,
              new Shape(LONG_COPY),
              buffer -> rewriteFromTheEnd(buffer, LONG_COPY));
      assertSameFloats(copied.toFloatArray(), values);
      float[] rewritten = new float[LONG_COPY];
      Arrays.fill(rewritten, REWRITTEN);
      assertSameFloats(sent.toFloatArray(), rewritten);
    } finally {
      stream.close();
    }
  }

  @Test(groups = "rocm")
  public void closingDuringACopyKeepsTheCopiedValues() {
    requireGpu();
    PtStream stream = transferStream();
    try (NDManager manager = NDManager.newBaseManager(Device.gpu(0))) {
      float[] values = sequence(LONG_COPY, -0.25f);
      PinnedUploads region = PinnedUploads.on(manager);
      region.write(DataType.FLOAT32, values.length).asFloatBuffer().put(values);
      NDArray copied = region.enqueue(manager, stream).array(DataType.FLOAT32);
      region.close();
      assertSameFloats(copied.toFloatArray(), values);
    } finally {
      stream.close();
    }
  }

  /** 長い写しをストリームに積んだ直後に、同じ型のバッファを指定の要素数で書き直す。写し先には初めの値だけが届き、書き直した値は次の写しで届く。 */
  private static void rewriteDuringCopy(int elements) {
    requireGpu();
    PtStream stream = transferStream();
    try (NDManager manager = NDManager.newBaseManager(Device.gpu(0));
        PinnedUploads region = PinnedUploads.on(manager)) {
      float[] values = sequence(LONG_COPY, 0.75f);
      region.write(DataType.FLOAT32, values.length).asFloatBuffer().put(values);
      NDArray copied = region.enqueue(manager, stream).array(DataType.FLOAT32);
      rewriteFromTheEnd(region.write(DataType.FLOAT32, elements), elements);
      assertSameFloats(copied.toFloatArray(), values);
      PinnedUploads.Transfer rewritten = region.enqueue(manager, stream);
      rewritten.waitOnStream();
      float[] expected = new float[elements];
      Arrays.fill(expected, REWRITTEN);
      assertSameFloats(rewritten.array(DataType.FLOAT32).toFloatArray(), expected);
    } finally {
      stream.close();
    }
  }

  /** バッファの末尾の要素から順に書き換える（写しが読むのは先頭からなので、待たずに書けば必ず写し先に現れる）。 */
  private static void rewriteFromTheEnd(ByteBuffer buffer, int elements) {
    for (int index = elements - 1; index >= 0; index--) {
      buffer.putFloat(index * Float.BYTES, REWRITTEN);
    }
  }

  /** 2 つの配列が同じかを確かめる。違うときは、最初に違う要素の位置だけをメッセージに出す（全要素は出さない）。 */
  private static void assertSameFloats(float[] actual, float[] expected) {
    assertEquals(Arrays.mismatch(actual, expected), -1, "first different element");
  }

  private static NDArray send(
      PinnedUploads uploads, NDManager manager, float[] values, Shape shape, ByteBuffer[] views, int slot) {
    return uploads.send(
        manager,
        DataType.FLOAT32,
        shape,
        buffer -> {
          views[slot] = buffer;
          buffer.asFloatBuffer().put(values);
        });
  }

  private static float[] sequence(int size, float step) {
    float[] values = new float[size];
    for (int index = 0; index < size; index++) {
      values[index] = 1f + index * step;
    }
    return values;
  }

  private static ByteBuffer allocate(int bytes) {
    return ByteBuffer.allocate(bytes).order(ByteOrder.nativeOrder());
  }

  private static ByteBuffer bytes(int... values) {
    ByteBuffer buffer = allocate(values.length);
    for (int index = 0; index < values.length; index++) {
      buffer.put(index, (byte) values[index]);
    }
    return buffer;
  }

  private static void assertContents(NDArray actual, Shape shape, DataType type, ByteBuffer expected) {
    assertEquals(actual.getShape(), shape);
    assertEquals(actual.getDataType(), type);
    assertEquals(actual.toByteBuffer(), expected, type.toString());
  }

  private static void requireGpu() {
    if (Engine.getEngine(PtEngine.ENGINE_NAME).getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
  }

  /** 計算とは別の、写し用のストリーム。 */
  private static PtStream transferStream() {
    return ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(0));
  }
}
