package com.epsilon.classic.model;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.input.Buckets;
import com.epsilon.classic.input.Buckets.Bucket;
import com.epsilon.classic.input.ClassicInputs;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.input.Tensor;
import java.nio.ByteBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * ページ固定のバッファを通して送った行のバッチが、Java の配列からそのままデバイスに作った配列と、型・形・バイトで同じになることを確かめる。
 * どの tensor も、先頭から使う行の数までを比べる。同じバッファを、形と区分の違う入力で使い回しても、前のバッチは変わらない。GPU の
 * グループ（rocm）では、同期の写しと非同期の写しの両方を確かめる。
 */
public class ClassicBatchUploadTest {

  private static final int ROWS = 3;
  private static final int[] USED_ROWS = {1, 2, ROWS};
  private static final Bucket SMALLEST = Buckets.of(0);
  private static final Bucket LARGEST = Buckets.required(ClassicSchema.MAX_LEGAL, ClassicSchema.MAX_TRANSITIONS);

  /** 確かめる前の値の控え（バイトで持つので、入力を書き直しても変わらない）。 */
  private record Expected(Tensor tensor, Shape shape, DataType type, ByteBuffer bytes) {}

  private record Sent(ClassicBatch batch, List<Expected> expected) {}

  @Test
  public void sendMatchesThePlainCopy() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PinnedUploads uploads = PinnedUploads.on(manager)) {
      List<Sent> sent = new ArrayList<>();
      for (Architecture architecture : Architecture.values()) {
        for (Bucket bucket : List.of(SMALLEST, LARGEST, SMALLEST)) {
          ClassicInputs inputs = filled(architecture, bucket, 17L * sent.size() + 3);
          for (int rows : USED_ROWS) {
            sent.add(new Sent(ClassicBatch.send(manager, uploads, inputs, rows), plainCopy(manager, inputs, rows)));
          }
        }
      }
      sent.forEach(ClassicBatchUploadTest::assertMatches);
    }
  }

  @Test
  public void rowsOutsideTheInputsAreRefused() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PinnedUploads uploads = PinnedUploads.on(manager)) {
      ClassicInputs inputs = filled(Architecture.NANO, SMALLEST, 5);
      assertThrows(IllegalArgumentException.class, () -> ClassicBatch.send(manager, uploads, inputs, 0));
      assertThrows(IllegalArgumentException.class, () -> ClassicBatch.send(manager, uploads, inputs, ROWS + 1));
    }
  }

  @Test
  public void aTensorOfTheOtherArchitectureIsRefused() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PinnedUploads uploads = PinnedUploads.on(manager)) {
      ClassicBatch nano = ClassicBatch.send(manager, uploads, filled(Architecture.NANO, SMALLEST, 7), 1);
      assertThrows(IllegalArgumentException.class, () -> nano.tensor(Tensor.POINT_LEDGER_100));
    }
  }

  @Test(groups = "rocm")
  public void bothCopiesMatchThePlainCopyOnTheGpu() {
    if (Engine.getEngine(PtEngine.ENGINE_NAME).getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    Device gpu = Device.gpu(0);
    PtStream stream = ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(gpu);
    try (NDManager manager = NDManager.newBaseManager(gpu);
        PinnedUploads uploads = PinnedUploads.on(manager);
        PinnedUploads region = PinnedUploads.on(manager)) {
      List<Sent> sent = new ArrayList<>();
      for (Architecture architecture : Architecture.values()) {
        for (Bucket bucket : List.of(SMALLEST, LARGEST, SMALLEST)) {
          ClassicInputs inputs = filled(architecture, bucket, 19L * sent.size() + 11);
          for (int rows : USED_ROWS) {
            sent.add(new Sent(ClassicBatch.send(manager, uploads, inputs, rows), plainCopy(manager, inputs, rows)));
            ClassicBatch.write(region, inputs);
            ClassicBatch batch = ClassicBatch.enqueue(manager, region, inputs, rows, stream).receive();
            sent.add(new Sent(batch, plainCopy(manager, inputs, rows)));
          }
        }
      }
      sent.forEach(ClassicBatchUploadTest::assertMatches);
    } finally {
      stream.close();
    }
  }

  /** 全ての要素が違う値になるよう、乱数で埋めた入力。 */
  private static ClassicInputs filled(Architecture architecture, Bucket bucket, long seed) {
    ClassicInputs inputs = new ClassicInputs(ClassicSchema.of(architecture), bucket, ROWS);
    Random random = new Random(seed);
    short[] categorical = inputs.categorical();
    for (int index = 0; index < categorical.length; index++) {
      categorical[index] = (short) random.nextInt();
    }
    float[] numeric = inputs.numeric();
    for (int index = 0; index < numeric.length; index++) {
      numeric[index] = (float) random.nextGaussian();
    }
    return inputs;
  }

  /** 入力の形が持つ tensor ごとに、Java の配列の区間からそのままデバイスに配列を作り、その型・形・バイトを控える。 */
  private static List<Expected> plainCopy(NDManager manager, ClassicInputs inputs, int rows) {
    ClassicSchema schema = inputs.schema();
    Bucket bucket = inputs.bucket();
    List<Expected> expected = new ArrayList<>();
    for (Tensor tensor : Tensor.values()) {
      if (!tensor.in(schema.architecture())) {
        continue;
      }
      int start = inputs.start(tensor, 0);
      int end = start + rows * inputs.rowWidth(tensor);
      int width = schema.width(tensor);
      Shape shape =
          switch (tensor.axis()) {
            case ROW -> new Shape(rows, width);
            case ACTION -> new Shape(rows, bucket.actions(), width);
            case TRANSITION -> new Shape(rows, bucket.actions(), bucket.transitions(), width);
          };
      NDArray plain =
          tensor.slab() == Tensor.Slab.CATEGORICAL
              ? manager.create(ShortBuffer.wrap(Arrays.copyOfRange(inputs.categorical(), start, end)), shape, DataType.INT16)
              : manager.create(Arrays.copyOfRange(inputs.numeric(), start, end), shape);
      expected.add(new Expected(tensor, plain.getShape(), plain.getDataType(), plain.toByteBuffer()));
    }
    return expected;
  }

  private static void assertMatches(Sent sent) {
    for (Expected expected : sent.expected()) {
      NDArray actual = sent.batch().tensor(expected.tensor());
      assertEquals(actual.getShape(), expected.shape(), expected.tensor().name());
      assertEquals(actual.getDataType(), expected.type(), expected.tensor().name());
      assertEquals(actual.toByteBuffer(), expected.bytes(), expected.tensor().name());
    }
  }
}
