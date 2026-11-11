package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.model.FitProbe.Stage;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 区切り（{@link Context#scoped}）が途中の配列だけを閉じることを確かめる。学習でも閉じる。Attention の記録が記録した配列は、記録の
 * マネージャーへ移って残る。
 */
public class ContextTest {

  @Test
  public void inferenceClosesIntermediatesAndKeepsCarriedAndOutput() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        NDManager working = manager.newSubManager()) {
      Context context =
          new Context(new ParameterStore(manager, false), working, Precision.FLOAT32, false);
      NDArray carried = working.create(new float[] {1, 2, 3});
      NDArray[] intermediate = new NDArray[1];
      NDArray output =
          context.scoped(
              scope -> {
                intermediate[0] = carried.add(1);
                return intermediate[0].mul(2);
              },
              carried,
              carried,
              null);
      Assert.assertTrue(intermediate[0].isReleased(), "intermediate");
      Assert.assertFalse(carried.isReleased(), "carried");
      Assert.assertSame(carried.getManager(), working, "carried manager");
      Assert.assertSame(output.getManager(), working, "output manager");
      Assert.assertEquals(output.toFloatArray(), new float[] {4, 6, 8});
    }
  }

  @Test
  public void trainingClosesIntermediatesButTheTapeKeepsItsRecords() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        NDManager working = manager.newSubManager()) {
      AttentionTape tape = new AttentionTape(new ReferenceAttention());
      tape.open(working);
      Context context =
          new Context(
              new ParameterStore(manager, false), working, Precision.FLOAT32, true, tape, FitProbe.NONE);
      NDArray carried = working.create(new float[] {1, 2, 3});
      NDArray[] arrays = new NDArray[3];
      NDArray output =
          context.scoped(
              scope -> {
                arrays[0] = carried.mul(3);
                arrays[1] = carried.add(1);
                arrays[2] = scope.cut(arrays[1], Stage.O_FFN_BACKWARD, 0);
                return arrays[2].mul(2);
              },
              carried);
      Assert.assertTrue(arrays[0].isReleased(), "intermediate");
      Assert.assertFalse(arrays[1].isReleased(), "recorded source");
      Assert.assertFalse(arrays[2].isReleased(), "recorded leaf");
      Assert.assertTrue(tape.owns(arrays[1]) && tape.owns(arrays[2]), "the tape owns its records");
      Assert.assertSame(carried.getManager(), working, "carried manager");
      Assert.assertSame(output.getManager(), working, "output manager");
      Assert.assertEquals(output.toFloatArray(), new float[] {4, 6, 8});
    }
  }
}
