package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.jni.JniUtils;
import com.epsilon.prism.hip.HipAttention;
import com.epsilon.prism.hip.HipLibrary;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 同じ開始重みと入力で各演算の出力、上流勾配、全パラメーターの勾配、更新後の重みと損失を照合する。 */
public class TrainingReplayTest {
  @Test(groups = "rocm")
  public void trainingMatchesTwoReferenceRuns() throws IOException, MalformedModelException {
    String path = System.getProperty("prism.training.references");
    Assert.assertNotNull(path, "set prism.training.references to two reference training traces");
    Path directory = Files.createTempDirectory("prism-training-");
    Path candidate = directory.resolve("candidate.gz");
    try {
      record(candidate);
      TrainingTensorTrace.compare(Path.of(path, "first.gz"), Path.of(path, "repeat.gz"), candidate);
    } finally {
      Files.deleteIfExists(candidate);
      Files.delete(directory);
    }
  }

  private static void record(Path path) throws IOException, MalformedModelException {
    TrainingStepRecorder.record(
        path, "", (network, stream) -> HipAttention.on(HipLibrary.load(), stream));
  }

  @Test(groups = "native")
  public void aMissingGradientCannotReuseThePreviousStep() throws Exception {
    try (var manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      var weight = (PtNDArray) manager.create(new float[] {0.5f, 2f});
      var parameter =
          Parameter.builder()
              .setName("weight")
              .setType(Parameter.Type.WEIGHT)
              .optArray(weight)
              .build();
      weight.setRequiresGradient(true);
      try (var collector = manager.getEngine().newGradientCollector()) {
        collector.backward(weight.square().sum());
      }
      try (var grad = JniUtils.getGradient(weight)) {
        Assert.assertEquals(grad.toFloatArray(), new float[] {1f, 4f});
      }
      TrainingStepRecorder.resetGradient(parameter, manager);
      Assert.assertEquals(parameter.getArray().toFloatArray(), new float[] {0.5f, 2f});
    }
  }

  @Test
  public void comparisonChecksIdentityShapeRawBitsAndMeasuredVariation() throws IOException {
    byte[] one = sample("grad", "FP32[2]", 0f, 3f);
    byte[] repeat = sample("grad", "FP32[2]", 0f, 3.0001f);
    compare(one, repeat, sample("grad", "FP32[2]", 0f, 3.00005f));
    for (byte[] invalid :
        List.of(
            sample("grad", "FP32[2]", -0f, 3f),
            sample("other", "FP32[2]", 0f, 3f),
            sample("grad", "FP32[1,2]", 0f, 3f),
            sample("grad", "FP32[2]", 0f),
            Arrays.copyOf(one, one.length + 1))) {
      Assert.expectThrows(AssertionError.class, () -> compare(one, one, invalid));
    }
    Assert.expectThrows(
        AssertionError.class, () -> compare(one, repeat, sample("grad", "FP32[2]", 0f, 3.5f)));
    Assert.expectThrows(
        EOFException.class, () -> compare(one, one, Arrays.copyOf(one, one.length - 1)));
    Assert.expectThrows(AssertionError.class, () -> sample("grad", "FP32[1]", Float.NaN));
  }

  private static byte[] sample(String name, String type, float... values) throws IOException {
    var buffer = new ByteArrayOutputStream();
    try (var data = new DataOutputStream(buffer)) {
      TrainingTensorTrace.write(data, name, type, values);
      data.writeUTF("");
    }
    return buffer.toByteArray();
  }

  private static void compare(byte[] first, byte[] repeat, byte[] candidate) throws IOException {
    TrainingTensorTrace.compare(
        new DataInputStream(new ByteArrayInputStream(first)),
        new DataInputStream(new ByteArrayInputStream(repeat)),
        new DataInputStream(new ByteArrayInputStream(candidate)));
  }
}
