package com.epsilon.prism.model;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.backend.HipFixtures;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 区分を持つ候補層が、学習のテープを持たない推論で、推論用の Attention と通常の Attention を切り替えられることを確かめる。 */
public class CandidateInferenceTapeTest {
  /** 候補の区分を持つ形のアーム。 */
  @DataProvider
  public static Object[][] segmentArms() {
    return java.util.Arrays.stream(ShapeArm.values())
        .filter(arm -> arm.shape() instanceof PrismShape.V4)
        .map(arm -> new Object[] {arm})
        .toArray(Object[][]::new);
  }

  @Test(dataProvider = "segmentArms")
  public void fusedInferenceWithoutTapeCanFallBackToStockAttention(ShapeArm arm) {
    var shape = arm.shape();
    var format = shape.inputFormat(FOUR_PLAYER);
    var segments = new CandidateSegments(shape, format);
    // 通常の Attention を通した後、同じ重みで推論用の Attention に切り替える。
    class Features implements CandidateLayer.Features {
      boolean requestFused;
      int calls;

      @Override
      public boolean fusedAttention(Context context) {
        return requestFused;
      }

      @Override
      public NDArray attention(
          Context context,
          NetworkInputs inputs,
          BatchIndex index,
          NDArray query,
          NDArray memory,
          NDArray base) {
        calls++;
        Assert.assertNull(context.tape());
        return segments.attention(context, inputs, index, query, memory, base);
      }

      @Override
      public CandidateLayer.Offsets offsets(Context context, NetworkInputs inputs, NDArray base) {
        return segments.offsets(context, inputs, base);
      }
    }
    var features = new Features();
    try (var model = NDManager.newBaseManager(Device.cpu());
        var scope = model.newSubManager()) {
      var layer = new CandidateLayer(shape, format, features);
      layer.initialize(model, DataType.FLOAT32);
      segments.initialize(model, DataType.FLOAT32);
      var batch = HipFixtures.encode(HipFixtures.mixedRows(2, 57), shape);
      var inputs = NetworkInputs.of(scope, PinnedUploads.on(scope), batch);
      var index = BatchIndex.send(inputs, Pass.inference(Precision.FLOAT32));
      var states = scope.ones(new Shape(inputs.rows(), format.fixedTokens(), shape.hidden()));
      var context = new Context(new ParameterStore(model, false), scope, Precision.FLOAT32, false);
      float[] expected = layer.forward(context, inputs, index, states).toFloatArray();
      features.requestFused = true;
      float[] actual = layer.forward(context, inputs, index, states).toFloatArray();
      Assert.assertEquals(features.calls, 1);
      Assert.assertEquals(actual, expected);
    }
  }
}
