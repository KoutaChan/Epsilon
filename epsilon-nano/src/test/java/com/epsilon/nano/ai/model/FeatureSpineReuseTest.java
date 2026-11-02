package com.epsilon.nano.ai.model;

import ai.djl.Device;
import ai.djl.engine.Autocast;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.nano.ai.decision.input.DecisionDeviceBatch;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.input.DecisionInferenceInputs;
import com.epsilon.nano.ai.decision.input.DecisionInputLayout;
import com.epsilon.nano.ai.decision.input.DecisionNetworkInputs;
import com.epsilon.nano.ai.decision.input.DecisionPolicyInputs;
import com.epsilon.nano.ai.decision.policy.TransitionPooling;
import com.epsilon.network.NetworkRuntime;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class FeatureSpineReuseTest {
  static {
    NetworkRuntime.initialize();
  }

  @Test(groups = "native", dataProvider = "transitionBuckets")
  public void compactCandidateInputsPreserveFeaturesAndParameters(boolean multiTransition)
      throws Exception {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      var source = spine(manager);
      freeze(source);
      byte[] saved = savedParameters(source);
      var host = observation(multiTransition);
      try (var frozen = source.newFrozenInference(manager, DataType.FLOAT32)) {
        var expected = frozenOutput(manager, frozen, host, null);
        assertOutput(frozenOutput(manager, frozen, host, null, true), expected, 2e-4f);
      }
      Assert.assertEquals(savedParameters(source), saved);
    }
  }

  private static final int WIDTH = 16;

  @Test(groups = "native")
  public void frozenFeatureOwnersPreserveParametersRandomnessAndLifetime() throws Exception {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      var source = spine(manager);
      freeze(source);
      byte[] saved = savedParameters(source);
      var host = observation(true);
      FrozenOutput expected;
      try (var scope = manager.newSubManager()) {
        var batch = DecisionBatchTransfer.transferTrainingToDevice(scope, host);
        var store = new ParameterStore(scope, false);
        var state =
            source.encodeState(
                store, batch.inputs(), batch.playerMemoryPresentIndices(), false, new PairList<>());
        expected =
            new FrozenOutput(
                floats(state.stateEmbedding()),
                floats(candidates(source, store, batch, state, false, true)));
      }
      manager.getEngine().setRandomSeed(7604);
      float[] expectedRandom = manager.randomUniform(0, 1, new Shape(16)).toFloatArray();
      manager.getEngine().setRandomSeed(7604);
      try (var first = source.newFrozenInference(manager, DataType.BFLOAT16);
          var second = source.newFrozenInference(manager, DataType.FLOAT32)) {
        Assert.assertEquals(
            manager.randomUniform(0, 1, new Shape(16)).toFloatArray(), expectedRandom);
        Assert.assertEquals(first.inputNumericType(), DataType.FLOAT32);
        Assert.assertFalse(first.fusionEnabled());
        assertOutput(frozenOutput(manager, first, host, null), expected, 2e-4f);
        assertOutput(frozenOutput(manager, second, host, null), expected, 2e-4f);
        first.close();
        Assert.expectThrows(IllegalStateException.class, first::candidates);
        try (var execution = second.newExecution(1, 1)) {
          assertOutput(frozenOutput(manager, second, host, execution), expected, 2e-4f);
          assertOutput(frozenOutput(manager, second, host, execution), expected, 2e-4f);
          Assert.assertFalse(execution.hasIncompleteWork());
        }
      }
      Assert.assertEquals(
          savedParameters(source), saved, "Frozen owners must not change saved weights");
      for (var parameter : parameters(source).values()) {
        Assert.assertEquals(parameter.getArray().getDataType(), DataType.FLOAT32);
        Assert.assertFalse(parameter.requiresGradient());
      }
    }
  }

  @Test(groups = "rocm", dataProvider = "transitionBuckets")
  public void frozenFeaturesUseAndReleaseProductionStateFusions(boolean multiTransition)
      throws Exception {
    if (Engine.getEngine("PyTorch").getGpuCount() == 0)
      throw new SkipException("This test requires a ROCm GPU");
    try (NDManager manager = NDManager.newBaseManager(Device.gpu(0), "PyTorch")) {
      manager.getEngine().setRandomSeed(7628);
      var source = spine(manager, 256);
      freeze(source);
      byte[] saved = savedParameters(source);
      var host = observation(multiTransition);
      try (var frozen = source.newFrozenInference(manager, DataType.BFLOAT16);
          var execution = frozen.newExecution(1, 1)) {
        Assert.assertTrue(
            frozen.fusionEnabled(), "Production BF16 width 256 must execute fused state plans");
        Assert.assertEquals(frozen.inputNumericType(), DataType.BFLOAT16);
        var expected = frozenOutput(manager, frozen, host, null);
        FrozenOutput first = null;
        for (int repeat = 0; repeat < 3; repeat++) {
          var actual = frozenOutput(manager, frozen, host, execution);
          assertBfloat16Features(actual.state(), expected.state(), "Frozen state");
          assertBfloat16Features(actual.candidates(), expected.candidates(), "Frozen candidates");
          var compact = frozenOutput(manager, frozen, host, execution, true);
          assertBfloat16Features(compact.state(), actual.state(), "Compact state");
          assertBfloat16Features(compact.candidates(), actual.candidates(), "Compact candidates");
          if (first == null) first = actual;
          else assertOutput(actual, first, 0f);
          Assert.assertFalse(execution.hasIncompleteWork());
        }
      }
      Assert.assertEquals(savedParameters(source), saved);
    }
  }

  private record FrozenOutput(float[] state, float[] candidates) {}

  @Test(groups = "rocm", dataProvider = "transitionBuckets")
  public void frozenEagerFeaturesReusePackedTransitions(boolean multiTransition) throws Exception {
    if (Engine.getEngine("PyTorch").getGpuCount() == 0)
      throw new SkipException("This test requires a ROCm GPU");
    try (NDManager manager = NDManager.newBaseManager(Device.gpu(0), "PyTorch")) {
      manager.getEngine().setRandomSeed(7631);
      var source = spine(manager);
      freeze(source);
      byte[] saved = savedParameters(source);
      var host = observation(multiTransition);
      try (var frozen = source.newFrozenInference(manager, DataType.BFLOAT16);
          var execution = frozen.newExecution(1, 1)) {
        Assert.assertFalse(frozen.fusionEnabled());
        var expected = frozenOutput(manager, frozen, host, null);
        for (int repeat = 0; repeat < 3; repeat++) {
          var actual = frozenOutput(manager, frozen, host, execution, true);
          assertBfloat16Features(actual.state(), expected.state(), "Eager frozen state");
          assertBfloat16Features(
              actual.candidates(), expected.candidates(), "Eager frozen candidates");
          Assert.assertFalse(execution.hasIncompleteWork());
        }
      }
      Assert.assertEquals(savedParameters(source), saved);
    }
  }

  private static FrozenOutput frozenOutput(
      NDManager manager,
      EpsilonDecisionFrozenFeatures frozen,
      DecisionHostBatch host,
      DecisionInferenceExecution execution)
      throws Exception {
    return frozenOutput(manager, frozen, host, execution, false);
  }

  private static FrozenOutput frozenOutput(
      NDManager manager,
      EpsilonDecisionFrozenFeatures frozen,
      DecisionHostBatch host,
      DecisionInferenceExecution execution,
      boolean compact)
      throws Exception {
    try (var scope = manager.newSubManager();
        var inference = manager.getEngine().newInferenceMode();
        var ignored = autocast(manager)) {
      DecisionPolicyInputs inputs;
      NDArray players, transitions;
      if (compact) {
        var batch =
            DecisionBatchTransfer.transferInferenceToDevice(
                scope,
                host.sliceRows(0, host.size()),
                null,
                DecisionTensorTransfer.DIRECT_BUFFER,
                frozen.inputNumericType());
        inputs = batch.inputs();
        players = batch.playerMemoryPresentIndices();
        transitions = batch.transitionPresentIndices();
        Assert.assertTrue(
            batch.inputByteCount() < host.sliceRows(0, host.size()).inputByteCount(),
            "Compact transfer must omit padded transitions");
      } else {
        var batch =
            DecisionBatchTransfer.transferTrainingToDevice(
                scope,
                host.sliceRows(0, host.size()),
                null,
                DecisionTensorTransfer.DIRECT_BUFFER,
                frozen.inputNumericType());
        inputs = batch.inputs();
        players = batch.playerMemoryPresentIndices();
        transitions = batch.transitionPresentIndices();
      }
      var store = frozen.parameters(scope);
      var forward = execution == null ? null : execution.beginForward(scope);
      var state = frozen.encodeState(store, inputs, players, forward, new PairList<>());
      NDArray candidates =
          inputs instanceof DecisionInferenceInputs compactInputs
              ? frozen.encodeCandidates(
                  store, state, compactInputs, transitions, forward, new PairList<>())
              : frozen.encodeCandidates(
                  store,
                  state,
                  (DecisionNetworkInputs) inputs,
                  transitions,
                  forward,
                  new PairList<>());
      try (var dependency = forward == null ? null : forward.seal()) {
        return new FrozenOutput(floats(state.stateEmbedding()), floats(candidates));
      }
    }
  }

  private static void assertOutput(FrozenOutput actual, FrozenOutput expected, float tolerance) {
    Assert.assertEquals(actual.state(), expected.state(), tolerance, "Frozen state");
    Assert.assertEquals(actual.candidates(), expected.candidates(), tolerance, "Frozen candidates");
  }

  private static void assertBfloat16Features(float[] actual, float[] expected, String label) {
    Assert.assertEquals(actual.length, expected.length, label);
    double squaredError = 0, expectedSquares = 0, actualSquares = 0, dot = 0;
    double maximumError = 0, maximumExpected = 0;
    for (int index = 0; index < expected.length; index++) {
      Assert.assertTrue(Float.isFinite(actual[index]), label + " must be finite");
      double reference = expected[index], value = actual[index], difference = value - reference;
      squaredError += difference * difference;
      expectedSquares += reference * reference;
      actualSquares += value * value;
      dot += value * reference;
      maximumError = Math.max(maximumError, Math.abs(difference));
      maximumExpected = Math.max(maximumExpected, Math.abs(reference));
    }
    double relativeRms = Math.sqrt(squaredError / Math.max(expectedSquares, 1e-30));
    double cosine = dot / Math.sqrt(Math.max(expectedSquares * actualSquares, 1e-30));
    String metrics =
        label + " maxAbs=" + maximumError + " relativeRms=" + relativeRms + " cosine=" + cosine;
    // Native BF16 state plans retain BF16 player memory and change reduction order. Bound both
    // total error (two BF16 relative units) and isolated coordinates (four units of feature scale).
    // CPU/FLOAT32 cloning is checked separately with strict bounds and exact serialized weights.
    Assert.assertTrue(relativeRms <= 1d / 64, metrics);
    Assert.assertTrue(cosine >= .9998, metrics);
    Assert.assertTrue(maximumError <= Math.max(1, maximumExpected) / 32, metrics);
  }

  private static void freeze(EpsilonDecisionFeatureSpine source) {
    for (var parameter : parameters(source).values()) {
      parameter.freeze(true);
      parameter.getArray().setRequiresGradient(false);
    }
  }

  private static byte[] savedParameters(EpsilonDecisionFeatureSpine source) throws IOException {
    var bytes = new ByteArrayOutputStream();
    try (var output = new DataOutputStream(bytes)) {
      source.stateEncoder().saveParameters(output);
      source.stateReadout().saveParameters(output);
      for (var parameter : parameters(source).values()) parameter.save(output);
    }
    return bytes.toByteArray();
  }

  @Test(groups = "native")
  public void suppliedEntityIndicesPreserveStateAndGradients() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      var spine = spine(manager);
      var parameters = parameters(spine);
      var batch = DecisionBatchTransfer.transferTrainingToDevice(manager, observation(false));
      var store = new ParameterStore(manager, false);
      NDArray weights = weights(manager, new Shape(1, WIDTH));
      float[] expected;
      try (var collector = manager.getEngine().newGradientCollector()) {
        var memory =
            spine
                .stateEncoder()
                .encodeMemory(
                    store,
                    batch.inputs().stateCategories(),
                    batch.inputs().stateNumerics(),
                    batch.playerMemoryPresentIndices(),
                    true,
                    new PairList<>());
        var state = spine.readState(store, memory, true, new PairList<>());
        expected = state.stateEmbedding().toFloatArray();
        collector.backward(state.stateEmbedding().mul(weights).sum());
      }
      var expectedGradients = gradients(parameters);
      clearGradients(parameters);
      try (var collector = manager.getEngine().newGradientCollector()) {
        var state =
            spine.encodeState(
                store, batch.inputs(), batch.playerMemoryPresentIndices(), true, new PairList<>());
        Assert.assertEquals(state.stateEmbedding().toFloatArray(), expected, 2e-4f);
        collector.backward(state.stateEmbedding().mul(weights).sum());
      }
      assertGradients(parameters, expectedGradients, 2e-4f);
    }
  }

  @DataProvider
  public Object[][] transitionBuckets() {
    return new Object[][] {{false}, {true}};
  }

  @Test(groups = "native", dataProvider = "transitionBuckets")
  public void representationOnlyCandidatesPreserveOutputsAndGradients(boolean multiTransition) {
    assertCandidateParity(Device.cpu(), multiTransition);
  }

  @Test(groups = "rocm", dataProvider = "transitionBuckets")
  public void gpuRepresentationOnlyCandidatesPreserveOutputsAndGradients(boolean multiTransition) {
    if (Engine.getEngine("PyTorch").getGpuCount() == 0)
      throw new SkipException("This test requires a ROCm GPU");
    assertCandidateParity(Device.gpu(0), multiTransition);
  }

  private static void assertCandidateParity(Device device, boolean multiTransition) {
    try (NDManager manager = NDManager.newBaseManager(device, "PyTorch")) {
      manager.getEngine().setRandomSeed(3981);
      var spine = spine(manager);
      var parameters = parameters(spine);
      var store = new ParameterStore(manager, false);
      var batch =
          DecisionBatchTransfer.transferTrainingToDevice(manager, observation(multiTransition));
      EpsilonMahjongStateEncoder.EncodedState state;
      try (Autocast ignored = autocast(manager)) {
        state =
            spine.encodeState(
                store, batch.inputs(), batch.playerMemoryPresentIndices(), false, new PairList<>());
      }
      state.stateEmbedding().setRequiresGradient(true);
      NDArray weights = weights(manager, new Shape(1, batch.bucket().legalActionCapacity(), WIDTH));
      float[] expected;
      try (var collector = manager.getEngine().newGradientCollector()) {
        NDArray output;
        try (Autocast ignored = autocast(manager)) {
          output = candidates(spine, store, batch, state, true, false);
        }
        expected = floats(output);
        collector.backward(output.mul(weights).sum());
      }
      float[] expectedStateGradient = floats(state.stateEmbedding().getGradient());
      var expectedGradients = gradients(parameters);
      clearGradients(parameters);
      state.stateEmbedding().getGradient().fillI(0);
      float tolerance = device.isGpu() ? .008f : 2e-4f;
      try (var collector = manager.getEngine().newGradientCollector()) {
        NDArray output;
        try (Autocast ignored = autocast(manager)) {
          output = candidates(spine, store, batch, state, true, true);
        }
        Assert.assertEquals(floats(output), expected, tolerance);
        collector.backward(output.mul(weights).sum());
      }
      Assert.assertEquals(
          floats(state.stateEmbedding().getGradient()), expectedStateGradient, tolerance);
      assertGradients(parameters, expectedGradients, tolerance);
      double scorerGradient =
          gradients(parameters).entrySet().stream()
              .filter(entry -> entry.getKey().contains("discardCandidateScorer"))
              .mapToDouble(entry -> magnitude(entry.getValue()))
              .sum();
      if (multiTransition)
        Assert.assertTrue(scorerGradient > 1e-7, "Call pooling must retain scorer gradients");
      else Assert.assertEquals(scorerGradient, 0d, "A single transition has no pooling score");

      for (var parameter : parameters.values()) {
        parameter.freeze(true);
        parameter.getArray().setRequiresGradient(false);
      }
      state.stateEmbedding().setRequiresGradient(false);
      try (Autocast ignored = autocast(manager)) {
        float[] full = floats(candidates(spine, store, batch, state, false, false));
        float[] representation = floats(candidates(spine, store, batch, state, false, true));
        Assert.assertEquals(representation, full, tolerance);
      }
    }
  }

  private static NDArray candidates(
      EpsilonDecisionFeatureSpine spine,
      ParameterStore store,
      DecisionDeviceBatch batch,
      EpsilonMahjongStateEncoder.EncodedState state,
      boolean training,
      boolean representationOnly) {
    return representationOnly
        ? spine
            .candidates()
            .encodeRepresentation(
                store,
                state,
                batch.inputs(),
                batch.transitionPresentIndices(),
                training,
                new PairList<>())
        : spine
            .candidates()
            .encode(
                store,
                state,
                batch.inputs(),
                batch.transitionPresentIndices(),
                training,
                new PairList<>());
  }

  private static Autocast autocast(NDManager manager) {
    return manager.getDevice().isGpu()
        ? manager.getEngine().newAutocast(manager.getDevice(), DataType.BFLOAT16, true)
        : null;
  }

  private static EpsilonDecisionFeatureSpine spine(NDManager manager) {
    return spine(manager, WIDTH);
  }

  private static EpsilonDecisionFeatureSpine spine(NDManager manager, int width) {
    var spine = new EpsilonDecisionFeatureSpine(width, TransitionPooling.CANDIDATE_REPRESENTATION);
    spine
        .stateEncoder()
        .initialize(
            manager,
            DataType.FLOAT32,
            DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.STATE_CATEGORIES),
            DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.STATE_NUMERICS));
    spine.stateReadout().initialize(manager, DataType.FLOAT32, new Shape(1, width));
    spine.candidates().initialize(manager, DataType.FLOAT32);
    return spine;
  }

  private static Map<String, Parameter> parameters(EpsilonDecisionFeatureSpine spine) {
    var result = new LinkedHashMap<String, Parameter>();
    for (var entry : spine.stateEncoder().getParameters())
      result.put("state/" + entry.getKey(), entry.getValue());
    for (var entry : spine.stateReadout().getParameters())
      result.put("readout/" + entry.getKey(), entry.getValue());
    spine
        .candidates()
        .registerWith(
            (name, block) -> {
              for (var entry : block.getParameters())
                result.put(name + "/" + entry.getKey(), entry.getValue());
            });
    return result;
  }

  private static Map<String, float[]> gradients(Map<String, Parameter> parameters) {
    var result = new LinkedHashMap<String, float[]>();
    parameters.forEach(
        (name, parameter) -> {
          NDArray value = parameter.getArray();
          result.put(
              name,
              value.hasGradient() ? floats(value.getGradient()) : new float[(int) value.size()]);
        });
    return result;
  }

  private static void clearGradients(Map<String, Parameter> parameters) {
    for (var parameter : parameters.values()) {
      NDArray value = parameter.getArray();
      if (value.hasGradient()) value.getGradient().fillI(0);
    }
  }

  private static void assertGradients(
      Map<String, Parameter> parameters, Map<String, float[]> expected, float tolerance) {
    var actual = gradients(parameters);
    expected.forEach(
        (name, values) -> Assert.assertEquals(actual.get(name), values, tolerance, name));
  }

  private static NDArray weights(NDManager manager, Shape shape) {
    return manager.arange(Math.toIntExact(shape.size())).mul(.13f).sin().reshape(shape);
  }

  private static float[] floats(NDArray array) {
    return array.toType(DataType.FLOAT32, false).toFloatArray();
  }

  private static double magnitude(float[] values) {
    double magnitude = 0;
    for (float value : values) magnitude += Math.abs(value);
    return magnitude;
  }

  private static DecisionHostBatch observation(boolean multiTransition) {
    var engine = new GameEngine(770);
    GameStepResult step = engine.stepHanchan();
    for (int boundary = 0;
        boundary < 200 && !(step instanceof GameStepResult.HanchanEnded);
        boundary++) {
      if (step instanceof GameStepResult.RoundSettled) {
        step = engine.stepHanchan();
        continue;
      }
      var selections = new EngineSelectionBuffer();
      for (var point : ((GameStepResult.AwaitingDecisions) step).decisions()) {
        var bucket = DecisionBatchBuilder.selectInferenceBucket(point.legalActions());
        if ((bucket.actionTransitionCapacity() > 1) == multiTransition) {
          var builder = DecisionBatchBuilder.inference(1, bucket);
          builder.encodeObservedInferenceRow(
              0, engine.observation(point.player()), point.legalActions());
          return builder.buildEncodedInferenceRows();
        }
        selections.add(point.id(), point.legalActions().getFirst());
      }
      step = engine.commitDecisions(selections);
    }
    throw new AssertionError("Fixed fixture has no requested transition bucket");
  }
}
