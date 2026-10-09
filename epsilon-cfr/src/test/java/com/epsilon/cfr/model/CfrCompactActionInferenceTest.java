package com.epsilon.cfr.model;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.util.PairList;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.model.EpsilonMaskedRows;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.collection.Prediction;
import com.epsilon.cfr.data.ObservationPayload;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.nano.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.nano.ai.decision.policy.DecisionCandidateForward;
import com.epsilon.nano.ai.decision.policy.DecisionPolicyCandidatePrefixExecution;
import com.epsilon.nano.ai.decision.policy.DecisionPolicyCandidates;
import com.epsilon.nano.ai.decision.policy.EpsilonFeatureFusion;
import com.epsilon.nano.ai.decision.policy.fusion.DecisionPolicyFusionCandidatePrefixExecution;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionAutocast;
import com.epsilon.nano.ai.model.EpsilonDecisionFrozenFeatures;
import com.epsilon.nano.ai.model.EpsilonMahjongStateEncoder;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 盤面・遷移の共有を揃えた密な候補表現から、合法候補の集約だけの差を確認する。 */
public class CfrCompactActionInferenceTest {
  @Test(groups = "native")
  public void compactActionsPreserveSharedDenseScoresAndPolicies() throws Exception {
    verify(Device.cpu(), DecisionComputePrecision.FLOAT32);
  }

  @DataProvider
  public Object[][] precisions() {
    return new Object[][] {{DecisionComputePrecision.BFLOAT16}, {DecisionComputePrecision.FLOAT16}};
  }

  @Test(groups = "rocm", dataProvider = "precisions")
  public void compactActionsPreserveFrozenRoundingAndPolicies(DecisionComputePrecision precision)
      throws Exception {
    verify(Device.gpu(0), precision);
  }

  private static void verify(Device device, DecisionComputePrecision precision) throws Exception {
    var engine = Engine.getEngine("PyTorch");
    engine.setRandomSeed(19847);
    int hidden = device.isGpu() ? 256 : 16;
    var config =
        new CfrModelConfig(
            hidden,
            8,
            .01f,
            0,
            0,
            19847,
            32,
            precision,
            DecisionTensorTransfer.PINNED_BUFFER,
            EpsilonUtilityProfile.TENHOU);
    try (var owner = engine.newBaseManager(device);
        var inference = engine.newInferenceMode()) {
      var network = new CfrNetwork(config);
      network.initialize(owner, DataType.FLOAT32, new Shape(1, 1));
      var output = child(network, "output");
      var weight = output.getDirectParameters().get("weight").getArray();
      owner.randomUniform(-.25f, .25f, weight.getShape(), DataType.FLOAT32).copyTo(weight);
      output.getDirectParameters().get("bias").getArray().fillI(1);
      network.freezeParameters(true);
      Map<String, float[]> before = parameters(network);
      DataType type =
          switch (precision) {
            case FLOAT32 -> DataType.FLOAT32;
            case BFLOAT16 -> DataType.BFLOAT16;
            case FLOAT16 -> DataType.FLOAT16;
          };
      try (var frozen = network.frozenFeatures(owner, type);
          var transfer = new DecisionBatchTransfer.Workspace(owner)) {
        var first = CfrModelTest.context(877);
        var second = CfrModelTest.context(878);
        var call = CfrStagedInferenceTest.callContext();
        List<DecisionContext> contexts =
            List.of(
                first, subset(second, 3), call, subset(first, 2), second, subset(call, 1), first);
        for (boolean fused : new boolean[] {false, true})
          for (var selected :
              List.of(contexts, contexts.reversed(), List.of(call), List.of(first))) {
            var encoded = new CfrInputEncoder().encode(selected, null);
            var payloads = CfrInputEncoder.encodePayloads(encoded, null).join();
            float[] expected =
                sharedDenseReference(owner, network, frozen, encoded, precision, fused);
            try (var scope = owner.newSubManager();
                var autocast = EpsilonDecisionAutocast.open(scope, precision);
                var execution = fused ? frozen.newExecution(encoded.size(), 1) : null) {
              var forward = execution == null ? null : execution.beginForward(scope);
              var rows =
                  encoded.groups.stream()
                      .map(group -> group.host.sliceRows(0, group.host.size()))
                      .toList();
              var staged =
                  transfer.stageGroupedInference(rows, DecisionTensorTransfer.PINNED_BUFFER);
              var input =
                  transfer.enqueueGroupedInference(scope, staged, frozen.inputNumericType());
              float[] actual =
                  network
                      .inference(frozen.parameters(scope), input, frozen, forward)
                      .toType(DataType.FLOAT32, false)
                      .toFloatArray();
              if (forward != null) forward.seal().close();
              CfrStagedInferenceTest.assertGroupedScores(
                  actual, expected, precision, "compact actions fused=" + fused);
              float minimum = Float.POSITIVE_INFINITY, maximum = Float.NEGATIVE_INFINITY;
              for (float score : expected) {
                minimum = Math.min(minimum, score);
                maximum = Math.max(maximum, score);
              }
              Assert.assertTrue(maximum - minimum > .001, "Reference scores must be nonuniform");
              for (var role : CfrModel.Role.values()) {
                var wanted = predictions(expected, encoded, payloads, role);
                var found = predictions(actual, encoded, payloads, role);
                if (precision == DecisionComputePrecision.FLOAT32)
                  CfrStagedInferenceTest.assertPredictions(wanted, found, 1e-6);
                else
                  CfrStagedInferenceTest.assertReducedPrecisionPredictions(
                      wanted,
                      found,
                      precision == DecisionComputePrecision.BFLOAT16 ? 1d / 128 : 1d / 1024);
              }
            }
          }
      }
      var after = parameters(network);
      Assert.assertEquals(after.keySet(), before.keySet());
      for (String name : before.keySet())
        Assert.assertEquals(after.get(name), before.get(name), name);
    }
  }

  /**
   * 盤面・遷移は同じ全行入力で計算し、候補だけをNanoの密な経路と同じCFRヘッドで処理する。
   *
   * <p>容量区分による低精度の丸め差を混ぜず、合法位置の集約・候補射影・元の順序への復元を独立に確認する。
   * 通常Nanoが使う内部処理を参照するため、検証だけの公開APIや旧実装の複製を本番へ残さない。
   */
  private static float[] sharedDenseReference(
      NDManager owner,
      CfrNetwork network,
      EpsilonDecisionFrozenFeatures frozen,
      CfrModel.EncodedBatch encoded,
      DecisionComputePrecision precision,
      boolean fused)
      throws Exception {
    var values = new ArrayList<Float>();
    try (var scope = owner.newSubManager();
        var autocast = EpsilonDecisionAutocast.open(scope, precision);
        var execution = fused ? frozen.newExecution(encoded.size(), 1) : null;
        var transfer = new DecisionBatchTransfer.Workspace(owner)) {
      var forward = execution == null ? null : execution.beginForward(scope);
      var candidateForward = forward == null ? null : forward.candidates();
      var rows =
          encoded.groups.stream().map(group -> group.host.sliceRows(0, group.host.size())).toList();
      var staged = transfer.stageGroupedInference(rows, DecisionTensorTransfer.PINNED_BUFFER);
      var input = transfer.enqueueGroupedInference(scope, staged, frozen.inputNumericType());
      var store = frozen.parameters(scope);
      var runtime = new PairList<String, Object>();
      var state =
          frozen.encodeState(
              store,
              input.stateCategories(),
              input.stateNumerics(),
              input.playerMemoryPresentIndices(),
              forward,
              runtime);
      var encoder = frozen.candidates();
      var components = new LinkedHashMap<String, Block>();
      encoder.registerWith(components::put);
      var candidateFusion = (EpsilonFeatureFusion) components.get("candidateFusion");
      var contexts =
          ReferenceAccess.invoke(
              encoder, "encodePlayerTileContexts", store, state, candidateForward, false, runtime);
      NDArray byPlayer = ReferenceAccess.invoke(contexts, "byPlayer");
      NDArray allPlayers = ReferenceAccess.invoke(contexts, "allPlayers");
      NDArray sharedTransitions =
          ReferenceAccess.invoke(
              encoder,
              "encodeSharedTransitions",
              store,
              state,
              allPlayers,
              input,
              candidateForward,
              runtime);
      int rowOffset = 0;
      for (int index = 0; index < input.groups().size(); index++) {
        var group = input.groups().get(index);
        var host = encoded.groups.get(index).host;
        var inputs = group.inputs();
        int count = inputs.rowCount(), actions = inputs.bucket().legalActionCapacity();
        int capacity = inputs.bucket().actionTransitionCapacity();
        NDArray present = group.transitionPresentIndices();
        NDArray presentMask =
            capacity == 1
                ? null
                : EpsilonMaskedRows.scatter(
                        scope.ones(
                            new Shape(inputs.transitionCount(), 1),
                            state.stateEmbedding().getDataType()),
                        present,
                        (long) count * actions * capacity)
                    .reshape(count, actions, capacity)
                    .stopGradient();
        var transitions =
            ReferenceAccess.invoke(
                encoder,
                "poolCompactedTransitions",
                store,
                slice(sharedTransitions, group.transitionOffset(), inputs.transitionCount()),
                inputs.transitionCategories(),
                present,
                presentMask,
                (long) count,
                actions,
                capacity,
                false,
                runtime,
                false);
        var slicedState =
            new EpsilonMahjongStateEncoder.EncodedState(
                slice(state.stateEmbedding(), rowOffset, count),
                slice(state.tileEmbeddings(), rowOffset, count),
                slice(state.tileProjectionEmbeddings(), rowOffset, count),
                slice(state.playerMemory(), rowOffset, count),
                slice(state.playerMemoryMask(), rowOffset, count));
        var slicedContexts =
            ReferenceAccess.contexts(
                contexts, slice(byPlayer, rowOffset, count), slice(allPlayers, rowOffset, count));
        NDArray mask =
            inputs
                .legalActionMask()
                .toType(state.stateEmbedding().getDataType(), false)
                .stopGradient();
        var masks =
            ReferenceAccess.invoke(
                DecisionPolicyCandidates.class, "candidateMasks", inputs.actionCategories(), mask);
        // 最初の区分だけ共有計画の候補枠を使い、以降は候補射影の枠だけを個別に所有する。
        try (var candidateScope = scope.newSubManager();
            var candidateExecution =
                fused && frozen.fusionEnabled() && index > 0
                    ? new DecisionPolicyFusionCandidatePrefixExecution(
                        candidateScope, store, candidateFusion, frozen.inputNumericType(), count, 1)
                    : null) {
          var candidatePlan =
              candidateExecution == null ? null : candidateExecution.beginForward(candidateScope);
          var denseForward =
              index == 0
                  ? candidateForward
                  : candidatePlan == null ? null : ReferenceAccess.candidate(candidatePlan);
          NDArray candidates =
              ReferenceAccess.invoke(
                  encoder,
                  "encodeDenseCandidates",
                  store,
                  slicedState,
                  inputs,
                  slicedContexts,
                  transitions,
                  mask,
                  masks,
                  denseForward,
                  false,
                  runtime,
                  (long) count,
                  actions);
          var hidden =
              child(network, "candidateHidden")
                  .forward(store, new NDList(candidates), false, runtime)
                  .singletonOrThrow();
          float[] dense =
              child(network, "output")
                  .forward(
                      store, new NDList(EpsilonMahjongStateEncoder.silu(hidden)), false, runtime)
                  .singletonOrThrow()
                  .toType(DataType.FLOAT32, false)
                  .toFloatArray();
          if (candidatePlan != null) candidatePlan.seal().close();
          for (int row = 0; row < count; row++)
            for (int action = 0; action < host.legalActionCount(row); action++)
              values.add(dense[row * actions + action]);
        }
        rowOffset += count;
      }
      if (forward != null) forward.seal().close();
    }
    float[] result = new float[values.size()];
    for (int index = 0; index < result.length; index++) result[index] = values.get(index);
    return result;
  }

  private static NDArray slice(NDArray value, int from, int count) {
    return value.get("{}:{}", from, from + count);
  }

  /** 内部参照のためのreflectionを閉じ込め、構造の変更や演算の失敗はそのまま試験を失敗させる。 */
  private static final class ReferenceAccess {
    @SuppressWarnings("unchecked")
    static <T> T invoke(Object receiver, String name, Object... arguments) throws Exception {
      Class<?> type = receiver instanceof Class<?> staticType ? staticType : receiver.getClass();
      var methods =
          Arrays.stream(type.getDeclaredMethods())
              .filter(method -> method.getName().equals(name))
              .toList();
      Assert.assertEquals(methods.size(), 1, "Reference method must be unambiguous: " + name);
      var method = methods.getFirst();
      method.setAccessible(true);
      return (T) method.invoke(receiver instanceof Class<?> ? null : receiver, arguments);
    }

    static Object contexts(Object original, NDArray byPlayer, NDArray allPlayers) throws Exception {
      var constructor = original.getClass().getDeclaredConstructor(NDArray.class, NDArray.class);
      constructor.setAccessible(true);
      return constructor.newInstance(byPlayer, allPlayers);
    }

    static DecisionCandidateForward candidate(
        DecisionPolicyCandidatePrefixExecution.Forward prefix) {
      return (DecisionCandidateForward)
          Proxy.newProxyInstance(
              DecisionCandidateForward.class.getClassLoader(),
              new Class<?>[] {DecisionCandidateForward.class},
              (receiver, method, arguments) -> {
                Assert.assertEquals(
                    method.getName(), "candidate", "Reference must use dense candidates");
                return DecisionPolicyCandidatePrefixExecution.Forward.class
                    .getMethod("project", method.getParameterTypes())
                    .invoke(prefix, arguments);
              });
    }
  }

  private static List<Prediction> predictions(
      float[] scores,
      CfrModel.EncodedBatch encoded,
      List<ObservationPayload> payloads,
      CfrModel.Role role) {
    var predictions = new Prediction[encoded.size()];
    int offset = 0;
    for (var group : encoded.groups)
      for (int row = 0; row < group.host.size(); row++) {
        int count = group.host.legalActionCount(row), index = group.indices[row];
        predictions[index] =
            new Prediction(
                CfrModel.policy(Arrays.copyOfRange(scores, offset, offset + count), count, role),
                payloads.get(index));
        offset += count;
      }
    Assert.assertEquals(offset, scores.length);
    return List.of(predictions);
  }

  private static DecisionContext subset(DecisionContext source, int count) {
    return new DecisionContext(
        source.episodeId(),
        source.decisionId(),
        source.seat(),
        source.observation(),
        source.legalActions().subList(0, count),
        source.roundOrdinal());
  }

  private static Block child(Block parent, String name) {
    return parent.getChildren().stream()
        .filter(entry -> entry.getKey().endsWith(name))
        .findFirst()
        .orElseThrow()
        .getValue();
  }

  private static Map<String, float[]> parameters(Block network) {
    Map<String, float[]> values = new LinkedHashMap<>();
    for (var entry : network.getParameters())
      values.put(entry.getKey(), entry.getValue().getArray().toFloatArray());
    return values;
  }
}
