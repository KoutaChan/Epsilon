package com.epsilon.pico.ai.decision.runtime;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.core.Action;
import com.epsilon.core.Tile;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.pico.ai.decision.input.DecisionActionRouteEncoder;
import com.epsilon.pico.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.pico.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.pico.ai.decision.input.DecisionBucket;
import com.epsilon.pico.ai.decision.input.DecisionFeatureCodec;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import com.epsilon.pico.ai.decision.input.DecisionHostInputs;
import com.epsilon.pico.ai.decision.input.DecisionInputSchema;
import com.epsilon.pico.ai.decision.input.DecisionInputWriter;
import com.epsilon.pico.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.pico.ai.network.NetworkFactory;
import com.epsilon.pico.config.settings.DecisionInferenceSettings;
import com.epsilon.pico.config.settings.EpsilonSettings;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class GpuPolicyParityTest {
  @DataProvider
  public Object[][] precisions() {
    return new Object[][] {
      {"FLOAT32", 0.0002f, "EAGER", "FUSION"},
      {"BFLOAT16", 0.02f, "EAGER", "FUSION"},
      {"FLOAT32", 0.0002f, "FUSION", "EAGER"},
      {"BFLOAT16", 0.02f, "FUSION", "EAGER"}
    };
  }

  /** 同じ保存重みをCPUとGPU fusionで評価し、全合法手の確率と価値を比較する。 */
  @Test(groups = "rocm", dataProvider = "precisions")
  public void fusedGpuPredictionsAgreeWithCpu(
      String precision, float tolerance, String policyAffine, String candidatePrefix)
      throws Exception {
    Engine.getEngine("PyTorch").setRandomSeed(391);
    Path directory = Files.createTempDirectory("epsilon-pico-gpu-parity-");
    var config =
        EpsilonSettings.of(
            Map.of(
                "epsilon.decision.inference.computePrecision",
                precision,
                "epsilon.decision.inference.maxBatch",
                "16",
                "epsilon.decision.inference.multiTransitionMaxBatch",
                "16",
                "epsilon.decision.inference.fusion.policyAffine",
                policyAffine,
                "epsilon.decision.inference.fusion.candidatePrefix",
                candidatePrefix));
    var settings = config.bind(DecisionInferenceSettings.class);
    var fusion = config.bind(DecisionInferenceFusionSettings.class);
    try {
      try (Model original =
          NetworkFactory.createDecisionModel(Device.cpu(), false, 128, EpsilonUtilityProfile.TOP)) {
        // 学習済みの分岐オフセットを再現し、加算の欠落をゼロ初期値で隠さない。
        for (var parameter : original.getBlock().getParameters().values()) {
          if (parameter.getName().equals("alternativeOffsets")) {
            float[] offsets = new float[(int) parameter.getArray().size()];
            for (int i = 0; i < offsets.length; i++) offsets[i] = (i % 17 - 8) * 0.05f;
            parameter.getArray().set(offsets);
          }
        }
        EpsilonDecisionCheckpointManager.saveInitial(original, directory);
      }
      try (Model cpu =
              EpsilonDecisionCheckpointManager.loadForInference(directory, Device.cpu(), settings);
          Model gpu =
              EpsilonDecisionCheckpointManager.loadForInference(directory, Device.gpu(), settings);
          var reference = EpsilonDecisionInferenceServer.forFrozenModel(cpu, 16, settings, fusion);
          var actual = EpsilonDecisionInferenceServer.forFrozenModel(gpu, 16, settings, fusion)) {
        for (long seed = 700; seed < 716; seed++) {
          var engine = new GameEngine(seed);
          var boundary = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
          var point = boundary.decisions().getFirst();
          var builder =
              DecisionBatchBuilder.inference(
                  1, DecisionBatchBuilder.selectInferenceBucket(point.legalActions()));
          builder.addInferenceRow(engine, point, DecisionBoundaryContext.uniform());
          var host = builder.build();
          assertPredictionsMatch(reference, actual, host, tolerance);
        }
        // 初手では現れない六つの二択の判定とMELD/KAN型分岐を、固定入力で必ず通す。
        for (DecisionHostBatch host :
            List.of(
                branchInput(
                    Action.pass(),
                    Action.chiSequence(Tile.M3, Tile.M5),
                    Action.pon(Tile.M5),
                    Action.daiminkan(Tile.M5),
                    Action.ronAgari()),
                branchInput(
                    Action.tsumoAgari(),
                    Action.kyushuKyuhai(),
                    Action.ankan(Tile.P5),
                    Action.kakan(Tile.P5),
                    Action.dahai(Tile.S1),
                    Action.riichiDahai(Tile.S1)))) {
          assertPredictionsMatch(reference, actual, host, tolerance);
        }
      }
    } finally {
      try (var paths = Files.walk(directory)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }

  private static void assertPredictionsMatch(
      EpsilonDecisionInferenceServer reference,
      EpsilonDecisionInferenceServer actual,
      DecisionHostBatch host,
      float tolerance)
      throws Exception {
    var expected = reference.evaluateBatch(host).getFirst();
    var observed = actual.evaluateBatch(host).getFirst();
    float[] want = expected.policyProbabilities(), got = observed.policyProbabilities();
    Assert.assertEquals(got.length, want.length);
    double sum = 0;
    for (int i = 0; i < want.length; i++) {
      Assert.assertTrue(Float.isFinite(got[i]) && got[i] >= 0, "finite probability");
      Assert.assertEquals(got[i], want[i], tolerance, "action=" + i);
      sum += got[i];
    }
    Assert.assertEquals(sum, 1, 0.00001);
    Assert.assertEquals(observed.valueUtility(), expected.valueUtility(), tolerance);
  }

  private static DecisionHostBatch branchInput(Action... actions) {
    DecisionBucket bucket = DecisionBucket.forInferenceCounts(actions.length, 1);
    DecisionHostInputs inputs = new DecisionHostInputs(1, bucket);
    DecisionInputWriter writer = inputs.writer(0);
    writer.round(DecisionInputSchema.RoundInt.PLAYER_SEAT, 1);
    writer.round(DecisionInputSchema.RoundInt.CURRENT_PLAYER_RELATIVE_SEAT, 1);
    writer.round(DecisionInputSchema.RoundInt.SOURCE_PLAYER_RELATIVE_SEAT, 1);
    writer.round(DecisionInputSchema.RoundInt.KYOKU_INDEX, 1);
    writer.boundaryContext(DecisionBoundaryContext.uniform());
    DecisionActionRouteEncoder.encode(List.of(actions), writer);
    for (int slot = 0; slot < actions.length; slot++) {
      Action action = actions[slot];
      writer.action(slot, DecisionInputSchema.ActionInt.ID, DecisionFeatureCodec.actionId(action));
      writer.action(
          slot, DecisionInputSchema.ActionInt.TYPE, DecisionFeatureCodec.actionType(action.type()));
      writer.action(
          slot,
          DecisionInputSchema.ActionInt.GROUP,
          DecisionFeatureCodec.actionGroup(action.type().group()));
      writer.action(
          slot,
          DecisionInputSchema.ActionInt.PRIMARY_TILE,
          action.tileType() >= 0 ? action.tileType() + 1 : DecisionInputSchema.PAD_ID);
      writer.action(
          slot, DecisionInputSchema.ActionInt.TILE_SELECTION, action.tileSelection().ordinal() + 1);
      writer.action(
          slot, DecisionInputSchema.ActionInt.DISCARD_IDENTITY, action.discardIdentityIndex() + 1);
      encodeMinimalTransition(writer, slot, 0, action);
    }
    return DecisionHostBatch.fromEncodedRow(
        bucket, inputs.denseCategories(), inputs.denseNumerics());
  }

  private static void encodeMinimalTransition(
      DecisionInputWriter writer, int actionSlot, int transitionSlot, Action action) {
    boolean discard =
        action.type() == Action.Type.DAHAI || action.type() == Action.Type.RIICHI_DAHAI;
    DecisionInputSchema.ActionTransitionKind kind =
        discard
            ? DecisionInputSchema.ActionTransitionKind.DISCARD
            : switch (action.type()) {
              case DAIMINKAN, ANKAN, KAKAN ->
                  DecisionInputSchema.ActionTransitionKind.RINSHAN_PENDING;
              case TSUMO_AGARI, RON_AGARI, KYUSHU_KYUHAI ->
                  DecisionInputSchema.ActionTransitionKind.TERMINAL;
              default -> DecisionInputSchema.ActionTransitionKind.IDENTITY;
            };
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionInt.KIND,
        kind.ordinal() + 1);
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionInt.DISCARD_CONTEXT,
        (discard
                    ? DecisionInputSchema.DiscardContext.TURN
                    : DecisionInputSchema.DiscardContext.NONE)
                .ordinal()
            + 1);
    if (discard) {
      Action canonicalDiscard =
          action.type() == Action.Type.RIICHI_DAHAI
              ? Action.dahai(action.tileType(), action.tileSelection())
              : action;
      writer.transition(
          actionSlot,
          transitionSlot,
          DecisionInputSchema.ActionTransitionInt.DISCARD_ACTION_ID,
          DecisionFeatureCodec.actionId(canonicalDiscard));
      writer.transition(
          actionSlot,
          transitionSlot,
          DecisionInputSchema.ActionTransitionInt.DISCARD_TILE,
          action.tileType() + 1);
      writer.transition(
          actionSlot,
          transitionSlot,
          DecisionInputSchema.ActionTransitionInt.TILE_SELECTION,
          action.tileSelection().ordinal() + 1);
    }
    writer.transition(
        actionSlot,
        transitionSlot,
        DecisionInputSchema.ActionTransitionInt.RESULTING_RON_FURITEN_KIND,
        DecisionInputSchema.RonFuritenKind.NONE.ordinal() + 1);
    writer.transition(
        actionSlot, transitionSlot, DecisionInputSchema.ActionTransitionInt.PRESENT, 1);
    for (int tileType = 0;
        tileType < DecisionInputSchema.ACTION_TRANSITION_TILE_COUNT;
        tileType++) {
      writer.transitionTile(
          actionSlot,
          transitionSlot,
          tileType,
          DecisionFeatureCodec.transitionTile(
              0, 0, false, false, false, discard && tileType == action.tileType(), false, false));
    }
  }
}
