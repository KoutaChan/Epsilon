package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.Pair;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.prism.model.Batches.Batch;
import com.epsilon.prism.model.Batches.Move;
import com.epsilon.prism.model.Batches.Row;
import com.epsilon.prism4p.input.InputSchema;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 同値な打牌（同じ牌を手牌から切る手出しと、ツモ牌を切るツモ切り）では、教師がどちらの手に付いても同じ学習になることを確かめる（CPU、
 * FP32 と BF16）。手番の行は、同値な打牌の組とゲートの手（ツモか暗槓）だけを持つので、ゲートの値の土台になる最大の打牌はいつもこの組になる。
 * 同じ行を、教師を手出しの側に置いたものと、ツモ切りの側に置いたものの 2 通りに用意し、同じ初期値から数ステップ学ぶと、全ての重みが
 * ビットまで同じになる。
 */
public class TwinDiscardTrainingTest {

  /** 学ぶステップの数。 */
  private static final int STEPS = 3;

  @DataProvider
  public Object[][] precisions() {
    return new Object[][] {{Precision.FLOAT32}, {Precision.BFLOAT16}};
  }

  @Test(dataProvider = "precisions")
  public void eitherTwinCarryingTheLabelTrainsTheSameWeights(Precision precision) {
    Batch batch = Batches.build(List.of(turn(Action.Type.TSUMO_AGARI), turn(Action.Type.ANKAN)), 5);
    float nan = Float.NaN;
    List<ValueLabel> onTedashi =
        List.of(
            new ValueLabel(new float[] {0.4f, nan, 0.1f}, 0.1f, 1f),
            new ValueLabel(new float[] {0.2f, nan, -0.3f}, 0.1f, 1f));
    List<ValueLabel> onTsumogiri =
        List.of(
            new ValueLabel(new float[] {nan, 0.4f, 0.1f}, 0.1f, 1f),
            new ValueLabel(new float[] {nan, 0.2f, -0.3f}, 0.1f, 1f));
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      Map<String, float[]> initial = initialWeights(manager, batch, precision);
      Map<String, float[]> fromTedashi = train(manager, initial, batch, onTedashi, precision);
      Map<String, float[]> fromTsumogiri = train(manager, initial, batch, onTsumogiri, precision);
      List<String> moved = new ArrayList<>();
      List<String> differing = new ArrayList<>();
      float largest = 0;
      for (Map.Entry<String, float[]> entry : fromTedashi.entrySet()) {
        float[] tedashi = entry.getValue();
        float[] tsumogiri = fromTsumogiri.get(entry.getKey());
        if (!Arrays.equals(tedashi, initial.get(entry.getKey()))) {
          moved.add(entry.getKey());
        }
        if (!Arrays.equals(tsumogiri, tedashi)) {
          differing.add(entry.getKey());
          for (int i = 0; i < tedashi.length; i++) {
            largest = Math.max(largest, Math.abs(tsumogiri[i] - tedashi[i]));
          }
        }
      }
      Assert.assertFalse(moved.isEmpty(), "training must move the weights");
      Assert.assertEquals(
          differing,
          List.of(),
          precision + ": weights that differ with the label on the tsumogiri twin (largest difference " + largest + ")");
    }
  }

  /** 手番の行: 打牌の組（0 が手出し、1 がツモ切り）とゲートの手 {@code gate}。 */
  private static Row turn(Action.Type gate) {
    return new Row(
        EngineDecisionKind.TURN,
        3,
        List.of(
            new Move(Action.Type.DAHAI, 1, -1, 0),
            new Move(Action.Type.DAHAI, 0, -1, 0),
            new Move(gate, 1, -1, 2)));
  }

  /** 初期値を作る。全てのパラメーターを少し動かして値を 0 でなくし、同値な打牌の値がビットまで同じことを確かめる。 */
  private static Map<String, float[]> initialWeights(NDManager manager, Batch batch, Precision precision) {
    manager.getEngine().setRandomSeed(7);
    PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, Batches.small(PrismShape.MIN_DEPTH));
    network.initialize(manager);
    Batches.perturb(network, 0.05f, 31);
    try (NDManager scope = manager.newSubManager()) {
      float[] scores =
          network
              .forward(new ParameterStore(manager, false), batch.inputs(scope), Pass.inference(precision))
              .scores()
              .toFloatArray();
      Assert.assertTrue(Batches.maxAbs(scores) > 1e-4f, "scores must be non-trivial");
      int capacity = scores.length / batch.rows().size();
      for (int row = 0; row < batch.rows().size(); row++) {
        int tedashi = row * capacity;
        Assert.assertEquals(
            Float.floatToRawIntBits(scores[tedashi + 1]), Float.floatToRawIntBits(scores[tedashi]), "row " + row);
      }
    }
    return weights(network);
  }

  /** 初期値から、教師 {@code labels} で {@link #STEPS} ステップ学んだ重みを返す。 */
  private static Map<String, float[]> train(
      NDManager manager, Map<String, float[]> initial, Batch batch, List<ValueLabel> labels, Precision precision) {
    PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, Batches.small(PrismShape.MIN_DEPTH));
    network.initialize(manager);
    for (Pair<String, Parameter> pair : network.getParameters()) {
      pair.getValue().getArray().set(initial.get(pair.getKey()));
    }
    ParameterStore store = new ParameterStore(manager, false);
    boolean[] measured = ValueLabel.measured(labels);
    PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
    Optimizer optimizer = Batches.adam(1e-3f);
    for (int step = 0; step < STEPS; step++) {
      try (NDManager scope = manager.newSubManager();
          GradientCollector collector = manager.getEngine().newGradientCollector()) {
        Outputs outputs = network.forward(store, batch.inputs(scope), Pass.training(precision, measured, null));
        collector.backward(loss.total(scope, outputs, labels, null, 1f, labels.size(), 1f));
      }
      Batches.step(network, optimizer);
    }
    return weights(network);
  }

  private static Map<String, float[]> weights(PrismNetwork network) {
    Map<String, float[]> weights = new LinkedHashMap<>();
    for (Pair<String, Parameter> pair : network.getParameters()) {
      weights.put(pair.getKey(), pair.getValue().getArray().toFloatArray());
    }
    return weights;
  }
}
