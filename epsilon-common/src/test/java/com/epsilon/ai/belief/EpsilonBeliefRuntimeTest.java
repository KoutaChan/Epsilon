package com.epsilon.ai.belief;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.core.Tile;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** Beliefの不均等なバッチの損失集計、評価時のパラメーター不変性、推論順序とモデルの所有権を検証する。 */
public class EpsilonBeliefRuntimeTest {

  @Test(groups = "native")
  public void unevenBatchesRetainMeanLossAndEvaluationDoesNotUpdateParameters() {
    TrackingInputs inputs = new TrackingInputs();
    TinyBlock block = new TinyBlock();
    try (Model model = model(block)) {
      try (var trainer = new EpsilonBeliefTrainer<>(model, inputs, 2, .01f, 0f, 1f)) {
        var samples = List.of(sample(1f), sample(1f), sample(3f));
        var metrics = trainer.evaluate(samples);
        Assert.assertEquals(metrics.samples(), 3);
        Assert.assertEquals(metrics.batches(), 2);
        Assert.assertEquals(inputs.batchSizes, List.of(2, 1));
        Assert.assertEquals(metrics.handLoss(), (float) Math.log(4.0 / 3.0), 1e-6f);
        Assert.assertEquals(metrics.waitLoss(), (float) Math.log(2.0), 1e-6f);
        // 最終1行も1 バッチとして平均する既存の集約契約を維持する。
        float expectedScalar = 1.5f + (float) Math.log(2.0);
        Assert.assertEquals(metrics.scalarLoss(), expectedScalar, 1e-6f);
        Assert.assertEquals(
            metrics.loss(),
            (float) (Math.log(4.0 / 3.0) + .5 * Math.log(2.0) + .25 * expectedScalar),
            1e-6f);
        Assert.assertEquals(
            block.bias().toFloatArray(), new float[EpsilonBeliefLayout.OUTPUT_SIZE]);
        inputs.assertClosed();

        // BCEのabs/最大値の折れ点を避け、各出力層の更新方向を非ゼロロジットから検証する。
        float initialBias = .01f;
        block.bias().addi(initialBias);
        var trained = trainer.train(samples, 8);
        Assert.assertEquals(trained.samples(), 24);
        Assert.assertEquals(trained.batches(), 16);
        float[] updated = block.bias().toFloatArray();
        for (int opponent = 0; opponent < EpsilonBeliefLayout.OPPONENT_COUNT; opponent++) {
          int hand = opponent * Tile.NUM_TILE_TYPES;
          int wait = EpsilonBeliefLayout.OPPONENT_HAND_SIZE + hand;
          int scalar =
              EpsilonBeliefLayout.OPPONENT_HAND_SIZE
                  + EpsilonBeliefLayout.OPPONENT_WAIT_SIZE
                  + opponent * EpsilonBeliefLayout.OPPONENT_SCALAR_COUNT;
          Assert.assertTrue(updated[hand] < initialBias);
          Assert.assertTrue(updated[hand + 1] > initialBias);
          Assert.assertTrue(updated[wait] < initialBias);
          Assert.assertTrue(updated[wait + 1] > initialBias);
          Assert.assertTrue(updated[scalar + EpsilonBeliefLayout.SCALAR_SHANTEN] > initialBias);
          Assert.assertTrue(updated[scalar + EpsilonBeliefLayout.SCALAR_TENPAI] > initialBias);
        }
        Assert.assertTrue(trainer.evaluate(samples).loss() < metrics.loss());
        Assert.assertEquals(block.bias().toFloatArray(), updated);
        inputs.assertClosed();
      }
      Assert.assertTrue(model.getNDManager().isOpen());
      Assert.assertEquals(block.bias().toFloatArray().length, EpsilonBeliefLayout.OUTPUT_SIZE);
    }
  }

  @Test(groups = "native")
  public void inferenceSplitsConsecutiveBucketsAndKeepsInputOrderWithoutOwningModel() {
    TrackingInputs inputs = new TrackingInputs();
    TinyBlock block = new TinyBlock();
    try (Model model = model(block)) {
      block.output.getParameters().get("weight").getArray().addi(1f);
      List<EpsilonBeliefPrior> priors;
      try (var evaluator = new EpsilonBeliefInferenceServer<>(model, 2, inputs)) {
        priors =
            evaluator.evaluateBatch(
                List.of(
                    new State(1f, 0),
                    new State(2f, 0),
                    new State(3f, 0),
                    new State(4f, 1),
                    new State(5f, 1),
                    new State(6f, 0)));
        Assert.assertEquals(inputs.batchSizes, List.of(2, 1, 2, 1));
        Assert.assertEquals(priors.size(), 6);
        inputs.assertClosed();
        Assert.assertTrue(evaluator.evaluateBatch(List.of()).isEmpty());
        Assert.assertEquals(inputs.batchSizes, List.of(2, 1, 2, 1));
      }
      Assert.assertTrue(model.getNDManager().isOpen());
      Assert.assertEquals(block.bias().getFloat(0), 0f);
      for (int row = 0; row < priors.size(); row++) {
        float expected = 11f * (row + 1);
        for (float value : priors.get(row).opponentHandLogits()) {
          Assert.assertEquals(value, expected);
        }
        for (float value : priors.get(row).opponentWaitLogits()) {
          Assert.assertEquals(value, expected);
        }
        for (float value : priors.get(row).opponentScalars()) {
          Assert.assertEquals(value, expected);
        }
      }
    }
  }

  private static EpsilonBeliefSample<State> sample(float shanten) {
    float[] hands = new float[EpsilonBeliefLayout.OPPONENT_HAND_SIZE];
    float[] hidden = new float[Tile.NUM_TILE_TYPES];
    float[] waits = new float[EpsilonBeliefLayout.OPPONENT_WAIT_SIZE];
    float[] shantenValues = new float[EpsilonBeliefLayout.OPPONENT_COUNT];
    float[] tenpai = new float[EpsilonBeliefLayout.OPPONENT_COUNT];
    hidden[0] = 1f;
    hidden[1] = 3f;
    for (int opponent = 0; opponent < EpsilonBeliefLayout.OPPONENT_COUNT; opponent++) {
      hands[opponent * Tile.NUM_TILE_TYPES + 1] = 1f;
      waits[opponent * Tile.NUM_TILE_TYPES + 1] = 1f;
      shantenValues[opponent] = shanten;
      tenpai[opponent] = 1f;
    }
    return new EpsilonBeliefSample<>(
        new State(0f, 0), new EpsilonBeliefTarget(hands, hidden, shantenValues, tenpai, waits));
  }

  private static Model model(TinyBlock block) {
    Model model = Model.newInstance("belief-runtime-test", Device.cpu(), "PyTorch");
    model.setBlock(block);
    block.initialize(model.getNDManager(), DataType.FLOAT32, new Shape(-1, 1), new Shape(-1, 1));
    for (var parameter : block.getParameters()) {
      parameter.getValue().getArray().muli(0f);
    }
    return model;
  }

  private record State(float value, int bucket) {}

  private static final class TrackingInputs implements BeliefBatchFactory<State> {
    private final List<Integer> batchSizes = new ArrayList<>();
    private final List<NDManager> managers = new ArrayList<>();

    @Override
    public NDList transfer(NDManager manager, List<State> inputs) {
      batchSizes.add(inputs.size());
      managers.add(manager);
      float[] categories = new float[inputs.size()];
      float[] numerics = new float[inputs.size()];
      for (int row = 0; row < inputs.size(); row++) {
        categories[row] = inputs.get(row).value();
        numerics[row] = 10f * inputs.get(row).value();
      }
      Shape shape = new Shape(inputs.size(), 1);
      return new NDList(manager.create(categories, shape), manager.create(numerics, shape));
    }

    @Override
    public boolean sameBucket(State first, State second) {
      return first.bucket() == second.bucket();
    }

    private void assertClosed() {
      for (NDManager manager : managers) {
        Assert.assertFalse(manager.isOpen());
      }
    }
  }

  private static final class TinyBlock extends AbstractBlock {
    private final Linear output =
        addChildBlock("output", Linear.builder().setUnits(EpsilonBeliefLayout.OUTPUT_SIZE).build());

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType type, Shape... shapes) {
      output.initialize(manager, type, shapes[0]);
    }

    @Override
    protected NDList forwardInternal(
        ParameterStore store,
        NDList inputs,
        boolean training,
        PairList<String, Object> parameters) {
      return output.forward(
          store, new NDList(inputs.get(0).add(inputs.get(1))), training, parameters);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
      return output.getOutputShapes(new Shape[] {inputShapes[0]});
    }

    private NDArray bias() {
      return output.getParameters().get("bias").getArray();
    }
  }
}
