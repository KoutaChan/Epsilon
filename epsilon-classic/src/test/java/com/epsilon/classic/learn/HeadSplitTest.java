package com.epsilon.classic.learn;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.RandomCheckpoints;
import com.epsilon.classic.input.Buckets;
import com.epsilon.classic.input.Buckets.Bucket;
import com.epsilon.classic.input.ClassicExtras;
import com.epsilon.classic.input.ClassicInputs;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.input.ClassicTokenizer;
import com.epsilon.classic.model.ClassicBatch;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.classic.model.ClassicNetwork;
import com.epsilon.classic.model.PolicyGraph;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Random;
import java.util.stream.Stream;
import org.testng.annotations.Test;

/**
 * 学習器がパラメーターを 2 つの組に分けるやり方（{@link ClassicNetwork#valueParameter}）を確かめる。価値の損失の勾配は価値の層にだけ
 * 流れ、共有の符号化の層へは流れない。方策の損失の勾配は価値の層へ流れない。Nano と Major を CPU の FP32 で確かめる。
 */
public class HeadSplitTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final Bucket BUCKET = Buckets.required(ClassicSchema.MAX_LEGAL, ClassicSchema.MAX_TRANSITIONS);
  private static final int PER_SEAT = 12;

  /** 勾配を流す損失（方策のヘッドか価値のヘッドか）。 */
  private enum Loss {
    POLICY,
    VALUE
  }

  @Test(groups = "native")
  public void eachLossOnlyReachesItsOwnParameters() throws IOException {
    Trajectories.Pool pool = new Trajectories.Pool(VARIANT, 13, PER_SEAT);
    Path root = Files.createTempDirectory("head-split");
    try {
      for (Architecture architecture : Architecture.values()) {
        Path checkpoint = root.resolve(architecture.name());
        RandomCheckpoints.write(checkpoint, architecture, VARIANT, 21);
        for (Loss loss : Loss.values()) {
          check(checkpoint, architecture, pool, loss);
        }
      }
    } finally {
      try (Stream<Path> paths = Files.walk(root)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
          Files.deleteIfExists(path);
        }
      }
    }
  }

  private static void check(Path checkpoint, Architecture architecture, Trajectories.Pool pool, Loss loss)
      throws IOException {
    int rows = PER_SEAT * VARIANT.seats();
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      ClassicNetwork network = ClassicCheckpoint.read(checkpoint, VARIANT, manager);
      for (Pair<String, Parameter> pair : network.getParameters()) {
        pair.getValue().getArray().setRequiresGradient(true);
      }
      ClassicInputs inputs = new ClassicInputs(ClassicSchema.of(architecture), BUCKET, rows);
      ClassicTokenizer tokenizer = new ClassicTokenizer(architecture, VARIANT);
      Observation reader = new Observation(VARIANT);
      float[] chosen = new float[rows * BUCKET.actions()];
      float[] utilities = new float[rows];
      Random random = new Random(17);
      for (int row = 0; row < rows; row++) {
        Observation o = pool.bind(row % VARIANT.seats(), row / VARIANT.seats(), reader);
        tokenizer.write(o, ClassicExtras.empty(), inputs, row);
        chosen[row * BUCKET.actions() + random.nextInt(o.legalActions())] = 1;
        utilities[row] = (float) (random.nextDouble() * 1.6 - 1);
      }
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        ClassicBatch batch = ClassicBatch.send(manager, PinnedUploads.on(manager), inputs, rows);
        ClassicNetwork.Output output =
            network.forward(new ParameterStore(manager, false), batch, Precision.FLOAT32, true, ClassicNetwork.Heads.BOTH);
        NDArray objective =
            loss == Loss.POLICY
                ? PolicyGraph.logProbabilities(output.policy(), batch)
                    .mul(manager.create(chosen, new Shape(rows, BUCKET.actions())))
                    .sum()
                    .neg()
                : network.support().targets(manager.create(utilities)).mul(output.value().logSoftmax(1)).sum().neg();
        collector.backward(objective);
      }
      int reached = 0;
      int valueParameters = 0;
      for (Pair<String, Parameter> pair : network.getParameters()) {
        boolean value = ClassicNetwork.valueParameter(pair.getKey());
        valueParameters += value ? 1 : 0;
        double peak = peak(pair.getValue().getArray());
        if (value == (loss == Loss.VALUE)) {
          reached += peak > 0 ? 1 : 0;
        } else {
          assertEquals(peak, 0.0, 0.0, architecture + " " + loss + " loss reached " + pair.getKey());
        }
      }
      assertTrue(reached > 0, architecture + " " + loss + " loss reached nothing");
      assertTrue(valueParameters > 0 && valueParameters < network.getParameters().size(), "split " + valueParameters);
    }
  }

  /** 勾配の絶対値の最大（勾配が無ければ 0）。 */
  private static double peak(NDArray array) {
    PtNDArray gradient = JniUtils.getGradient((PtNDArray) array);
    if (gradient == null) {
      return 0;
    }
    try (gradient) {
      return gradient.abs().max().getFloat();
    }
  }
}
