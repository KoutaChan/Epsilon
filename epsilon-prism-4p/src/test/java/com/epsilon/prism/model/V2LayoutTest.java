package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.model.Batches.Batch;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism4p.input.InputSchema;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Set;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * v2 の作りが変わらないことのテスト。パラメーターの名前・順・形、保存した model.json の shape の項目、乱数を足したネットワークと公開した
 * 重み（v2b のバージョン 760）の s のビットを、固定したフィンガープリントと照合する。
 */
public class V2LayoutTest {

  /** 公開した重みの場所。読むだけで書かない。無ければテストを飛ばす。 */
  private static final Path PUBLISHED =
      Path.of(System.getProperty("user.home"), "epsilon-ai/crr/runs/prism-v2b/snapshots/00760/m0");

  private static final String PARAMETERS_DEPTH_2 =
      "3c307451da99c0cbf8823d4f2a001783376402ef8b575c187172e263d6ce2bd1";
  private static final String PARAMETERS_DEPTH_3 =
      "f6341f188efc2ddbbe05a452e3ce5ce1e8ccb230e1bf7806ec3d9cb60de0353f";
  private static final String PERTURBED_SCORES =
      "690b61adee0a18401556e943d2166fde87a3611e982c25442339a5626a4588af";
  private static final String PUBLISHED_SCORES =
      "104957ae130f6489c0e1bbb44adfb32a97dcd3b6e54ca33a6931a95ad26e594c";

  @Test
  public void parameterNamesOrderAndShapesStay() {
    Assert.assertEquals(parameters(2), PARAMETERS_DEPTH_2, "depth 2");
    Assert.assertEquals(parameters(3), PARAMETERS_DEPTH_3, "depth 3");
  }

  @Test
  public void savedShapeKeepsTheV2Fields() throws IOException {
    Path root = Files.createTempDirectory("prism-v2-layout");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, PrismShape.standard());
      network.initialize(manager);
      Path directory = root.resolve("model");
      network.save(directory);
      JsonObject json =
          JsonParser.parseString(Files.readString(directory.resolve("model.json")))
              .getAsJsonObject();
      Assert.assertEquals(
          json.getAsJsonObject("shape").keySet(),
          Set.of("hidden", "ffn", "gate", "depth"));
      Assert.assertEquals(PrismNetwork.readShape(directory), PrismShape.standard());
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void perturbedNetworkScoresTheSameBits() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      manager.getEngine().setRandomSeed(41);
      PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, PrismShape.standard(3));
      network.initialize(manager);
      Batches.perturb(network, 0.05f, 43);
      try (NDManager scope = manager.newSubManager()) {
        Batch batch = Batches.build(Batches.allKinds(7), 47);
        float[] scores =
            network
                .forward(
                    new ParameterStore(manager, false),
                    batch.inputs(scope),
                    Pass.inference(Precision.FLOAT32))
                .scores()
                .toFloatArray();
        Assert.assertEquals(digest(scores), PERTURBED_SCORES);
      }
    }
  }

  @Test
  public void publishedWeightsScoreTheSameBits() throws IOException {
    if (!Files.isRegularFile(PUBLISHED.resolve("weights.bin"))) {
      throw new SkipException("published weights are not here: " + PUBLISHED);
    }
    try (FrozenNetwork network = FrozenNetwork.load(PUBLISHED, Device.cpu(), Precision.FLOAT32);
        NDManager scope = network.newScope();
        InferenceMode inference = network.inference()) {
      Batch batch = Batches.build(Batches.allKinds(9), 53);
      float[] scores = network.scores(batch.inputs(scope)).toFloatArray();
      Assert.assertEquals(digest(scores), PUBLISHED_SCORES);
    }
  }

  /** パラメーターの名前と形を登録の順に並べた文字列のフィンガープリント。 */
  private static String parameters(int depth) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, PrismShape.standard(depth));
      network.initialize(manager);
      StringBuilder text = new StringBuilder();
      for (Pair<String, Parameter> pair : network.getParameters()) {
        text.append(pair.getKey()).append(pair.getValue().getArray().getShape()).append(';');
      }
      return sha256(text.toString().getBytes(StandardCharsets.UTF_8));
    }
  }

  /** 値のビット列のフィンガープリント。 */
  private static String digest(float[] values) {
    ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      bytes.putInt(Float.floatToRawIntBits(value));
    }
    return sha256(bytes.array());
  }

  private static String sha256(byte[] bytes) {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    } catch (NoSuchAlgorithmException missing) {
      throw new IllegalStateException(missing);
    }
  }
}
