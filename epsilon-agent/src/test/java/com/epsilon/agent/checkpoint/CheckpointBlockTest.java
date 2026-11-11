package com.epsilon.agent.checkpoint;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotEquals;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;
import org.testng.annotations.Test;

/** 小さな DJL のブロックの重みを書き、別に組んだブロックへ読み込むと、値も名前も同じになる。重みが同じディレクトリは SHA-256 も同じになる。 */
public class CheckpointBlockTest {

  public record Header(String architecture, Variant variant, int hidden) {}

  private static final Shape INPUT = new Shape(2, 5);
  private static final Header HEADER = new Header("test-mlp", Variant.TENHOU_4P, 3);

  @Test(groups = "native")
  public void parametersRoundTripThroughTheDirectory() throws IOException {
    Path root = Files.createTempDirectory("checkpoint-block");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      Block written = initialized(manager);
      Path directory = root.resolve("00001");
      CheckpointFiles.write(directory, written, HEADER);
      assertEquals(CheckpointFiles.readHeader(directory, Header.class), HEADER);

      Block read = initialized(manager);
      CheckpointFiles.loadParameters(directory, read, manager);
      PairList<String, Parameter> expected = written.getParameters();
      PairList<String, Parameter> actual = read.getParameters();
      assertEquals(actual.keys(), expected.keys());
      for (int index = 0; index < expected.size(); index++) {
        Pair<String, Parameter> parameter = expected.get(index);
        assertEquals(
            actual.get(index).getValue().getArray().toFloatArray(),
            parameter.getValue().getArray().toFloatArray(),
            parameter.getKey());
      }

      Path copy = root.resolve("00002");
      CheckpointFiles.write(copy, read, HEADER);
      assertEquals(CheckpointFiles.sha256(copy), CheckpointFiles.sha256(directory));
      Path other = root.resolve("00003");
      CheckpointFiles.write(other, initialized(manager), HEADER);
      assertNotEquals(CheckpointFiles.sha256(other), CheckpointFiles.sha256(directory));
    } finally {
      try (Stream<Path> files = Files.walk(root)) {
        for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
          Files.delete(file);
        }
      }
    }
  }

  /** 線形・ReLU・線形のブロックを、乱数の初期値で作る。 */
  private static Block initialized(NDManager manager) {
    SequentialBlock block =
        new SequentialBlock()
            .add(Linear.builder().setUnits(HEADER.hidden()).build())
            .add(Activation.reluBlock())
            .add(Linear.builder().setUnits(1).build());
    block.initialize(manager, DataType.FLOAT32, INPUT);
    return block;
  }
}
