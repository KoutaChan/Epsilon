package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism4p.input.InputSchema;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 保存したネットワークの model.json のフィンガープリントと、読むときに重みの形を照合することを確かめる。 */
public class PrismNetworkFormatTest {

  private static PrismNetwork network(PrismShape shape, NDManager manager) {
    PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, shape);
    network.initialize(manager);
    return network;
  }

  @Test
  public void savedModelCarriesTheOldFingerprints() throws IOException {
    Path root = Files.createTempDirectory("prism-format");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      Path directory = root.resolve("model");
      PrismShape shape = Batches.small(2);
      network(shape, manager).save(directory);
      JsonObject json =
          JsonParser.parseString(Files.readString(directory.resolve("model.json")))
              .getAsJsonObject();
      Assert.assertEquals(json.get("architecture").getAsString(), PrismNetwork.ARCHITECTURE);
      Assert.assertEquals(json.get("inputSchema").getAsString(), "eeeb4f583cd2b92a");
      Assert.assertEquals(json.get("headSchema").getAsString(), "391be70aa4a7bf7b");
      Assert.assertEquals(PrismNetwork.readShape(directory), shape);
      Assert.assertEquals(PrismNetwork.load(directory, manager).shape(), shape);
      Assert.assertThrows(FileAlreadyExistsException.class, () -> network(shape, manager).save(root));
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void loadRejectsWeightsOfAnotherShape() throws IOException {
    Path root = Files.createTempDirectory("prism-format");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismShape built = Batches.small(2);
      PrismShape other = new PrismShape(built.hidden(), 2 * built.ffn(), built.gate(), built.depth(), null);
      Path directory = root.resolve("built");
      Path wrong = root.resolve("other");
      network(built, manager).save(directory);
      network(other, manager).save(wrong);
      Files.copy(
          wrong.resolve("weights.bin"),
          directory.resolve("weights.bin"),
          StandardCopyOption.REPLACE_EXISTING);
      Assert.assertEquals(PrismNetwork.readShape(directory), built);
      IOException thrown =
          Assert.expectThrows(IOException.class, () -> PrismNetwork.load(directory, manager));
      Assert.assertTrue(thrown.getMessage().contains("shape"), thrown.getMessage());
    } finally {
      Snapshot.deleteTree(root);
    }
  }
}
