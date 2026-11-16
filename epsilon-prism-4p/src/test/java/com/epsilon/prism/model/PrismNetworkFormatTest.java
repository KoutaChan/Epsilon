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
      Assert.assertEquals(json.get("architecture").getAsString(), PrismArchitecture.V2.id());
      Assert.assertEquals(json.get("inputSchema").getAsString(), "eeeb4f583cd2b92a");
      Assert.assertEquals(json.get("headSchema").getAsString(), "391be70aa4a7bf7b");
      Assert.assertEquals(json.get("shape").toString(), "{\"hidden\":64,\"ffn\":128,\"gate\":64,\"depth\":2}");
      Assert.assertEquals(PrismNetwork.readShape(directory), shape);
      Assert.assertEquals(PrismNetwork.load(directory, manager).shape(), shape);
      Assert.assertThrows(FileAlreadyExistsException.class, () -> network(shape, manager).save(root));
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void typedShapeIsSavedAsV3AndReadUnderEitherName() throws IOException {
    Path root = Files.createTempDirectory("prism-format");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismShape small = Batches.small(2);
      PrismShape typed = new PrismShape.V3(small.hidden(), small.ffn(), small.gate(), 128, 128, 128, true);
      Path directory = root.resolve("typed");
      network(typed, manager).save(directory);
      Assert.assertEquals(architecture(directory), PrismArchitecture.V3.id());
      Assert.assertEquals(
          shapeJson(directory),
          "{\"hidden\":64,\"ffn\":128,\"gate\":64,\"depth\":3,"
              + "\"typed\":{\"tile\":128,\"context\":128,\"candidate\":128,\"reread\":true}}");
      Assert.assertEquals(PrismNetwork.load(directory, manager).shape(), typed);
      rename(directory, PrismArchitecture.V3.id(), PrismArchitecture.V2.id());
      Assert.assertEquals(PrismNetwork.load(directory, manager).shape(), typed);
      replace(directory, "\"depth\": 3", "\"depth\": 2");
      IOException shallow = Assert.expectThrows(IOException.class, () -> PrismNetwork.readShape(directory));
      Assert.assertTrue(shallow.getMessage().contains("typed FFNs need depth 3"), shallow.getMessage());

      Path untyped = root.resolve("untyped");
      network(small, manager).save(untyped);
      Assert.assertEquals(architecture(untyped), PrismArchitecture.V2.id());
      rename(untyped, PrismArchitecture.V2.id(), PrismArchitecture.V3.id());
      IOException thrown = Assert.expectThrows(IOException.class, () -> PrismNetwork.readShape(untyped));
      Assert.assertTrue(thrown.getMessage().contains("needs typed FFNs"), thrown.getMessage());
      rename(untyped, PrismArchitecture.V3.id(), "epsilon-prism-v9");
      thrown = Assert.expectThrows(IOException.class, () -> PrismNetwork.readShape(untyped));
      Assert.assertTrue(thrown.getMessage().contains("unknown PRISM architecture"), thrown.getMessage());
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void architectureFollowsTheShape() {
    Assert.assertEquals(PrismShape.standard(3).architecture(), PrismArchitecture.V2);
    Assert.assertEquals(ShapeArm.V3.shape(ShapeArm.V3_DEPTH).architecture(), PrismArchitecture.V3);
    for (PrismArchitecture architecture : PrismArchitecture.values()) {
      Assert.assertEquals(PrismArchitecture.byId(architecture.id()), architecture);
    }
    IllegalArgumentException thrown =
        Assert.expectThrows(IllegalArgumentException.class, () -> PrismArchitecture.byId("epsilon-prism-v9"));
    Assert.assertTrue(thrown.getMessage().contains("unknown PRISM architecture"), thrown.getMessage());
  }

  private static String architecture(Path directory) throws IOException {
    return JsonParser.parseString(Files.readString(directory.resolve("model.json")))
        .getAsJsonObject()
        .get("architecture")
        .getAsString();
  }

  private static String shapeJson(Path directory) throws IOException {
    return JsonParser.parseString(Files.readString(directory.resolve("model.json")))
        .getAsJsonObject()
        .get("shape")
        .toString();
  }

  private static void rename(Path directory, String from, String to) throws IOException {
    replace(directory, '"' + from + '"', '"' + to + '"');
  }

  private static void replace(Path directory, String from, String to) throws IOException {
    Path json = directory.resolve("model.json");
    String text = Files.readString(json);
    Assert.assertTrue(text.contains(from), from);
    Files.writeString(json, text.replace(from, to));
  }

  @Test
  public void loadRejectsWeightsOfAnotherShape() throws IOException {
    Path root = Files.createTempDirectory("prism-format");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismShape built = Batches.small(2);
      PrismShape other = new PrismShape.V2(built.hidden(), 2 * built.ffn(), built.gate(), built.depth());
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
