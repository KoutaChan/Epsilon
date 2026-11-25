package com.epsilon.prism.model;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.run.Snapshot;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 保存したネットワークの model.json のフィンガープリントと、読むときに重みの形を照合することを確かめる。 */
public class PrismNetworkFormatTest {

  private static PrismNetwork network(PrismShape shape, NDManager manager) {
    PrismNetwork network = new PrismNetwork(FOUR_PLAYER, shape);
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
      Assert.assertEquals(
          json.getAsJsonObject("shape").toString(),
          "{\"hidden\":64,\"ffn\":128,\"gate\":64,\"depth\":2}");
      Assert.assertEquals(PrismNetwork.readShape(directory), shape);
      Assert.assertEquals(PrismNetwork.load(directory, manager).shape(), shape);
      Assert.assertThrows(
          FileAlreadyExistsException.class, () -> network(shape, manager).save(root));
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void experimentalInputsRoundTripAndRejectAnotherFingerprint() throws IOException {
    Path root = Files.createTempDirectory("prism-v4-schema");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      for (ShapeArm arm : List.of(ShapeArm.V4_S, ShapeArm.V4_M)) {
        Path directory = root.resolve(arm.name());
        PrismNetwork net = network(arm.shape(), manager);
        net.save(directory);
        var loaded = PrismNetwork.load(directory, manager);
        Assert.assertEquals(loaded.shape(), arm.shape());
        Assert.assertSame(loaded.format(), net.format());
        String fingerprint = net.format().fingerprint();
        Assert.assertNotEquals(fingerprint, FOUR_PLAYER.fingerprint());
        replace(directory, fingerprint, FOUR_PLAYER.fingerprint());
        Assert.assertThrows(IOException.class, () -> PrismNetwork.readHeader(directory));
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void typedShapeIsSavedAsV3AndReadUnderEitherName() throws IOException {
    Path root = Files.createTempDirectory("prism-format");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismShape.V2 small = Batches.small(2);
      PrismShape typed =
          new PrismShape.V3(small.hidden(), small.ffn(), small.gate(), 128, 128, 128, true);
      Path directory = root.resolve("typed");
      network(typed, manager).save(directory);
      Assert.assertEquals(architecture(directory), PrismArchitecture.V3.id());
      Assert.assertEquals(
          shapeJson(directory).toString(),
          "{\"hidden\":64,\"ffn\":128,\"gate\":64,\"depth\":3,"
              + "\"typed\":{\"tile\":128,\"context\":128,\"candidate\":128,\"reread\":true}}");
      Assert.assertEquals(PrismNetwork.load(directory, manager).shape(), typed);
      rename(directory, PrismArchitecture.V3.id(), PrismArchitecture.V2.id());
      Assert.assertEquals(PrismNetwork.load(directory, manager).shape(), typed);
      replace(directory, "\"depth\": 3", "\"depth\": 2");
      IOException shallow =
          Assert.expectThrows(IOException.class, () -> PrismNetwork.readShape(directory));
      Assert.assertTrue(
          shallow.getMessage().contains("typed FFNs need depth 3"), shallow.getMessage());

      Path untyped = root.resolve("untyped");
      network(small, manager).save(untyped);
      Assert.assertEquals(architecture(untyped), PrismArchitecture.V2.id());
      rename(untyped, PrismArchitecture.V2.id(), PrismArchitecture.V3.id());
      IOException thrown =
          Assert.expectThrows(IOException.class, () -> PrismNetwork.readShape(untyped));
      Assert.assertTrue(thrown.getMessage().contains("needs typed FFNs"), thrown.getMessage());
      rename(untyped, PrismArchitecture.V3.id(), "epsilon-prism-v9");
      thrown = Assert.expectThrows(IOException.class, () -> PrismNetwork.readShape(untyped));
      Assert.assertTrue(
          thrown.getMessage().contains("unknown PRISM architecture"), thrown.getMessage());
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void coreMetadataRequiresItsDeclaredFields() throws IOException {
    Path root = Files.createTempDirectory("prism-sized-core");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      Path directory = root.resolve("medium");
      network(ShapeArm.V4_M.shape(), manager).save(directory);
      Path metadata = directory.resolve("model.json");
      var header = JsonParser.parseString(Files.readString(metadata)).getAsJsonObject();
      JsonObject saved = header.getAsJsonObject("shape");
      Assert.assertEquals(saved.keySet(), Set.of("hidden", "layer0Ffn", "depth", "eventAttention"));
      var width = saved.remove("layer0Ffn");
      Files.writeString(metadata, header.toString());
      Assert.assertThrows(IOException.class, () -> PrismNetwork.readHeader(directory));
      saved.add("layer0Ffn", width);
      saved.addProperty("reread", false);
      Files.writeString(metadata, header.toString());
      Assert.assertThrows(IOException.class, () -> PrismNetwork.readHeader(directory));
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @DataProvider
  public Object[][] invalidV4Fields() {
    return new Object[][] {
      {"eventMode", "true"},
      {"eventMode", "null"},
      {"eventMode", "\"UNKNOWN\""},
      {"reread", "null"},
      {"reread", "1"},
      {"waitRelation", "null"},
      {"waitRelation", "1"},
      {"layer0Ffn", "384.5"},
      {"eventMlpWidth", "0"},
      {"seatBias", "true"},
      {"frontier", "false"},
      {"relationKeys", "false"},
      {"requiredFacts", "null"},
      {"requiredFacts", "[12]"},
      {"requiredFacts", "[\"UNKNOWN\"]"},
      {"requiredFacts", "[\"TEDASHI\",\"TEDASHI\"]"}
    };
  }

  @Test(dataProvider = "invalidV4Fields")
  public void v4RejectsInvalidShapeFields(String field, String json) throws IOException {
    Path root = Files.createTempDirectory("prism-format");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      Path directory = root.resolve("model");
      network(new PrismShape.V4(), manager).save(directory);
      Path metadata = directory.resolve("model.json");
      JsonObject header = JsonParser.parseString(Files.readString(metadata)).getAsJsonObject();
      header.getAsJsonObject("shape").add(field, JsonParser.parseString(json));
      Files.writeString(metadata, header.toString());
      Assert.assertThrows(IOException.class, () -> PrismNetwork.readShape(directory));
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
        Assert.expectThrows(
            IllegalArgumentException.class, () -> PrismArchitecture.byId("epsilon-prism-v9"));
    Assert.assertTrue(
        thrown.getMessage().contains("unknown PRISM architecture"), thrown.getMessage());
  }

  private static String architecture(Path directory) throws IOException {
    return JsonParser.parseString(Files.readString(directory.resolve("model.json")))
        .getAsJsonObject()
        .get("architecture")
        .getAsString();
  }

  private static JsonObject shapeJson(Path directory) throws IOException {
    return JsonParser.parseString(Files.readString(directory.resolve("model.json")))
        .getAsJsonObject()
        .getAsJsonObject("shape");
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

  @Test(dataProvider = "savedShapes", dataProviderClass = ModelShapes.class)
  public void roundTripPreservesShapeAndScores(PrismShape shape) throws IOException {
    Path root = Files.createTempDirectory("prism-format");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = network(shape, manager);
      Batches.perturb(network, 0.05f, 83);
      var host =
          com.epsilon.prism.backend.HipFixtures.encode(
              com.epsilon.prism.backend.HipFixtures.mixedRows(4, 89), shape);
      Path directory = root.resolve("model");
      network.save(directory);
      Assert.assertEquals(PrismNetwork.readShape(directory), shape);
      Assert.assertEquals(architecture(directory), shape.architecture().id());
      PrismNetwork loaded = PrismNetwork.load(directory, manager);
      ParameterStore store = new ParameterStore(manager, false);
      Pass pass = Pass.inference(Precision.FLOAT32);
      NetworkInputs inputs = NetworkInputs.of(manager, PinnedUploads.on(manager), host);
      Assert.assertEquals(
          loaded.forward(store, inputs, pass).scores().toFloatArray(),
          network.forward(store, inputs, pass).scores().toFloatArray());
      for (String key : loaded.inferenceParameters().keys()) {
        Assert.assertFalse(key.contains("reading_") || key.contains("outcome_"));
      }
      replace(directory, network.format().fingerprint(), "0000000000000000");
      Assert.assertThrows(IOException.class, () -> PrismNetwork.readShape(directory));
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test(dataProvider = "savedShapes", dataProviderClass = ModelShapes.class)
  public void loadRejectsWeightsOfAnotherShape(PrismShape built) throws IOException {
    Path root = Files.createTempDirectory("prism-format");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismShape other =
          switch (built) {
            case PrismShape.V2 v2 ->
                new PrismShape.V2(v2.hidden(), 2 * v2.ffn(), v2.gate(), v2.depth());
            case PrismShape.V3 v3 ->
                new PrismShape.V3(
                    v3.hidden(),
                    v3.ffn(),
                    v3.gate(),
                    2 * v3.tile(),
                    v3.context(),
                    v3.candidate(),
                    v3.reread());
            case PrismShape.V4 v4 ->
                new PrismShape.V4(v4.hidden(), 2 * v4.layer0Ffn(), v4.depth(), v4.eventAttention());
          };
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
