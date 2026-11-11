package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;
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
import java.util.Arrays;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Stream;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * チェックポイントを確かめる。確かめるのは、GrpNetwork・GruSinkhorn・OutcomeNetwork を書いて読み戻せること、SHA-256 が変わらない
 * こと、書き換えを拒み、書きかけを片付けること、grp.json の中身、形に合わないファイルを拒むこと、2 つのチェックポイントから FinalRound を
 * 作り直せること。
 */
public class GrpCheckpointTest {

  private Path root;

  @BeforeMethod
  public void createRoot() throws IOException {
    root = Files.createTempDirectory("grp-checkpoint");
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRoot() throws IOException {
    try (Stream<Path> paths = Files.walk(root)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(path);
      }
    }
  }

  @Test
  public void checkpointsRoundTripWithAStableSha() throws IOException, NoSuchAlgorithmException {
    GrpShape shape = new GrpShape(Variant.TENHOU_4P, EnumSet.allOf(GrpField.class), 12, true);
    GrpNetwork network = GrpFixtures.network(shape, 21, 0.5);
    Path first = root.resolve("first");
    String sha = GrpCheckpoint.write(network, first);
    Assert.assertEquals(GrpCheckpoint.sha256(first), sha);

    GrpNetwork read = (GrpNetwork) GrpCheckpoint.read(first);
    Assert.assertEquals(read.shape(), shape);
    Assert.assertTrue(Arrays.equals(read.weights(), network.weights()));
    Path second = root.resolve("second");
    Assert.assertEquals(GrpCheckpoint.write(read, second), sha);
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.write(GrpFixtures.network(shape, 22, 0.5), first));
    Assert.assertEquals(GrpCheckpoint.sha256(first), sha);

    byte[] description = Files.readAllBytes(first.resolve(GrpCheckpoint.JSON_FILE));
    byte[] weights = Files.readAllBytes(first.resolve(GrpCheckpoint.WEIGHTS_FILE));
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    digest.update(description);
    digest.update(weights);
    Assert.assertEquals(HexFormat.of().formatHex(digest.digest()), sha);

    float[] expected = network.weights();
    Assert.assertEquals(weights.length, Float.BYTES * expected.length);
    ByteBuffer buffer = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN);
    for (int index = 0; index < expected.length; index++) {
      Assert.assertEquals(Float.floatToRawIntBits(buffer.getFloat()), Float.floatToRawIntBits(expected[index]));
    }
  }

  /** 書きかけで落ちた一時ディレクトリは次の書き込みが消し、親のディレクトリには書き終えたチェックポイントだけが残る。 */
  @Test
  public void aStaleStagingDirectoryIsReplaced() throws IOException {
    GrpShape shape = GrpFixtures.shape(Variant.TENHOU_4P, 6, false);
    Path directory = root.resolve("model");
    Path staging = root.resolve(".model.writing");
    Files.createDirectories(staging);
    Files.writeString(staging.resolve(GrpCheckpoint.JSON_FILE), "{");
    String sha = GrpCheckpoint.write(GrpFixtures.network(shape, 5, 1.0), directory);
    Assert.assertFalse(Files.exists(staging));
    Assert.assertEquals(GrpCheckpoint.sha256(directory), sha);
    try (Stream<Path> entries = Files.list(root)) {
      Assert.assertEquals(entries.map(path -> path.getFileName().toString()).toList(), List.of("model"));
    }
  }

  @Test
  public void differentWeightsGiveADifferentSha() throws IOException {
    GrpShape shape = GrpFixtures.shape(Variant.TENHOU_3P, 6, false);
    String first = GrpCheckpoint.write(GrpFixtures.network(shape, 1, 1.0), root.resolve("first"));
    String second = GrpCheckpoint.write(GrpFixtures.network(shape, 2, 1.0), root.resolve("second"));
    Assert.assertNotEquals(first, second);
    Assert.assertTrue(first.matches("[0-9a-f]{64}"), first);
  }

  @Test
  public void descriptionNamesTheModelAndTheFieldsOnly() throws IOException {
    GrpShape shape = GrpFixtures.shape(Variant.TENHOU_3P, 6, true);
    Path directory = root.resolve("three");
    GrpCheckpoint.write(GrpFixtures.network(shape, 3, 1.0), directory);
    JsonObject description = description(directory);
    Assert.assertEquals(description.keySet(), Set.of("model", "variant", "fields", "hidden", "phaseHeads"));
    Assert.assertEquals(description.get("model").getAsString(), "GrpNetwork");
    Assert.assertEquals(description.get("variant").getAsString(), "TENHOU_3P");
    Assert.assertEquals(description.get("hidden").getAsInt(), 6);
    Assert.assertTrue(description.get("phaseHeads").getAsBoolean());
    JsonObject fields = description.getAsJsonObject("fields");
    Assert.assertEquals(List.copyOf(fields.keySet()), List.of("ROUND", "HONBA", "KYOTAKU", "SCORE", "RANK", "GAP"));
    Assert.assertEquals(fields.get("ROUND").getAsInt(), 9);
    Assert.assertEquals(fields.get("RANK").getAsInt(), 9);
    Assert.assertEquals(fields.get("GAP").getAsInt(), 3);
  }

  @Test
  public void filesThatDoNotFitTheCodeAreRefused() throws IOException {
    GrpShape shape = GrpFixtures.shape(Variant.TENHOU_4P, 6, false);
    GrpNetwork network = GrpFixtures.network(shape, 4, 1.0);

    Path width = root.resolve("width");
    GrpCheckpoint.write(network, width);
    JsonObject wider = description(width);
    wider.getAsJsonObject("fields").addProperty("ROUND", 13);
    rewrite(width, wider);
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.read(width));

    Path model = root.resolve("model");
    GrpCheckpoint.write(network, model);
    JsonObject other = description(model);
    other.addProperty("model", "GruSinkhorn");
    rewrite(model, other);
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.read(model));

    Path extra = root.resolve("extra");
    GrpCheckpoint.write(network, extra);
    JsonObject versioned = description(extra);
    versioned.addProperty("version", 1);
    rewrite(extra, versioned);
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.read(extra));

    Path truncated = root.resolve("truncated");
    GrpCheckpoint.write(network, truncated);
    Path weights = truncated.resolve(GrpCheckpoint.WEIGHTS_FILE);
    byte[] bytes = Files.readAllBytes(weights);
    Files.write(weights, Arrays.copyOf(bytes, bytes.length - Float.BYTES));
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.read(truncated));

    Path infinite = root.resolve("infinite");
    GrpCheckpoint.write(network, infinite);
    Path infiniteWeights = infinite.resolve(GrpCheckpoint.WEIGHTS_FILE);
    byte[] infiniteBytes = Files.readAllBytes(infiniteWeights);
    ByteBuffer.wrap(infiniteBytes).order(ByteOrder.LITTLE_ENDIAN).putFloat(0, Float.POSITIVE_INFINITY);
    Files.write(infiniteWeights, infiniteBytes);
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.read(infinite));
  }

  /**
   * GruSinkhorn のチェックポイントを書いて読み戻すと、重みのビットも出す値も同じで、SHA-256 も変わらない。weights.f32 は重みの
   * 配列の順に並び、grp.json はモデルの種類と形だけを持つ。
   */
  @Test
  public void gruCheckpointsRoundTripWithAStableSha() throws IOException {
    GruShape shape = new GruShape(Variant.TENHOU_4P, 6, 2);
    GruSinkhorn gru = GrpFixtures.gru(shape, 23, 0.5);
    Path first = root.resolve("gru");
    String sha = GrpCheckpoint.write(gru, first);
    Assert.assertEquals(GrpCheckpoint.sha256(first), sha);
    Assert.assertTrue(sha.matches("[0-9a-f]{64}"), sha);

    GruSinkhorn read = (GruSinkhorn) GrpCheckpoint.read(first);
    Assert.assertEquals(read.shape(), shape);
    Assert.assertTrue(Arrays.equals(read.weights(), gru.weights()));
    Assert.assertEquals(GrpCheckpoint.write(read, root.resolve("again")), sha);
    Assert.assertNotEquals(GrpCheckpoint.write(GrpFixtures.gru(shape, 24, 0.5), root.resolve("other")), sha);

    Random random = new Random(25);
    RoundHistory history = GrpFixtures.history(Variant.TENHOU_4P, random, 4);
    GrpState state = GrpFixtures.state(Variant.TENHOU_4P, random);
    double[] expected = new double[16];
    double[] actual = new double[16];
    gru.marginals(history, state, GruWorkspace.of(gru), expected);
    read.marginals(history, state, GruWorkspace.of(read), actual);
    Assert.assertEquals(actual, expected);

    float[] weights = gru.weights();
    byte[] bytes = Files.readAllBytes(first.resolve(GrpCheckpoint.WEIGHTS_FILE));
    Assert.assertEquals(bytes.length, Float.BYTES * GruTensor.count(shape));
    ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    for (float weight : weights) {
      Assert.assertEquals(Float.floatToRawIntBits(buffer.getFloat()), Float.floatToRawIntBits(weight));
    }

    JsonObject description = description(first);
    Assert.assertEquals(description.keySet(), Set.of("model", "variant", "hidden", "layers"));
    Assert.assertEquals(description.get("model").getAsString(), "GruSinkhorn");
    Assert.assertEquals(description.get("variant").getAsString(), "TENHOU_4P");
    Assert.assertEquals(description.get("hidden").getAsInt(), 6);
    Assert.assertEquals(description.get("layers").getAsInt(), 2);
  }

  @Test
  public void gruFilesThatDoNotFitAreRefused() throws IOException {
    GruShape shape = new GruShape(Variant.TENHOU_3P, 4, 1);
    GruSinkhorn gru = GrpFixtures.gru(shape, 9, 1.0);

    Path layers = root.resolve("layers");
    GrpCheckpoint.write(gru, layers);
    JsonObject deeper = description(layers);
    deeper.addProperty("layers", 2);
    rewrite(layers, deeper);
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.read(layers));

    Path fields = root.resolve("fields");
    GrpCheckpoint.write(gru, fields);
    JsonObject withFields = description(fields);
    withFields.add("fields", new JsonObject());
    rewrite(fields, withFields);
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.read(fields));

    Path empty = root.resolve("empty");
    GrpCheckpoint.write(gru, empty);
    JsonObject zero = description(empty);
    zero.addProperty("hidden", 0);
    rewrite(empty, zero);
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.read(empty));

    Path nan = root.resolve("nan");
    GrpCheckpoint.write(gru, nan);
    Path weights = nan.resolve(GrpCheckpoint.WEIGHTS_FILE);
    byte[] bytes = Files.readAllBytes(weights);
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putFloat(Float.BYTES, Float.NaN);
    Files.write(weights, bytes);
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.read(nan));
  }

  /**
   * OutcomeNetwork のチェックポイントを書いて読み戻すと、重みのビットが同じで、SHA-256 も変わらない。weights.f32 は OutcomeTensor
   * の順に並び、grp.json はモデルの種類・フィールド・隠れ層の幅だけを持つ。葉のネットワークのチェックポイントと合わせて作り直した
   * FinalRound はビット単位で同じ p を出す。読むメソッドは 2 つとも、もう片方のチェックポイントを拒む。
   */
  @Test
  public void outcomeCheckpointsRoundTripAndRebuildTheFinalRound() throws IOException {
    OutcomeShape shape = new OutcomeShape(Variant.TENHOU_4P, GrpFixtures.plainFields(), 6);
    OutcomeNetwork outcomes = GrpFixtures.outcomes(shape, 27, 0.5);
    Path first = root.resolve("outcomes");
    String sha = GrpCheckpoint.write(outcomes, first);
    Assert.assertEquals(GrpCheckpoint.sha256(first), sha);
    Assert.assertTrue(sha.matches("[0-9a-f]{64}"), sha);

    OutcomeNetwork read = GrpCheckpoint.readOutcomes(first);
    Assert.assertEquals(read.shape(), shape);
    Assert.assertTrue(Arrays.equals(read.weights(), outcomes.weights()));
    Assert.assertEquals(GrpCheckpoint.write(read, root.resolve("again")), sha);
    Assert.assertNotEquals(GrpCheckpoint.write(GrpFixtures.outcomes(shape, 28, 0.5), root.resolve("other")), sha);
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.write(outcomes, first));

    byte[] bytes = Files.readAllBytes(first.resolve(GrpCheckpoint.WEIGHTS_FILE));
    float[] weights = outcomes.weights();
    Assert.assertEquals(bytes.length, Float.BYTES * OutcomeTensor.starts(shape)[OutcomeTensor.values().length]);
    ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    for (float weight : weights) {
      Assert.assertEquals(Float.floatToRawIntBits(buffer.getFloat()), Float.floatToRawIntBits(weight));
    }
    JsonObject description = description(first);
    Assert.assertEquals(description.keySet(), Set.of("model", "variant", "fields", "hidden"));
    Assert.assertEquals(description.get("model").getAsString(), "OutcomeNetwork");
    Assert.assertEquals(description.get("variant").getAsString(), "TENHOU_4P");
    Assert.assertEquals(description.get("hidden").getAsInt(), 6);
    Assert.assertEquals(description.getAsJsonObject("fields").get("GAP").getAsInt(), 6);

    GrpShape leafShape = new GrpShape(Variant.TENHOU_4P, GrpFixtures.plainFields(), 6, true);
    GrpNetwork leaf = GrpFixtures.network(leafShape, 29, 0.5);
    Path leafDirectory = root.resolve("leaf");
    GrpCheckpoint.write(leaf, leafDirectory);
    FinalRound original = FinalRound.of(outcomes, leaf, 1e-4);
    FinalRound rebuilt = FinalRound.of(
        GrpCheckpoint.readOutcomes(first), (GrpNetwork) GrpCheckpoint.read(leafDirectory), original.cutoff());
    GrpState south4 = new GrpState(Variant.TENHOU_4P, 7, 1, 1, new int[] {28_000, 26_000, 24_000, 21_000});
    double[] expected = new double[24];
    double[] actual = new double[24];
    original.permutations(south4, FinalRoundWorkspace.of(original), expected);
    rebuilt.permutations(south4, FinalRoundWorkspace.of(rebuilt), actual);
    for (int k = 0; k < expected.length; k++) {
      Assert.assertEquals(Double.doubleToRawLongBits(actual[k]), Double.doubleToRawLongBits(expected[k]), "π " + k);
    }

    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.read(first));
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.readOutcomes(leafDirectory));
  }

  @Test
  public void outcomeFilesThatDoNotFitAreRefused() throws IOException {
    OutcomeShape shape = new OutcomeShape(Variant.TENHOU_4P, GrpFixtures.plainFields(), 4);
    OutcomeNetwork outcomes = GrpFixtures.outcomes(shape, 31, 1.0);

    Path three = root.resolve("three");
    GrpCheckpoint.write(outcomes, three);
    JsonObject threePlayers = description(three);
    threePlayers.addProperty("variant", "TENHOU_3P");
    threePlayers.getAsJsonObject("fields").addProperty("ROUND", 9);
    threePlayers.getAsJsonObject("fields").addProperty("SCORE", 3);
    threePlayers.getAsJsonObject("fields").addProperty("RANK", 9);
    threePlayers.getAsJsonObject("fields").addProperty("GAP", 3);
    rewrite(three, threePlayers);
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.readOutcomes(three));

    Path heads = root.resolve("heads");
    GrpCheckpoint.write(outcomes, heads);
    JsonObject withHeads = description(heads);
    withHeads.addProperty("phaseHeads", false);
    rewrite(heads, withHeads);
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.readOutcomes(heads));

    Path width = root.resolve("width");
    GrpCheckpoint.write(outcomes, width);
    JsonObject wider = description(width);
    wider.getAsJsonObject("fields").addProperty("GAP", 7);
    rewrite(width, wider);
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.readOutcomes(width));

    Path truncated = root.resolve("truncated");
    GrpCheckpoint.write(outcomes, truncated);
    Path weights = truncated.resolve(GrpCheckpoint.WEIGHTS_FILE);
    byte[] bytes = Files.readAllBytes(weights);
    Files.write(weights, Arrays.copyOf(bytes, bytes.length - Float.BYTES));
    Assert.assertThrows(IOException.class, () -> GrpCheckpoint.readOutcomes(truncated));
  }

  private static JsonObject description(Path directory) throws IOException {
    String text = Files.readString(directory.resolve(GrpCheckpoint.JSON_FILE), StandardCharsets.UTF_8);
    return JsonParser.parseString(text).getAsJsonObject();
  }

  private static void rewrite(Path directory, JsonObject description) throws IOException {
    Files.writeString(directory.resolve(GrpCheckpoint.JSON_FILE), description.toString(), StandardCharsets.UTF_8);
  }
}
