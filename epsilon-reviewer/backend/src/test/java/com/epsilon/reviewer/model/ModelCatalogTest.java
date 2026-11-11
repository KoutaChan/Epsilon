package com.epsilon.reviewer.model;

import com.epsilon.agent.checkpoint.CheckpointFiles;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.KeyInfo;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.FixtureFamily;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import java.util.Optional;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * models.json の厳密な読み取り（系列・ルール・設定）と、エラーメッセージ（キーのパス・書いた値・選べる値）を確かめる。ほかに、
 * チェックポイントの中身を読まずに使えるかを決めること、作っておいたカタログ、今の revision も確かめる。
 */
public class ModelCatalogTest {
  private static final String SERIES = FixtureFamily.SERIES.id();

  /** テスト用の系列と、チェックポイントを使わない系列の入口。 */
  private static final ModelFamilies FAMILIES =
      ModelFamilies.of(List.of(new FixtureFamily(), new RuleBotFamily()));

  private Path directory;

  @BeforeMethod
  public void create() throws IOException {
    directory = TestFiles.directory("reviewer-catalog-");
    Path checkpoint = Files.createDirectory(directory.resolve("checkpoint"));
    Files.writeString(checkpoint.resolve("weights.bin"), "w");
    Files.createDirectory(directory.resolve("empty"));
  }

  @AfterMethod(alwaysRun = true)
  public void delete() {
    TestFiles.delete(directory);
  }

  /** models.json の options は、方策を開く設定の全てのキーを同じ既定値と範囲で持ち、ほかには温度だけを持つ。 */
  @Test
  public void modelOptionsCarryEveryPolicyOption() {
    List<String> policy = Settings.describe(PolicyOptions.class).stream().map(ModelCatalogTest::key).toList();
    List<String> model =
        Settings.describe(ModelOptions.class).stream()
            .filter(info -> !info.path().equals("temperature"))
            .map(ModelCatalogTest::key)
            .toList();
    Assert.assertEquals(model, policy);
  }

  /** キーのパス・既定値・範囲を 1 つの文字列にする。 */
  private static String key(KeyInfo info) {
    return info.path() + "=" + info.value() + " " + info.range();
  }

  @Test
  public void availabilityIsDecidedWithoutReadingCheckpoints() throws IOException {
    Path link =
        Files.createSymbolicLink(directory.resolve("link"), directory.resolve("checkpoint"));
    ModelCatalog catalog =
        load(
            "{\"models\":["
                + model("ok", "checkpoint", SERIES, "\"options\":{\"temperature\":0.004}")
                + ","
                + model("unsupported", "checkpoint", Series.PRISM.id(), null)
                + ","
                + model("missing", "absent", SERIES, null)
                + ","
                + model("empty", "empty", SERIES, null)
                + ","
                + model("linked", link.toString().replace("\\", "\\\\"), SERIES, null)
                + "],\"defaultModel\":{\"modelId\":\"ok\",\"revision\":\"r1\"}}");
    CatalogView view = catalog.view();
    Assert.assertEquals(
        view.models().stream().map(CatalogModel::modelId).toList(),
        List.of("ok", "unsupported", "missing", "empty", "linked"));
    Assert.assertEquals(
        view.models().stream().map(CatalogModel::available).toList(),
        List.of(true, false, false, false, false));
    Assert.assertEquals(view.defaultModel(), new ModelKey("ok", "r1"));
    Assert.assertSame(catalog.view(), view, "the view is built once");
    ModelEntry ok = catalog.require(new ModelKey("ok", "r1"));
    Assert.assertEquals(ok.temperature(), 0.004);
    Assert.assertEquals(
        ok.checkpoint(), directory.resolve("checkpoint").toAbsolutePath().normalize());
    for (String id : List.of("unsupported", "missing", "empty", "linked", "unknown")) {
      var key = new ModelKey(id, "r1");
      Assert.assertEquals(
          Assert.expectThrows(ReviewerException.class, () -> catalog.require(key)).code(),
          ErrorCode.MODEL_UNAVAILABLE,
          id);
    }
    catalog.markChanged(new ModelKey("ok", "r1"));
    Assert.assertFalse(catalog.view().models().getFirst().available());
    Assert.expectThrows(ReviewerException.class, () -> catalog.require(new ModelKey("ok", "r1")));
    Assert.assertFalse(view.toString().contains("checkpoint"), "the view hides paths");
  }

  @Test
  public void anUnservedSeriesIsUnsupported() throws IOException {
    ModelCatalog catalog =
        ModelCatalog.load(
            write("{\"models\":[" + model("a", "checkpoint", SERIES, null) + "]}"),
            ModelFamilies.of(List.of()));
    Assert.assertFalse(catalog.view().models().getFirst().available());
    Assert.assertEquals(
        Assert.expectThrows(ReviewerException.class, () -> catalog.require(new ModelKey("a", "r1")))
            .code(),
        ErrorCode.MODEL_UNAVAILABLE);
  }

  @Test
  public void seriesVariantAndOptionsAreRead() throws IOException {
    ModelCatalog catalog =
        load(
            "{\"models\":["
                + model("default", "checkpoint", SERIES, null)
                + ","
                + model(
                    "gpu",
                    "checkpoint",
                    SERIES,
                    "\"options\":{\"device\":\"gpu1\",\"precision\":\"BFLOAT16\",\"maxBatch\":16,\"temperature\":0.5}",
                    "TENHOU_3P")
                + ","
                + model("cpu", "checkpoint", SERIES, "\"options\":{\"device\":\"cpu\"}")
                + ","
                + model("rules", null, Series.RULE_BOT.id(), null)
                + "]}");
    ModelEntry plain = catalog.require(new ModelKey("default", "r1"));
    Assert.assertEquals(plain.series(), FixtureFamily.SERIES);
    Assert.assertEquals(plain.variant(), Variant.TENHOU_4P);
    Assert.assertEquals(plain.options(), PolicyOptions.defaults());
    Assert.assertNull(plain.temperature(), "without an override the policy's temperature is used");
    Assert.assertNull(plain.sha256());
    ModelEntry gpu = catalog.require(new ModelKey("gpu", "r1"));
    Assert.assertEquals(gpu.variant(), Variant.TENHOU_3P);
    Assert.assertEquals(gpu.options(), new PolicyOptions("gpu1", Precision.BFLOAT16, 16, Kernels.EAGER));
    Assert.assertEquals(gpu.temperature(), 0.5);
    Assert.assertEquals(
        catalog.require(new ModelKey("cpu", "r1")).options(),
        PolicyOptions.defaults());
    ModelEntry rules = catalog.require(new ModelKey("rules", "r1"));
    Assert.assertEquals(rules.series(), Series.RULE_BOT);
    Assert.assertNull(rules.checkpoint());
  }

  @Test
  public void malformedCatalogsStopTheStart() {
    String ok = model("a", "checkpoint", SERIES, null);
    String sha = "\"sha256\":\"" + "0".repeat(64) + "\"";
    for (String json :
        List.of(
            "{\"models\":[" + ok + "],\"extra\":1}",
            "{\"models\":[" + ok.replace("\"series\"", "\"note\":\"x\",\"series\"") + "]}",
            "{\"models\":[" + ok + "," + ok + "]}",
            "{\"models\":[" + ok + "],\"defaultModel\":{\"modelId\":\"b\",\"revision\":\"r1\"}}",
            "{\"models\":[" + model("a", "checkpoint", SERIES, "\"sha256\":\"ABC\"") + "]}",
            "{\"models\":[" + options("{\"temperature\":-1}") + "]}",
            "{\"models\":[" + options("{\"temperature\":0}") + "]}",
            "{\"models\":[" + options("{\"temperature\":\"0.004\"}") + "]}",
            "{\"models\":[" + options("{\"temperature\":\"x\"}") + "]}",
            "{\"models\":[" + options("{\"device\":1}") + "]}",
            "{\"models\":[" + options("{\"device\":\"gpu\"}") + "]}",
            "{\"models\":[" + options("{\"device\":\"gpu:0\"}") + "]}",
            "{\"models\":[" + options("{\"device\":\"gpu01\"}") + "]}",
            "{\"models\":[" + options("{\"device\":\"CPU\"}") + "]}",
            "{\"models\":[" + options("{\"hold\":\"true\"}") + "]}",
            "{\"models\":[" + options("{\"precision\":\"HALF\"}") + "]}",
            "{\"models\":[" + options("{\"maxBatch\":0}") + "]}",
            "{\"models\":[" + options("{\"kernels\":\"FAST\"}") + "]}",
            "{\"models\":[" + model("a", "checkpoint", "fixture-scenario", null) + "]}",
            "{\"models\":[" + model("a", "checkpoint", "prism", null) + "]}",
            "{\"models\":[" + model("a", "checkpoint", SERIES, null, "TENHOU_5P") + "]}",
            "{\"models\":[" + model("a", "checkpoint", SERIES, null, "tenhou_4p") + "]}",
            "{\"models\":[" + ok.replace(",\"variant\":\"TENHOU_4P\"", "") + "]}",
            "{\"models\":[" + model("a", "checkpoint", Series.RULE_BOT.id(), null) + "]}",
            "{\"models\":[" + model("a", null, SERIES, null) + "]}",
            "{\"models\":[" + model("a", null, Series.RULE_BOT.id(), sha) + "]}",
            "{\"models\":[" + ok.replace("\"a\"", "\" \"") + "]}",
            "{\"model\":[]}",
            "{\"models\":[" + ok + "]"))
      Assert.expectThrows(SettingsException.class, () -> load(json));
  }

  /**
   * エラーメッセージは誤り 1 つにつき 1 行で、キーのパス・書いた値・書いたファイルを含む。引用符で囲んだ数には、引用符を外すよう書き添える。
   * enum と device には選べる値を書く。誤りは全て集めて、まとめて知らせる。
   */
  @Test
  public void catalogErrorsNameThePathTheValueAndTheChoices() {
    String quoted = problems("{\"models\":[" + options("{\"temperature\":\"0.004\"}") + "]}");
    Assert.assertTrue(quoted.contains("models[0].options.temperature=0.004"), quoted);
    Assert.assertTrue(quoted.contains("models.json"), quoted);
    Assert.assertTrue(quoted.contains("(written as a string: drop the quotes)"), quoted);
    String range = problems("{\"models\":[" + options("{\"temperature\":0}") + "]}");
    Assert.assertTrue(range.contains("models[0].options.temperature=0"), range);
    Assert.assertFalse(range.contains("drop the quotes"), range);
    String device = problems("{\"models\":[" + options("{\"device\":\"gpu-1\"}") + "]}");
    Assert.assertTrue(device.contains("models[0].options.device"), device);
    Assert.assertTrue(device.contains("expected cpu or gpuN"), device);
    String precision = problems("{\"models\":[" + options("{\"precision\":\"HALF\"}") + "]}");
    Assert.assertTrue(precision.contains("models[0].options.precision=HALF"), precision);
    for (Precision choice : Precision.values()) {
      Assert.assertTrue(precision.contains(choice.name()), precision);
    }
    String variant = problems("{\"models\":[" + model("a", "checkpoint", SERIES, null, "tenhou_4p") + "]}");
    Assert.assertTrue(variant.contains("models[0].variant=tenhou_4p"), variant);
    Assert.assertTrue(variant.contains("TENHOU_4P"), variant);
    String series = problems("{\"models\":[" + model("a", "checkpoint", "prism", null) + "]}");
    Assert.assertTrue(series.contains("models[0].series"), series);
    Assert.assertTrue(series.contains(Series.PRISM.id()), series);
    String both =
        problems(
            "{\"models\":["
                + options("{\"temperature\":\"0.004\"}")
                + ","
                + model("b", "checkpoint", SERIES, "\"options\":{\"device\":\"gpu-1\"}")
                + "]}");
    Assert.assertEquals(both.lines().count(), 2L, both);
    Assert.assertTrue(
        both.contains("models[0].options.temperature") && both.contains("models[1].options.device"), both);
  }

  @Test
  public void currentIsTheLastListedRevision() throws IOException {
    ModelCatalog catalog =
        load(
            "{\"models\":["
                + model("a", "checkpoint", SERIES, null).replace("\"r1\"", "\"r2\"")
                + ","
                + model("b", "checkpoint", SERIES, null)
                + ","
                + model("a", "absent", SERIES, null).replace("\"r1\"", "\"r0\"")
                + "]}");
    Assert.assertEquals(catalog.current("a"), Optional.of(new ModelKey("a", "r0")));
    Assert.assertEquals(catalog.current("b"), Optional.of(new ModelKey("b", "r1")));
    Assert.assertEquals(catalog.current("c"), Optional.empty());
    Assert.assertTrue(ModelCatalog.empty().current("a").isEmpty());
  }

  @Test
  public void aScenarioUsesTheWrittenRevisionWhileItIsAvailable() throws IOException {
    ModelCatalog catalog =
        load(
            "{\"models\":["
                + model("a", "checkpoint", SERIES, null)
                + ","
                + model("a", "checkpoint", SERIES, null).replace("\"r1\"", "\"r2\"")
                + ","
                + model("b", "absent", SERIES, null)
                + ","
                + model("b", "checkpoint", SERIES, null).replace("\"r1\"", "\"r2\"")
                + "]}");
    Assert.assertEquals(catalog.scenarioModel(new ModelKey("a", "r1")), new ModelKey("a", "r1"));
    Assert.assertEquals(catalog.scenarioModel(new ModelKey("a", "r0")), new ModelKey("a", "r2"));
    Assert.assertEquals(catalog.scenarioModel(new ModelKey("b", "r1")), new ModelKey("b", "r2"));
    Assert.assertEquals(
        Assert.expectThrows(
                ReviewerException.class, () -> catalog.scenarioModel(new ModelKey("c", "r1")))
            .code(),
        ErrorCode.MODEL_UNAVAILABLE);
  }

  @Test
  public void twoFamiliesForOneSeriesAreRefused() {
    Assert.expectThrows(
        IllegalStateException.class,
        () -> ModelFamilies.of(List.of(new FixtureFamily(), new FixtureFamily())));
  }

  /** models.json の sha256 は、前の Reviewer が求めていたフォルダーのハッシュと同じ値になる。 */
  @Test
  public void checkpointHashesMatchThePreviousDigest() throws IOException {
    Path checkpoint = directory.resolve("checkpoint");
    Files.writeString(checkpoint.resolve("model.json"), "{\"width\":256}");
    Files.writeString(checkpoint.resolve("a.txt"), "a");
    Path nested = Files.createDirectories(checkpoint.resolve("a").resolve("b"));
    Files.write(nested.resolve("c.bin"), new byte[] {0, 1, 2, (byte) 0xff});
    Files.writeString(checkpoint.resolve("a-b"), "dash");
    String hash = CheckpointFiles.sha256(checkpoint);
    Assert.assertTrue(hash.matches("[0-9a-f]{64}"), hash);
    Assert.assertEquals(hash, previousDigest(checkpoint));
  }

  @Test
  public void anEmptyCatalogHasNoModels() {
    Assert.assertTrue(ModelCatalog.empty().view().models().isEmpty());
    Assert.assertNull(ModelCatalog.empty().view().defaultModel());
  }

  /** 読めない models.json のエラーメッセージ（1 行に 1 つ）。 */
  private String problems(String json) {
    return String.join("\n", Assert.expectThrows(SettingsException.class, () -> load(json)).problems());
  }

  private ModelCatalog load(String json) throws IOException {
    return ModelCatalog.load(write(json), FAMILIES);
  }

  private Path write(String json) throws IOException {
    Path file = directory.resolve("models.json");
    Files.writeString(file, json);
    return file;
  }

  private static String options(String options) {
    return model("a", "checkpoint", SERIES, "\"options\":" + options);
  }

  /**
   * 天鳳の四人打ちのモデルを書いた、models.json の項目 1 つ。
   *
   * @param checkpoint チェックポイント。無ければ null
   * @param extra 足すフィールド（"name":value の並び）。無ければ null
   */
  static String model(String id, String checkpoint, String series, String extra) {
    return model(id, checkpoint, series, extra, Variant.TENHOU_4P.name());
  }

  static String model(String id, String checkpoint, String series, String extra, String variant) {
    return "{\"modelId\":\""
        + id
        + "\",\"revision\":\"r1\",\"displayName\":\"Model\",\"version\":\"1\",\"series\":\""
        + series
        + "\",\"variant\":\""
        + variant
        + "\""
        + (checkpoint == null ? "" : ",\"checkpoint\":\"" + checkpoint + "\"")
        + (extra == null ? "" : "," + extra)
        + "}";
  }

  /** 前の Reviewer と同じ作り方で求めたフォルダーのハッシュ。 */
  private static String previousDigest(Path directory) throws IOException {
    MessageDigest digest;
    try {
      digest = MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException failure) {
      throw new IllegalStateException(failure);
    }
    for (Path file : CheckpointFingerprint.files(directory)) {
      digest.update(CheckpointFingerprint.relative(directory, file).getBytes(StandardCharsets.UTF_8));
      digest.update((byte) 0);
      digest.update(ByteBuffer.allocate(Long.BYTES).putLong(Files.size(file)).array());
      try (InputStream in = new DigestInputStream(Files.newInputStream(file), digest)) {
        in.transferTo(OutputStream.nullOutputStream());
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  /** チェックポイントを使わない系列の入口。開こうとすると例外を投げる。 */
  private static final class RuleBotFamily implements ModelFamily {
    @Override
    public Series series() {
      return Series.RULE_BOT;
    }

    @Override
    public Policy open(
        Path checkpoint, Variant variant, PolicyOptions options, NetworkRuntime runtime) {
      throw new UnsupportedOperationException();
    }
  }
}
