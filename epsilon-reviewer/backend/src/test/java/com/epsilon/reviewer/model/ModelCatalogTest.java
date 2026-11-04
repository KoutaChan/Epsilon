package com.epsilon.reviewer.model;

import com.epsilon.reviewer.FixturePolicyProvider;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/** models.json の厳密な読みと、中身を読まずに決める使えるか、作り置きのカタログ。 */
public class ModelCatalogTest {
  private static final PolicyProviders PROVIDERS = PolicyProviders.load();
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

  @Test
  public void availabilityIsDecidedWithoutReadingCheckpoints() throws IOException {
    Path link =
        Files.createSymbolicLink(directory.resolve("link"), directory.resolve("checkpoint"));
    String series = FixturePolicyProvider.SERIES;
    ModelCatalog catalog =
        load(
            "{\"models\":["
                + model("ok", "checkpoint", series, "\"options\":{\"temperature\":\"0.004\"}")
                + ","
                + model("unsupported", "checkpoint", "nope", null)
                + ","
                + model("missing", "absent", series, null)
                + ","
                + model("empty", "empty", series, null)
                + ","
                + model("linked", link.toString().replace("\\", "\\\\"), series, null)
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
  public void malformedCatalogsStopTheStart() {
    String series = FixturePolicyProvider.SERIES;
    String ok = model("a", "checkpoint", series, null);
    for (String json :
        List.of(
            "{\"models\":[" + ok + "],\"extra\":1}",
            "{\"models\":[" + ok.replace("\"series\"", "\"note\":\"x\",\"series\"") + "]}",
            "{\"models\":[" + ok + "," + ok + "]}",
            "{\"models\":[" + ok + "],\"defaultModel\":{\"modelId\":\"b\",\"revision\":\"r1\"}}",
            "{\"models\":[" + model("a", "checkpoint", series, "\"sha256\":\"ABC\"") + "]}",
            "{\"models\":["
                + model("a", "checkpoint", series, "\"options\":{\"temperature\":\"-1\"}")
                + "]}",
            "{\"models\":["
                + model("a", "checkpoint", series, "\"options\":{\"device\":1}")
                + "]}",
            "{\"models\":[" + ok.replace("\"a\"", "\" \"") + "]}",
            "{\"model\":[]}",
            "{\"models\":[" + ok + "]"))
      Assert.expectThrows(IOException.class, () -> load(json));
  }

  @Test
  public void anEmptyCatalogHasNoModels() {
    Assert.assertTrue(ModelCatalog.empty().view().models().isEmpty());
    Assert.assertNull(ModelCatalog.empty().view().defaultModel());
  }

  private ModelCatalog load(String json) throws IOException {
    Path file = directory.resolve("models.json");
    Files.writeString(file, json);
    return ModelCatalog.load(file, PROVIDERS);
  }

  /** @param extra 足す欄（"name":value の並び）。無ければ null */
  static String model(String id, String checkpoint, String series, String extra) {
    return "{\"modelId\":\""
        + id
        + "\",\"revision\":\"r1\",\"displayName\":\"Model\",\"version\":\"1\",\"series\":\""
        + series
        + "\",\"checkpoint\":\""
        + checkpoint
        + "\""
        + (extra == null ? "" : "," + extra)
        + "}";
  }
}
