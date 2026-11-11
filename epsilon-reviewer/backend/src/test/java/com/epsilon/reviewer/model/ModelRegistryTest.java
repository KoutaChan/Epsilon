package com.epsilon.reviewer.model;

import com.epsilon.agent.checkpoint.CheckpointFiles;
import com.epsilon.reviewer.FixtureFamily;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 開いた方策 1 つを、全ての貸し出しで共有することを確かめる。チェックポイントのフィンガープリントが変わったときと、sha256 が合わない
 * ときは、モデルを使えなくする。しばらく使われていない方策は閉じる。開ける数を超えたときは、最も長く使われていない方策を閉じる。
 * 貸し出し中の方策は閉じない。開けないモデルは、何度求められても断る。
 */
public class ModelRegistryTest {
  private static final ModelFamilies FAMILIES = FixtureFamily.families();
  private static final String SERIES = FixtureFamily.SERIES.id();
  private static final ModelKey FIRST = new ModelKey("first", "r1");
  private static final ModelKey SECOND = new ModelKey("second", "r1");
  private Path directory;
  private Path checkpoint;

  @BeforeMethod
  public void create() throws IOException {
    directory = TestFiles.directory("reviewer-registry-");
    checkpoint = Files.createDirectory(directory.resolve("checkpoint"));
    Files.writeString(checkpoint.resolve("weights.bin"), "weights");
  }

  @AfterMethod(alwaysRun = true)
  public void delete() {
    TestFiles.delete(directory);
  }

  @Test
  public void oneOpenPolicyIsSharedByEveryLease() throws IOException {
    try (var registry = registry(catalog(null), 2, Duration.ofMinutes(10))) {
      int opened = FixtureFamily.opened.get();
      try (ModelLease first = registry.acquire(FIRST);
          ModelLease second = registry.acquire(FIRST)) {
        Assert.assertEquals(FixtureFamily.opened.get(), opened + 1);
        Assert.assertEquals(first.info().modelId(), "first");
        Assert.assertEquals(second.temperature(), FixtureFamily.TEMPERATURE);
      }
    }
  }

  @Test
  public void idleAndLeastRecentlyUsedPoliciesAreClosedButLeasedOnesStay() throws IOException {
    ModelCatalog catalog = catalog(null);
    int closed = FixtureFamily.closed.get();
    try (var registry = registry(catalog, 1, Duration.ofMinutes(10))) {
      registry.acquire(FIRST).close();
      try (ModelLease second = registry.acquire(SECOND)) {
        Assert.assertEquals(FixtureFamily.closed.get(), closed + 1, "first was closed");
        try (ModelLease first = registry.acquire(FIRST)) {
          Assert.assertEquals(
              FixtureFamily.closed.get(), closed + 1, "leased models are not closed");
        }
      }
    }
    closed = FixtureFamily.closed.get();
    try (var registry = registry(catalog, 2, Duration.ZERO)) {
      registry.acquire(FIRST).close();
      try (ModelLease kept = registry.acquire(SECOND)) {
        registry.sweep();
        Assert.assertEquals(FixtureFamily.closed.get(), closed + 1, "only the idle one");
      }
    }
  }

  @Test
  public void aChangedCheckpointMakesTheModelUnavailable() throws IOException {
    ModelCatalog catalog = catalog(null);
    try (var registry = registry(catalog, 2, Duration.ZERO)) {
      registry.acquire(FIRST).close();
      registry.sweep();
      Files.writeString(checkpoint.resolve("weights.bin"), "changed weights");
      refused(registry, FIRST);
      Assert.assertFalse(catalog.view().models().getFirst().available());
      refused(registry, FIRST);
    }
  }

  @Test
  public void theSha256IsCheckedAtTheFirstOpen() throws IOException {
    String digest = CheckpointFiles.sha256(checkpoint);
    Assert.assertTrue(digest.matches("[0-9a-f]{64}"));
    try (var registry = registry(catalog(digest), 2, Duration.ofMinutes(10))) {
      registry.acquire(FIRST).close();
    }
    ModelCatalog wrong = catalog("0".repeat(64));
    try (var registry = registry(wrong, 2, Duration.ofMinutes(10))) {
      refused(registry, FIRST);
      Assert.assertFalse(wrong.view().models().getFirst().available());
    }
  }

  @Test
  public void modelsThatCannotOpenStayRefusable() throws IOException {
    Files.writeString(checkpoint.resolve(FixtureFamily.BROKEN), "");
    Path models = directory.resolve("models.json");
    Files.writeString(
        models, "{\"models\":[" + ModelCatalogTest.model("first", "checkpoint", SERIES, null) + "]}");
    for (String failure : List.of("", "io", "linkage")) {
      Files.writeString(checkpoint.resolve(FixtureFamily.BROKEN), failure);
      try (var registry =
          registry(ModelCatalog.load(models, FAMILIES), 2, Duration.ofMinutes(10))) {
        refused(registry, FIRST);
        refused(registry, FIRST);
      }
    }
  }

  private ModelCatalog catalog(String sha256) throws IOException {
    String extra = sha256 == null ? null : "\"sha256\":\"" + sha256 + "\"";
    Path models = directory.resolve("models.json");
    Files.writeString(
        models,
        "{\"models\":["
            + ModelCatalogTest.model("first", "checkpoint", SERIES, extra)
            + ","
            + ModelCatalogTest.model("second", "checkpoint", SERIES, null)
            + "]}");
    return ModelCatalog.load(models, FAMILIES);
  }

  private static ModelRegistry registry(ModelCatalog catalog, int maxOpen, Duration idle) {
    return new ModelRegistry(catalog, FAMILIES, maxOpen, idle);
  }

  private static void refused(ModelRegistry registry, ModelKey key) {
    ReviewerException failure =
        Assert.expectThrows(ReviewerException.class, () -> registry.acquire(key));
    Assert.assertEquals(failure.code(), ErrorCode.MODEL_UNAVAILABLE);
  }
}
