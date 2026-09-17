package com.epsilon.reviewer.model;

import com.epsilon.reviewer.ApiException;
import java.nio.file.*;
import java.util.Comparator;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 読み込み済みモデルの利用権を保ち、元ファイルが変更された後の新規ジョブには同じ版として渡さないことを検証する。 */
public final class ModelCatalogTest {
  @Test
  public void fixedRevisionLeasesSurviveSourceChangesAndNewJobsRejectReplacement()
      throws Exception {
    Path directory = Files.createTempDirectory("reviewer-catalog-test-");
    try {
      Path checkpoint = Files.createDirectory(directory.resolve("checkpoint"));
      Files.writeString(checkpoint.resolve("manifest.json"), "original");
      Files.write(checkpoint.resolve("network.params"), new byte[] {1, 2, 3});
      Path config = directory.resolve("models.json");
      Files.writeString(
          config,
          "{\"models\":[{\"modelId\":\"epsilon\",\"revision\":\"r1\",\"displayName\":\"Epsilon\",\"version\":\"1\",\"series\":\"epsilon\",\"checkpoint\":\"checkpoint\"}]}");
      var catalog = new ModelCatalog(config);
      Assert.assertEquals(catalog.publicCatalog().models().getFirst().availability(), "available");
      Assert.assertFalse(catalog.publicCatalog().toString().contains("checkpoint"));
      Path leased;
      try (var lease = catalog.acquire("epsilon", "r1", directory.resolve("leases"))) {
        leased = lease.definition().checkpoint();
        Files.writeString(checkpoint.resolve("manifest.json"), "changed");
        Assert.assertEquals(Files.readString(leased.resolve("manifest.json")), "original");
        ApiException changed =
            Assert.expectThrows(
                ApiException.class, () -> catalog.requireAvailableModelRevision("epsilon", "r1"));
        Assert.assertEquals(changed.code(), "model_changed");
        Assert.assertEquals(
            catalog.publicCatalog().models().getFirst().reasonCode(), changed.code());
        Assert.expectThrows(
            ApiException.class,
            () -> catalog.acquire("epsilon", "r1", directory.resolve("leases")));
      }
      Assert.assertFalse(Files.exists(leased));
    } finally {
      try (var paths = Files.walk(directory)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }
}
