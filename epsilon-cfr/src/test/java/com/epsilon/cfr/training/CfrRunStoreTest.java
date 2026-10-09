package com.epsilon.cfr.training;

import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.io.CfrFiles;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import org.testng.Assert;
import org.testng.annotations.Test;

/** モデルの実行エンジンやネイティブライブラリを使わず、Javaだけで永続化を検証する。 */
public class CfrRunStoreTest {
  @Test
  public void publishedStateRoundTripsAndDetectsChangedConfiguration() throws Exception {
    Path directory = Files.createTempDirectory("cfr-run-store-");
    try {
      try (CfrRunStore store = new CfrRunStore(directory)) {
        CfrRunState state = initialState(store);
        store.publish(state);
        Assert.assertEquals(store.read(), state);
        Files.writeString(directory.resolve("config.json"), "{}\n");
        Assert.expectThrows(IOException.class, store::read);
      }
    } finally {
      delete(directory);
    }
  }

  @Test
  public void failedPublicationPreservesThePreviousRecoveryPoint() throws Exception {
    Path directory = Files.createTempDirectory("cfr-run-atomic-");
    try {
      try (CfrRunStore store = new CfrRunStore(directory)) {
        CfrRunState state = initialState(store);
        store.publish(state);
        byte[] pointer = Files.readAllBytes(directory.resolve("current.json"));
        CfrRunState invalid =
            new CfrRunState(
                2,
                state.solveId(),
                0,
                1,
                CfrRunState.Phase.READY,
                0,
                0,
                "artifacts/missing-model",
                state.averageCheckpoint(),
                null,
                0,
                null,
                "disabled",
                state.configHash());
        Assert.expectThrows(IOException.class, () -> store.publish(invalid));
        Assert.assertEquals(Files.readAllBytes(directory.resolve("current.json")), pointer);
        Assert.assertEquals(store.read(), state);
      }
    } finally {
      delete(directory);
    }
  }

  @Test
  public void fullArtifactChecksRunAtExplicitVerification() throws Exception {
    Path directory = Files.createTempDirectory("cfr-run-verify-");
    try {
      try (CfrRunStore store = new CfrRunStore(directory)) {
        CfrRunState state = initialState(store);
        store.publish(state);
        Assert.assertEquals(store.verify(), state);

        Path weights = store.resolve(state.averageCheckpoint()).resolve("weights.bin");
        Files.write(weights, new byte[] {1, 2, 4});
        // 通常の状態遷移では参照先だけを確認し、モデルの内容を読み直さない。
        store.publish(state);
        Assert.assertEquals(store.read(), state);
        Assert.expectThrows(IOException.class, store::verify);

        Files.write(weights, new byte[] {1, 2, 3});
        Assert.assertEquals(store.verify(), state);
      }
    } finally {
      delete(directory);
    }
  }

  @Test
  public void writerLockExcludesConcurrentWriterAndReleasesOnClose() throws Exception {
    Path directory = Files.createTempDirectory("cfr-lock-");
    try {
      try (CfrRunStore first = new CfrRunStore(directory)) {
        Assert.expectThrows(
            Exception.class,
            () -> {
              try (CfrRunStore ignored = new CfrRunStore(directory)) {
                Assert.fail("Second writer acquired lock");
              }
            });
      }
      try (CfrRunStore afterClose = new CfrRunStore(directory)) {
        Assert.assertEquals(afterClose.root(), directory);
      }
    } finally {
      delete(directory);
    }
  }

  @Test
  public void artifactChecksumAndRelativePathBoundaryAreEnforced() throws Exception {
    Path directory = Files.createTempDirectory("cfr-artifact-");
    try {
      Path artifact = directory.resolve("artifact");
      Files.createDirectories(artifact);
      Files.write(artifact.resolve("weights.bin"), new byte[] {1, 2, 3});
      CfrRunStore.sealArtifact(artifact);
      CfrRunStore.verifyArtifact(artifact);
      Files.write(artifact.resolve("weights.bin"), new byte[] {1, 2, 4});
      Assert.expectThrows(IOException.class, () -> CfrRunStore.verifyArtifact(artifact));
      Assert.expectThrows(IOException.class, () -> CfrFiles.inside(directory, "../outside.bin"));
      Assert.expectThrows(
          IOException.class, () -> CfrFiles.inside(directory, directory.toString()));
    } finally {
      delete(directory);
    }
  }

  private static CfrRunState initialState(CfrRunStore store) throws IOException {
    CfrConfig.defaults().write(store.root().resolve("config.json"));
    Path model = store.newArtifact("model", 0);
    Files.write(model.resolve("weights.bin"), new byte[] {1, 2, 3});
    CfrRunStore.sealArtifact(model);
    return new CfrRunState(
        2,
        "solve-test",
        0,
        1,
        CfrRunState.Phase.READY,
        0,
        0,
        store.relative(model),
        store.relative(model),
        null,
        0,
        null,
        "disabled",
        CfrFiles.sha256(store.root().resolve("config.json")));
  }

  private static void delete(Path directory) throws IOException {
    try (var files = Files.walk(directory)) {
      for (Path path : files.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    }
  }
}
