package com.epsilon.agent.app;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.expectThrows;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;
import org.testng.annotations.Test;

/** 公開したバージョンの探し方を確かめる。 */
public class RunFilesTest {

  @Test
  public void theLatestSnapshotIsTheNewestCompleteVersion() throws IOException {
    Path root = Files.createTempDirectory("run");
    try {
      expectThrows(IOException.class, () -> RunFiles.latestSnapshot(root, "policy.json"));
      for (int version : new int[] {9, 12, 13}) {
        Files.createDirectories(RunFiles.snapshot(root, version));
      }
      Files.createDirectories(root.resolve(RunFiles.SNAPSHOTS).resolve("notes"));
      expectThrows(IOException.class, () -> RunFiles.latestSnapshot(root, "policy.json"));
      Files.writeString(RunFiles.snapshot(root, 9).resolve("policy.json"), "{}");
      Files.writeString(RunFiles.snapshot(root, 12).resolve("policy.json"), "{}");
      assertEquals(RunFiles.snapshot(root, 12), root.resolve("snapshots").resolve("00012"));
      assertEquals(RunFiles.latestSnapshot(root, "policy.json"), RunFiles.snapshot(root, 12));
    } finally {
      delete(root);
    }
  }

  private static void delete(Path root) throws IOException {
    try (Stream<Path> files = Files.walk(root)) {
      for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(file);
      }
    }
  }
}
