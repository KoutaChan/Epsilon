package com.epsilon.base.io;

import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class AtomicFilesTest {
  @Test
  public void failedWriterLeavesTheOtherWritersCompleteFile() throws IOException {
    var directory = Files.createTempDirectory("atomic-files");
    var target = directory.resolve("output");
    try {
      Assert.expectThrows(
          IOException.class,
          () ->
              AtomicFiles.replace(
                  target,
                  first -> {
                    first.write(1);
                    AtomicFiles.replace(target, second -> second.write(2));
                    throw new IOException("Interrupted");
                  }));
      Assert.assertEquals(Files.readAllBytes(target), new byte[] {2});
      try (var entries = Files.list(directory)) {
        Assert.assertEquals(entries.toList(), List.of(target));
      }
    } finally {
      Files.deleteIfExists(target);
      Files.delete(directory);
    }
  }
}
