package com.epsilon.io;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

public class AtomicFilesTest {
  @Test
  public void failedWriterPreservesPublishedFileAndRemovesTemporary() throws Exception {
    Path directory = Files.createTempDirectory("epsilon-atomic-");
    Path file = directory.resolve("state.bin");
    AtomicFiles.write(file, new byte[] {1, 2, 3});
    Assert.expectThrows(
        IOException.class,
        () ->
            AtomicFiles.write(
                file,
                stream -> {
                  stream.write(new byte[] {4, 5});
                  throw new IOException("interrupted serialization");
                }));
    Assert.assertEquals(Files.readAllBytes(file), new byte[] {1, 2, 3});
    try (var entries = Files.list(directory)) {
      Assert.assertEquals(entries.count(), 1L);
    }
    AtomicFiles.write(file, new byte[] {6, 7});
    Assert.assertEquals(Files.readAllBytes(file), new byte[] {6, 7});
  }
}
