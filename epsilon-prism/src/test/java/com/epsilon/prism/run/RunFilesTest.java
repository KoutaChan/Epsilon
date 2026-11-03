package com.epsilon.prism.run;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/** run の記録: 番号で引き（書きかけの行は読み飛ばす）、書きかけの行の後の追記は新しい行に書く。 */
public class RunFilesTest {

  @Test
  public void reportsAreLookedUpByNumberAndAppendedOnAFreshLine() throws IOException {
    Path directory = Files.createTempDirectory("prism-run-files");
    try {
      var files = new RunFiles(directory);
      Assert.assertFalse(files.recorded(RunFiles.ITERATIONS, "iteration", 2));
      files.append(RunFiles.ITERATIONS, Map.of("iteration", 2));
      files.append(RunFiles.ITERATIONS, Map.of("iteration", 3));
      Path path = directory.resolve("reports").resolve(RunFiles.ITERATIONS);
      Files.writeString(path, "{\"iteration\": 4, \"he", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
      Assert.assertTrue(files.recorded(RunFiles.ITERATIONS, "iteration", 2));
      Assert.assertTrue(files.recorded(RunFiles.ITERATIONS, "iteration", 3));
      Assert.assertFalse(files.recorded(RunFiles.ITERATIONS, "iteration", 4));
      files.append(RunFiles.ITERATIONS, Map.of("iteration", 4));
      Assert.assertEquals(
          Files.readAllLines(path),
          List.of("{\"iteration\":2}", "{\"iteration\":3}", "{\"iteration\": 4, \"he", "{\"iteration\":4}"));
      Assert.assertTrue(files.recorded(RunFiles.ITERATIONS, "iteration", 4));
    } finally {
      Snapshot.deleteTree(directory);
    }
  }
}
