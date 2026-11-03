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
      Assert.assertFalse(files.recorded(RunFiles.EVALUATIONS, "version", 2));
      files.append(RunFiles.EVALUATIONS, Map.of("version", 2));
      files.append(RunFiles.EVALUATIONS, Map.of("version", 3));
      Path path = directory.resolve("reports").resolve(RunFiles.EVALUATIONS);
      Files.writeString(path, "{\"version\": 4, \"he", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
      Assert.assertTrue(files.recorded(RunFiles.EVALUATIONS, "version", 2));
      Assert.assertTrue(files.recorded(RunFiles.EVALUATIONS, "version", 3));
      Assert.assertFalse(files.recorded(RunFiles.EVALUATIONS, "version", 4));
      files.append(RunFiles.EVALUATIONS, Map.of("version", 4));
      Assert.assertEquals(
          Files.readAllLines(path),
          List.of("{\"version\":2}", "{\"version\":3}", "{\"version\": 4, \"he", "{\"version\":4}"));
      Assert.assertTrue(files.recorded(RunFiles.EVALUATIONS, "version", 4));
    } finally {
      Snapshot.deleteTree(directory);
    }
  }
}
