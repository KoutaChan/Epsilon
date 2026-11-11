package com.epsilon.reviewer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Reviewer の本体が PRISM と classic のパッケージを import しないことを確かめる（モデルは実行時に ServiceLoader で見つける）。
 * テストが import してよいのは、乱数の重みでチェックポイントを書いて開くためのクラスだけ。ワイルドカードの import は、本体でもテストでも認めない。
 */
public class ImportDirectionTest {
  private static final Pattern IMPORT =
      Pattern.compile("^import\\s+(?:static\\s+)?([\\w.]+(?:\\.\\*)?)\\s*;", Pattern.MULTILINE);

  private static final Pattern MODEL_MODULES =
      Pattern.compile("com\\.epsilon\\.(prism|prism4p|prism3p|classic)\\..*");

  /** テストが import してよい、チェックポイント用のヘルパー。 */
  private static final Set<String> TEST_HELPERS =
      Set.of(
          "com.epsilon.classic.RandomCheckpoints",
          "com.epsilon.classic.Architecture",
          "com.epsilon.classic.ClassicFamily",
          "com.epsilon.prism.input.InputFormat",
          "com.epsilon.prism.model.PrismNetwork",
          "com.epsilon.prism.model.PrismShape",
          "com.epsilon.prism.policy.PolicyParams",
          "com.epsilon.prism.run.Snapshot");

  @Test
  public void mainImportsNoModelModule() throws IOException {
    List<String> wrong = new ArrayList<>();
    for (Import found : imports(Path.of("src/main/java")))
      if (MODEL_MODULES.matcher(found.name).matches()
          || found.name.endsWith(".*")) wrong.add(found.toString());
    Assert.assertEquals(wrong, List.of());
  }

  @Test
  public void testsImportOnlyTheCheckpointHelpers() throws IOException {
    List<String> wrong = new ArrayList<>();
    for (Import found : imports(Path.of("src/test/java")))
      if ((MODEL_MODULES.matcher(found.name).matches() && !TEST_HELPERS.contains(found.name))
          || found.name.endsWith(".*")) wrong.add(found.toString());
    Assert.assertEquals(wrong, List.of());
  }

  private record Import(Path file, String name) {
    @Override
    public String toString() {
      return file.getFileName() + ": " + name;
    }
  }

  private static List<Import> imports(Path root) throws IOException {
    Assert.assertTrue(Files.isDirectory(root), root.toAbsolutePath().toString());
    List<Import> imports = new ArrayList<>();
    try (var files = Files.walk(root)) {
      for (Path file : files.filter(path -> path.toString().endsWith(".java")).toList()) {
        Matcher matcher = IMPORT.matcher(Files.readString(file, StandardCharsets.UTF_8));
        while (matcher.find()) imports.add(new Import(file, matcher.group(1)));
      }
    }
    Assert.assertFalse(imports.isEmpty(), "no imports under " + root);
    return imports;
  }
}
