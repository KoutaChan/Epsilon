package com.epsilon.reviewer;

import com.epsilon.mahjong.ModuleDependencies;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Set;
import java.util.regex.Pattern;
import org.testng.annotations.Test;

/**
 * Reviewer の本体が PRISM と classic のパッケージを import しないことを確かめる（モデルは実行時に ServiceLoader で見つける）。
 * テストが import してよいのは、乱数の重みでチェックポイントを書いて開くためのクラスだけ。ワイルドカードの import は、本体でもテストでも認めない。
 */
public class ImportDirectionTest {
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
    ModuleDependencies.checkImports(Path.of("src/main/java"), MODEL_MODULES.asMatchPredicate());
  }

  @Test
  public void testsImportOnlyTheCheckpointHelpers() throws IOException {
    ModuleDependencies.checkImports(
        Path.of("src/test/java"),
        name -> MODEL_MODULES.matcher(name).matches() && !TEST_HELPERS.contains(name));
  }
}
