package com.epsilon.classic;

import com.epsilon.mahjong.ModuleDependencies;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import org.testng.annotations.Test;

/**
 * classic の依存の向きを確かめる。main とテストのソースは PRISM を import せず、ワイルドカードの import も使わない。pom は PRISM・
 * Reviewer に依存しない。PRISM のクラスは classpath から見えない。
 */
public class ImportDirectionTest {

  private static final List<Path> SOURCES =
      List.of(Path.of("src", "main", "java"), Path.of("src", "test", "java"));

  /** PRISM のパッケージ。 */
  private static final Pattern FORBIDDEN = Pattern.compile("com\\.epsilon\\.prism(\\..*)?");

  /** classic が使わないモジュールの artifactId。 */
  private static final Set<String> FORBIDDEN_MODULES =
      Set.of(
          "epsilon-prism",
          "epsilon-prism-kernel-common",
          "epsilon-prism-hip",
          "epsilon-prism-cuda",
          "epsilon-reviewer");

  /** classpath に無いことを確かめる、PRISM の代表的なクラス。 */
  private static final List<String> PRISM_CLASSES = List.of("com.epsilon.prism.Limits");

  @Test
  public void importsNoPrism() throws IOException {
    for (Path root : SOURCES) ModuleDependencies.checkImports(root, FORBIDDEN.asMatchPredicate());
  }

  @Test
  public void pomNamesNoPrismNorReviewer() throws IOException {
    ModuleDependencies.absentArtifacts(FORBIDDEN_MODULES);
  }

  @Test
  public void classpathHoldsNoPrism() {
    ModuleDependencies.absentClasses(PRISM_CLASSES);
  }
}
