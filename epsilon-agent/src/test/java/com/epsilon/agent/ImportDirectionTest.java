package com.epsilon.agent;

import com.epsilon.mahjong.ModuleDependencies;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import org.testng.annotations.Test;

/**
 * agent の依存の向きを確かめる。本体とテストのソースは grp・classic・prism を import せず、ワイルドカードの import も使わない。pom には
 * agent を使う側のモジュールを書かない。PRISM のクラスはクラスパスに無い。
 */
public class ImportDirectionTest {

  private static final List<Path> SOURCES =
      List.of(Path.of("src", "main", "java"), Path.of("src", "test", "java"));

  /** agent を使う側（grp・classic・prism）のパッケージ。 */
  private static final Pattern FORBIDDEN =
      Pattern.compile("com\\.epsilon\\.(grp|classic|prism)(\\..*)?");

  /** agent を使う側のモジュールの artifactId。 */
  private static final Set<String> FORBIDDEN_MODULES =
      Set.of(
          "epsilon-grp",
          "epsilon-classic",
          "epsilon-prism",
          "epsilon-prism-hip",
          "epsilon-reviewer");

  /** PRISM を代表するクラス。 */
  private static final List<String> DEPENDENT_CLASSES = List.of("com.epsilon.prism.Limits");

  @Test
  public void importsNoDependentPackage() throws IOException {
    for (Path root : SOURCES) ModuleDependencies.checkImports(root, FORBIDDEN.asMatchPredicate());
  }

  @Test
  public void pomNamesNoDependentModule() throws IOException {
    ModuleDependencies.absentArtifacts(FORBIDDEN_MODULES);
  }

  @Test
  public void classpathHoldsNoDependentModule() {
    ModuleDependencies.absentClasses(DEPENDENT_CLASSES);
  }
}
