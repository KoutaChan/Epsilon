package com.epsilon.agent;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * agent の依存の向きを確かめる。本体とテストのソースは grp・classic・prism を import せず、ワイルドカードの import も使わない。pom には
 * agent を使う側のモジュールを書かない。PRISM のクラスはクラスパスに無い。
 */
public class ImportDirectionTest {

  private static final List<Path> SOURCES = List.of(Path.of("src", "main", "java"), Path.of("src", "test", "java"));
  private static final Path POM = Path.of("pom.xml");
  private static final Pattern IMPORT =
      Pattern.compile("^import\\s+(?:static\\s+)?([\\w.]+?)(\\.\\*)?;", Pattern.MULTILINE);
  private static final Pattern ARTIFACT = Pattern.compile("<artifactId>\\s*([^<\\s]+)\\s*</artifactId>");

  /** agent を使う側（grp・classic・prism）のパッケージ。 */
  private static final Pattern FORBIDDEN = Pattern.compile("com\\.epsilon\\.(grp|classic|prism)(\\..*)?");

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
    List<String> violations = new ArrayList<>();
    for (Path root : SOURCES) {
      Assert.assertTrue(Files.isDirectory(root), "run from the epsilon-agent directory: " + root);
      try (Stream<Path> files = Files.walk(root)) {
        for (Path file : files.filter(path -> path.toString().endsWith(".java")).toList()) {
          Matcher matcher = IMPORT.matcher(Files.readString(file));
          while (matcher.find()) {
            if (matcher.group(2) != null) {
              violations.add(file + ": " + matcher.group() + " (wildcard)");
            } else if (FORBIDDEN.matcher(matcher.group(1)).matches()) {
              violations.add(file + ": " + matcher.group());
            }
          }
        }
      }
    }
    Assert.assertEquals(violations, List.of());
  }

  @Test
  public void pomNamesNoDependentModule() throws IOException {
    Assert.assertTrue(Files.isRegularFile(POM), "run from the epsilon-agent directory");
    List<String> named = new ArrayList<>();
    Matcher matcher = ARTIFACT.matcher(Files.readString(POM));
    while (matcher.find()) {
      if (FORBIDDEN_MODULES.contains(matcher.group(1))) {
        named.add(matcher.group(1));
      }
    }
    Assert.assertEquals(named, List.of());
  }

  @Test
  public void classpathHoldsNoDependentModule() {
    Assert.assertEquals(DEPENDENT_CLASSES.stream().filter(ImportDirectionTest::loadable).toList(), List.of());
  }

  private static boolean loadable(String className) {
    try {
      Class.forName(className, false, ImportDirectionTest.class.getClassLoader());
      return true;
    } catch (ClassNotFoundException absent) {
      return false;
    }
  }
}
