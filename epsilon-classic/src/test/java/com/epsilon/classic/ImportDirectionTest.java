package com.epsilon.classic;

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
 * classic の依存の向きを確かめる。main とテストのソースは PRISM を import せず、ワイルドカードの import も使わない。pom は PRISM・
 * Reviewer に依存しない。PRISM のクラスは classpath から見えない。
 */
public class ImportDirectionTest {

  private static final List<Path> SOURCES = List.of(Path.of("src", "main", "java"), Path.of("src", "test", "java"));
  private static final Path POM = Path.of("pom.xml");
  private static final Pattern IMPORT =
      Pattern.compile("^import\\s+(?:static\\s+)?([\\w.]+?)(\\.\\*)?;", Pattern.MULTILINE);
  private static final Pattern ARTIFACT = Pattern.compile("<artifactId>\\s*([^<\\s]+)\\s*</artifactId>");

  /** PRISM のパッケージ。 */
  private static final Pattern FORBIDDEN = Pattern.compile("com\\.epsilon\\.prism(\\..*)?");

  /** classic が使わないモジュールの artifactId。 */
  private static final Set<String> FORBIDDEN_MODULES = Set.of("epsilon-prism", "epsilon-prism-hip", "epsilon-reviewer");

  /** classpath に無いことを確かめる、PRISM の代表的なクラス。 */
  private static final List<String> PRISM_CLASSES = List.of("com.epsilon.prism.Limits");

  @Test
  public void importsNoPrism() throws IOException {
    List<String> violations = new ArrayList<>();
    for (Path root : SOURCES) {
      Assert.assertTrue(Files.isDirectory(root), "run from the epsilon-classic directory: " + root);
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
  public void pomNamesNoPrismNorReviewer() throws IOException {
    Assert.assertTrue(Files.isRegularFile(POM), "run from the epsilon-classic directory");
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
  public void classpathHoldsNoPrism() {
    Assert.assertEquals(PRISM_CLASSES.stream().filter(ImportDirectionTest::loadable).toList(), List.of());
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
