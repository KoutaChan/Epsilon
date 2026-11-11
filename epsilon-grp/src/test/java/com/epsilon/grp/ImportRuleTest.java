package com.epsilon.grp;

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
 * epsilon-grp の import と依存の決まりを確かめる。本体が import するのは java・slf4j・gson・mahjong・grp だけで、DJL（ai.djl）を
 * import するのは、本体でもテストでも grp.train の下だけ。ワイルドカードの import は使わない。pom には PRISM のモジュールを書かない。
 */
public class ImportRuleTest {

  private static final Path MAIN = Path.of("src", "main", "java");
  private static final Path TEST = Path.of("src", "test", "java");
  private static final Path TRAIN = Path.of("com", "epsilon", "grp", "train");
  private static final Path POM = Path.of("pom.xml");
  private static final Pattern IMPORT =
      Pattern.compile("^import\\s+(?:static\\s+)?([\\w.]+?)(\\.\\*)?;", Pattern.MULTILINE);
  private static final Pattern ARTIFACT = Pattern.compile("<artifactId>\\s*([^<\\s]+)\\s*</artifactId>");
  private static final List<String> MAIN_IMPORTS =
      List.of("java.", "org.slf4j.", "com.google.gson.", "com.epsilon.mahjong.", "com.epsilon.grp.");
  private static final String DJL = "ai.djl.";
  private static final Pattern DJL_NAME = Pattern.compile("\\bai\\.djl\\.[A-Za-z]");

  /** PRISM のモジュールの artifactId。 */
  private static final Set<String> PRISM_MODULES = Set.of("epsilon-prism", "epsilon-prism-hip");

  @Test
  public void mainImportsStayInsideTheAllowedLibraries() throws IOException {
    Assert.assertTrue(Files.isDirectory(MAIN), "run from the epsilon-grp directory");
    List<String> violations = new ArrayList<>();
    for (Path file : javaFiles(MAIN)) {
      boolean train = MAIN.relativize(file).startsWith(TRAIN);
      Matcher matcher = IMPORT.matcher(Files.readString(file));
      while (matcher.find()) {
        String imported = matcher.group(1);
        boolean allowed = MAIN_IMPORTS.stream().anyMatch(imported::startsWith) || train && imported.startsWith(DJL);
        if (matcher.group(2) != null || !allowed) {
          violations.add(file + ": " + matcher.group());
        }
      }
    }
    Assert.assertEquals(violations, List.of());
  }

  @Test
  public void onlyTrainingImportsDjl() throws IOException {
    List<String> violations = new ArrayList<>();
    for (Path root : List.of(MAIN, TEST)) {
      for (Path file : javaFiles(root)) {
        Matcher matcher = IMPORT.matcher(Files.readString(file));
        while (matcher.find()) {
          boolean djl = matcher.group(1).startsWith(DJL);
          if (matcher.group(2) != null || djl && !root.relativize(file).startsWith(TRAIN)) {
            violations.add(file + ": " + matcher.group());
          }
        }
      }
    }
    Assert.assertEquals(violations, List.of());
  }

  /** 完全修飾の名前でも、DJL の型を書くのは grp.train の下だけ。 */
  @Test
  public void onlyTrainingNamesDjl() throws IOException {
    List<String> violations = new ArrayList<>();
    for (Path root : List.of(MAIN, TEST)) {
      for (Path file : javaFiles(root)) {
        if (!root.relativize(file).startsWith(TRAIN) && DJL_NAME.matcher(Files.readString(file)).find()) {
          violations.add(file.toString());
        }
      }
    }
    Assert.assertEquals(violations, List.of());
  }

  @Test
  public void pomNamesNoPrismModule() throws IOException {
    Assert.assertTrue(Files.isRegularFile(POM), "run from the epsilon-grp directory");
    List<String> named = new ArrayList<>();
    Matcher matcher = ARTIFACT.matcher(Files.readString(POM));
    while (matcher.find()) {
      if (PRISM_MODULES.contains(matcher.group(1))) {
        named.add(matcher.group(1));
      }
    }
    Assert.assertEquals(named, List.of());
  }

  private static List<Path> javaFiles(Path root) throws IOException {
    try (Stream<Path> files = Files.walk(root)) {
      return files.filter(path -> path.toString().endsWith(".java")).toList();
    }
  }
}
