package com.epsilon.mahjong.action;

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
 * mahjong のパッケージを層と見て、import が下の層にだけ向いていることと、モジュールの依存を確かめる。層は下から util → tile →
 * rules → action → shape → score → engine → record → io → client の順で、どの層も上の層を import しない。mahjong の外で
 * 使ってよいのは java と slf4j だけで、gson を使ってよいのは io と client だけ。ワイルドカードの import は使わない。親の pom からは
 * slf4j 以外のライブラリを受け継がない。
 */
public class ImportDirectionTest {

  private static final List<String> LAYERS =
      List.of("util", "tile", "rules", "action", "shape", "score", "engine", "record", "io", "client");
  private static final String MAHJONG = "com.epsilon.mahjong.";

  /** gson を import してよい層。 */
  private static final Set<String> JSON_LAYERS = Set.of("io", "client");

  private static final String GSON = "com.google.gson.";
  private static final Path SOURCES = Path.of("src", "main", "java", "com", "epsilon", "mahjong");
  private static final Pattern IMPORT =
      Pattern.compile("^import\\s+(?:static\\s+)?([\\w.]+?)(\\.\\*)?;", Pattern.MULTILINE);

  /**
   * 親の pom（epsilon-parent）が持つライブラリ（DJL・PyTorch・zstd・logback）から 1 つずつ選んだクラス。どれかがテストの
   * classpath にあれば、pom か親の pom から依存が入り込んでいる。gson は mahjong が自分の依存として持つ。
   */
  private static final List<String> FOREIGN_CLASSES =
      List.of(
          "ai.djl.engine.Engine",
          "ai.djl.pytorch.engine.PtEngine",
          "com.github.luben.zstd.Zstd",
          "ch.qos.logback.classic.Logger");

  @Test
  public void layersImportOnlyDownward() throws IOException {
    Assert.assertTrue(Files.isDirectory(SOURCES), "run from the epsilon-mahjong directory");
    List<String> violations = new ArrayList<>();
    for (int layer = 0; layer < LAYERS.size(); layer++) {
      try (Stream<Path> files = Files.walk(SOURCES.resolve(LAYERS.get(layer)))) {
        for (Path file : files.filter(path -> path.toString().endsWith(".java")).toList()) {
          Matcher matcher = IMPORT.matcher(Files.readString(file));
          while (matcher.find()) {
            String problem = problem(layer, matcher.group(1), matcher.group(2) != null);
            if (problem != null) {
              violations.add(file + ": " + matcher.group() + " (" + problem + ")");
            }
          }
        }
      }
    }
    Assert.assertEquals(violations, List.of());
  }

  @Test
  public void classpathHoldsNoInheritedLibrary() {
    Assert.assertEquals(FOREIGN_CLASSES.stream().filter(ImportDirectionTest::loadable).toList(), List.of());
  }

  private static String problem(int layer, String imported, boolean wildcard) {
    if (wildcard) {
      return "wildcard";
    }
    if (imported.startsWith("java.") || imported.startsWith("org.slf4j.")) {
      return null;
    }
    if (imported.startsWith(GSON)) {
      return JSON_LAYERS.contains(LAYERS.get(layer)) ? null : "gson outside " + JSON_LAYERS;
    }
    if (!imported.startsWith(MAHJONG)) {
      return "outside java, slf4j, gson and mahjong";
    }
    String target = imported.substring(MAHJONG.length()).split("\\.")[0];
    int targetLayer = LAYERS.indexOf(target);
    return targetLayer < 0 || targetLayer > layer ? "upward" : null;
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
