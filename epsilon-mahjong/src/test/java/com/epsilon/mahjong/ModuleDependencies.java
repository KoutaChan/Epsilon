package com.epsilon.mahjong;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import org.testng.Assert;

/** モジュールの import、pom、クラスパスの依存を調べる。 */
public final class ModuleDependencies {
  private static final Pattern IMPORT =
      Pattern.compile("^import\\s+(?:static\\s+)?([\\w.]+(?:\\.\\*)?)\\s*;", Pattern.MULTILINE);
  private static final Pattern ARTIFACT =
      Pattern.compile("<artifactId>\\s*([^<\\s]+)\\s*</artifactId>");

  private ModuleDependencies() {}

  public static void checkImports(Path root, Predicate<String> forbidden) throws IOException {
    var violations = new ArrayList<String>();
    try (var files = Files.walk(root)) {
      var sources = files.filter(p -> p.toString().endsWith(".java")).toList();
      Assert.assertFalse(sources.isEmpty(), root.toString());
      for (Path source : sources) {
        var matcher = IMPORT.matcher(Files.readString(source));
        while (matcher.find()) {
          String name = matcher.group(1);
          if (name.endsWith(".*") || forbidden.test(name)) violations.add(source + ": " + name);
        }
      }
    }
    Assert.assertEquals(violations, List.of());
  }

  public static void absentArtifacts(Set<String> forbidden) throws IOException {
    var matcher = ARTIFACT.matcher(Files.readString(Path.of("pom.xml")));
    while (matcher.find())
      Assert.assertFalse(forbidden.contains(matcher.group(1)), matcher.group(1));
  }

  public static void absentClasses(List<String> classes) {
    for (String name : classes)
      Assert.expectThrows(
          ClassNotFoundException.class,
          () -> Class.forName(name, false, ModuleDependencies.class.getClassLoader()));
  }
}
