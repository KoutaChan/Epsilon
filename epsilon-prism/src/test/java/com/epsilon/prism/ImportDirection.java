package com.epsilon.prism;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/** PRISM のモジュール（prism・prism-4p）の依存の向きを調べるテスト用のヘルパー。テストはモジュールのディレクトリで実行する。 */
public final class ImportDirection {

  private static final List<Path> SOURCES = List.of(Path.of("src", "main", "java"), Path.of("src", "test", "java"));
  private static final Path POM = Path.of("pom.xml");
  private static final Pattern IMPORT =
      Pattern.compile("^import\\s+(?:static\\s+)?([\\w.]+?)(\\.\\*)?;", Pattern.MULTILINE);
  /** 完全修飾で書いた epsilon のパッケージやクラスの名前。 */
  private static final Pattern QUALIFIED = Pattern.compile("\\bcom\\.epsilon(?:\\.\\w+)+");

  private static final Pattern DEPENDENCY = Pattern.compile("<dependency>(.*?)</dependency>", Pattern.DOTALL);
  private static final Pattern ARTIFACT = Pattern.compile("<artifactId>\\s*([^<\\s]+)\\s*</artifactId>");
  private static final Pattern SCOPE = Pattern.compile("<scope>\\s*([^<\\s]+)\\s*</scope>");

  /** 実行時にだけ class path に置く classic のパッケージを表す正規表現。 */
  public static final String CLASSIC_PACKAGES = "com\\.epsilon\\.classic(\\..*)?";

  /** classic のクラスは直接使わず、ModelFamily の実装として実行時に見つける。そのため classic は runtime の scope に置く。 */
  private static final String CLASSIC = "epsilon-classic";

  private ImportDirection() {}

  /**
   * main とテストのソースの import のうち、禁止したパッケージに当たるものと、ワイルドカードを使ったものを返す。
   *
   * @param forbidden 禁止するパッケージの正規表現（import した名前の全体と照合する）
   * @return 違反した import の一覧
   */
  public static List<String> violations(String forbidden) throws IOException {
    Pattern pattern = Pattern.compile(forbidden);
    List<String> violations = new ArrayList<>();
    for (Path root : SOURCES) {
      if (!Files.isDirectory(root)) throw new IllegalStateException("run from the module directory: " + root);
      try (Stream<Path> files = Files.walk(root)) {
        for (Path file : files.filter(path -> path.toString().endsWith(".java")).toList()) {
          Matcher matcher = IMPORT.matcher(Files.readString(file));
          while (matcher.find()) {
            if (matcher.group(2) != null) {
              violations.add(file + ": " + matcher.group() + " (wildcard)");
            } else if (pattern.matcher(matcher.group(1)).matches()) {
              violations.add(file + ": " + matcher.group());
            }
          }
        }
      }
    }
    return violations;
  }

  /**
   * main のソースで、1 つのパッケージ（その下のパッケージを含む）のファイルが、禁止したパッケージの名前を書いている所を返す。import の
   * ほかに、完全修飾の名前で書いた参照（コードの中の参照と Javadoc の {@code @link}）も数える。
   *
   * @param packageName 調べるパッケージ（例: {@code com.epsilon.prism.luck}）
   * @param forbidden 禁止するパッケージの正規表現（書いた名前の全体と照合する）
   * @return 違反した名前の一覧（ファイルと行の番号つき）
   */
  public static List<String> packageViolations(String packageName, String forbidden) throws IOException {
    Path root = SOURCES.get(0).resolve(packageName.replace('.', '/'));
    if (!Files.isDirectory(root)) throw new IllegalStateException("no such package in the main sources: " + root);
    Pattern pattern = Pattern.compile(forbidden);
    List<String> violations = new ArrayList<>();
    try (Stream<Path> files = Files.walk(root)) {
      for (Path file : files.filter(path -> path.toString().endsWith(".java")).sorted().toList()) {
        List<String> lines = Files.readAllLines(file);
        for (int i = 0; i < lines.size(); i++) {
          Matcher matcher = QUALIFIED.matcher(lines.get(i));
          while (matcher.find())
            if (pattern.matcher(matcher.group()).matches())
              violations.add(file + ":" + (i + 1) + ": " + matcher.group());
        }
      }
    }
    return violations;
  }

  /** pom の依存のうち、classic を runtime 以外の scope に置いているものを返す。 */
  public static List<String> pomViolations() throws IOException {
    if (!Files.isRegularFile(POM)) throw new IllegalStateException("run from the module directory");
    List<String> violations = new ArrayList<>();
    Matcher dependency = DEPENDENCY.matcher(Files.readString(POM));
    while (dependency.find()) {
      Matcher artifact = ARTIFACT.matcher(dependency.group(1));
      if (!artifact.find()) continue;
      String name = artifact.group(1);
      Matcher scope = SCOPE.matcher(dependency.group(1));
      String scoped = scope.find() ? scope.group(1) : "compile";
      if (name.equals(CLASSIC) && !scoped.equals("runtime")) violations.add(name + " in scope " + scoped);
    }
    return violations;
  }

  /** 与えた名前のうち、class path から読めるクラスの名前を返す。 */
  public static List<String> loadable(List<String> classNames) {
    return classNames.stream().filter(ImportDirection::loadable).toList();
  }

  private static boolean loadable(String className) {
    try {
      Class.forName(className, false, ImportDirection.class.getClassLoader());
      return true;
    } catch (ClassNotFoundException absent) {
      return false;
    }
  }
}
