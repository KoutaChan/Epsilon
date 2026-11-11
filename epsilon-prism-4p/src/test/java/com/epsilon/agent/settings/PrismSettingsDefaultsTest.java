package com.epsilon.agent.settings;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.fail;

import com.epsilon.agent.app.Command;
import com.epsilon.prism.settings.PrismSettings;
import com.epsilon.prism4p.variant.FourPlayer;
import java.io.IOException;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.net.URI;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import org.testng.annotations.Test;

/**
 * PRISM（共通部分と四人打ち）の全ての設定の record が、既定値だけで読み込めることを 1 つのテストで確かめる。
 * epsilon-agent の {@link SettingsDefaultsTest} と同じことを PRISM の設定で確かめる。
 *
 * <p>全てのコンポーネントにキーのアノテーションがある record は設定の record とみなす。一部のコンポーネントだけにある record は、
 * プログラムが書くファイルとみなす。ほかの設定の record から使われず、選択肢の 1 つ（{@code @Case} の record）でもコマンドでもない
 * 設定の record は、いちばん外の record になる。そのキーは全て、いつ変えてよいかを自分で書く（{@code INHERIT} は不可）。
 * {@code Schema} を使うので、epsilon-agent の設定のパッケージに置く。
 */
public class PrismSettingsDefaultsTest {

  @Test
  public void everyDefaultOfEveryPrismRecordBinds() throws Exception {
    List<Class<? extends Record>> settings = new ArrayList<>();
    List<Class<? extends Record>> files = new ArrayList<>();
    for (Class<?> origin : List.of(PrismSettings.class, FourPlayer.class)) {
      for (Class<? extends Record> type : annotatedRecords(origin)) {
        boolean everyKey = Arrays.stream(type.getRecordComponents()).allMatch(c -> c.isAnnotationPresent(Key.class));
        (everyKey ? settings : files).add(type);
      }
    }
    assertTrue(settings.contains(PrismSettings.class), settings.toString());
    assertTrue(settings.size() > 40, "found only " + settings.size() + " settings records");

    Set<Class<?>> used = new HashSet<>();
    for (Class<? extends Record> type : settings) {
      for (RecordComponent component : type.getRecordComponents()) {
        collectUsed(component.getGenericType(), used);
      }
    }
    List<String> failures = new ArrayList<>();
    List<String> topLevel = new ArrayList<>();
    for (Class<? extends Record> type : settings) {
      try {
        Settings.describe(type);
        Settings.section(type, "section", List.of());
      } catch (SettingsException missing) {
        missing.problems().stream()
            .filter(problem -> !problem.startsWith("Missing setting section."))
            .forEach(problem -> failures.add(type.getName() + ": " + problem));
      } catch (RuntimeException broken) {
        failures.add(type.getName() + ": " + broken);
      }
      if (!used.contains(type) && !type.isAnnotationPresent(Case.class) && !Command.class.isAssignableFrom(type)) {
        topLevel.add(type.getName());
        for (RecordComponent component : type.getRecordComponents()) {
          if (component.getAnnotation(Key.class).change() == Change.INHERIT) {
            failures.add(type.getName() + "." + component.getName() + " is a top-level key that inherits its change");
          }
        }
      }
    }
    for (Class<? extends Record> type : files) {
      try {
        Schema.record(type);
      } catch (RuntimeException broken) {
        failures.add(type.getName() + ": " + broken);
      }
    }
    assertEquals(failures, List.of());
    assertTrue(topLevel.contains(PrismSettings.class.getName()), topLevel.toString());
  }

  /** 設定の型の中で使われている record を集める。選択肢の sealed interface なら、それを実装する record を集める。 */
  private static void collectUsed(Type type, Set<Class<?>> used) {
    if (type instanceof ParameterizedType parameterized) {
      for (Type argument : parameterized.getActualTypeArguments()) {
        collectUsed(argument, used);
      }
    } else if (type instanceof Class<?> plain) {
      if (plain.isRecord()) {
        used.add(plain);
      } else if (plain.isInterface() && plain.isSealed()) {
        used.addAll(Arrays.asList(plain.getPermittedSubclasses()));
      }
    }
  }

  /**
   * {@code origin} と同じ場所（ディレクトリか jar）にあるクラスから、キーのアノテーションを持つコンポーネントがある record を集める。
   */
  @SuppressWarnings("unchecked")
  private static List<Class<? extends Record>> annotatedRecords(Class<?> origin) throws Exception {
    Path location = Path.of(origin.getProtectionDomain().getCodeSource().getLocation().toURI());
    List<Class<? extends Record>> records = new ArrayList<>();
    for (String name : classNames(location)) {
      Class<?> type;
      try {
        type = Class.forName(name, false, origin.getClassLoader());
      } catch (LinkageError unloadable) {
        fail("Cannot load " + name, unloadable);
        continue;
      }
      if (type.isRecord() && Arrays.stream(type.getRecordComponents()).anyMatch(c -> c.isAnnotationPresent(Key.class))) {
        records.add((Class<? extends Record>) type);
      }
    }
    return records;
  }

  /** クラスのディレクトリか jar の中にある、com で始まるクラスの名前を名前の順に返す。 */
  private static List<String> classNames(Path location) throws IOException {
    if (Files.isDirectory(location)) {
      return classNames(location, location);
    }
    try (FileSystem jar = FileSystems.newFileSystem(URI.create("jar:" + location.toUri()), Map.of())) {
      return classNames(jar.getPath("/"), jar.getPath("/"));
    }
  }

  private static List<String> classNames(Path root, Path start) throws IOException {
    try (Stream<Path> walk = Files.walk(start)) {
      return walk.map(path -> root.relativize(path).toString())
          .filter(name -> name.endsWith(".class") && name.startsWith("com"))
          .map(name -> name.substring(0, name.length() - ".class".length()).replace('/', '.').replace('\\', '.'))
          .filter(name -> !name.endsWith("module-info") && !name.endsWith("package-info"))
          .sorted()
          .toList();
    }
  }
}
