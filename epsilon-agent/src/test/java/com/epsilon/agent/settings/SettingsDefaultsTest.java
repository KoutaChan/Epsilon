package com.epsilon.agent.settings;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.fail;

import com.epsilon.agent.app.Command;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.RuntimeSettings;
import java.io.File;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;
import org.testng.annotations.Test;

/**
 * このモジュールにある全ての設定の record について、全ての既定値が結び付けを通ることを 1 つのテストで確かめる。設定の record とは、
 * 全てのコンポーネントにキーの注釈を持つ record のこと。そのうち、ほかの設定の record から使われず、選ぶ形の 1 つでも、コマンドでも、全ての
 * キーが親に従う部品でもないものは、いちばん外側の record とみなす。いちばん外側の record では、どのキーも変えてよいときを親から
 * 受け継がずに自分で書いていることも確かめる。一部のコンポーネントだけにキーの注釈を持つ record は、プログラムが書くファイルとみなし、型を
 * 読めること（既定値が結び付けを通ること）を確かめる。
 */
public class SettingsDefaultsTest {

  @Test
  public void everyDefaultOfEveryRecordBinds() throws Exception {
    List<Class<? extends Record>> settings = new ArrayList<>();
    List<Class<? extends Record>> files = new ArrayList<>();
    for (Class<? extends Record> type : annotatedRecords()) {
      boolean everyKey = Arrays.stream(type.getRecordComponents()).allMatch(c -> c.isAnnotationPresent(Key.class));
      (everyKey ? settings : files).add(type);
    }
    assertTrue(
        settings.containsAll(List.of(InferenceSettings.class, TableSettings.class, RuntimeSettings.class)),
        settings.toString());

    Set<Class<?>> used = new HashSet<>();
    for (Class<? extends Record> type : settings) {
      for (RecordComponent component : type.getRecordComponents()) {
        collectUsed(component.getGenericType(), used);
      }
    }

    List<String> failures = new ArrayList<>();
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
      // 選ぶ形の 1 つ（制御の要求の形など）、コマンドの引数のセクション、全てのキーが親に従う部品（ほかのモジュールのセクションに
      // 置く GRP の学習の設定など）は、いちばん外側の record ではない
      boolean part =
          Arrays.stream(type.getRecordComponents())
              .allMatch(component -> component.getAnnotation(Key.class).change() == Change.INHERIT);
      boolean topLevel =
          !used.contains(type)
              && !type.isAnnotationPresent(Case.class)
              && !Command.class.isAssignableFrom(type)
              && !part;
      if (topLevel) {
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
  }

  /** 設定の型の中で使われている record を集める。選ぶ形（sealed interface）からは、形ごとの record を集める。 */
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

  /** このモジュールの本体のクラスから、キーの注釈を持つコンポーネントがある record を全て集める。 */
  @SuppressWarnings("unchecked")
  private static List<Class<? extends Record>> annotatedRecords() throws Exception {
    Path classes = Path.of(Settings.class.getProtectionDomain().getCodeSource().getLocation().toURI());
    assertTrue(Files.isDirectory(classes), classes.toString());
    List<Class<? extends Record>> records = new ArrayList<>();
    List<Path> paths;
    try (Stream<Path> walk = Files.walk(classes)) {
      paths = walk.filter(path -> path.toString().endsWith(".class")).sorted().toList();
    }
    for (Path path : paths) {
      String relative = classes.relativize(path).toString().replace(File.separatorChar, '.');
      String name = relative.substring(0, relative.length() - ".class".length());
      if (name.endsWith("module-info") || name.endsWith("package-info")) {
        continue;
      }
      Class<?> type;
      try {
        type = Class.forName(name, false, Settings.class.getClassLoader());
      } catch (LinkageError unloadable) {
        fail("Cannot load " + name, unloadable);
        continue;
      }
      if (type.isRecord()
          && Arrays.stream(type.getRecordComponents()).anyMatch(c -> c.isAnnotationPresent(Key.class))) {
        records.add((Class<? extends Record>) type);
      }
    }
    return records;
  }
}
