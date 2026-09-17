package com.epsilon.config.settings;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.RecordComponent;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** 一度読み込んだ設定値を所有する不変スナップショット。実行中の環境やシステムプロパティは参照しない。 */
public final class SettingsLoader {
  private final Map<String, String> values;
  private final Map<Class<?>, Object> bound = new ConcurrentHashMap<>();

  private SettingsLoader(Map<String, String> values) {
    this.values = Map.copyOf(values);
  }

  /** 呼び出し元のマップから独立したスナップショットを作る。 */
  public static SettingsLoader of(Map<String, String> values) {
    return new SettingsLoader(values);
  }

  /** 系列が指定した同梱既定値を読み込む。 */
  public static SettingsLoader fromResource(String resource) {
    try (InputStream stream = SettingsLoader.class.getClassLoader().getResourceAsStream(resource)) {
      if (stream == null) {
        throw new IllegalStateException("Bundled settings not found: " + resource);
      }
      String text = new String(stream.readAllBytes(), StandardCharsets.UTF_8);
      return parse(resource, text.lines().toList(), Map.of());
    } catch (IOException failure) {
      throw new UncheckedIOException("Failed to read bundled settings: " + resource, failure);
    }
  }

  /** 既定値、選択ファイル、明示上書き設定を一つのスナップショットへ合成する。 */
  public static SettingsLoader load(
      SettingsLoader defaults, Path path, Map<String, String> overrides) throws IOException {
    if (path == null) {
      return defaults.withOverrides(overrides);
    }
    Map<String, String> values = new LinkedHashMap<>(defaults.values);
    parseInto(path.toString(), Files.readAllLines(path, StandardCharsets.UTF_8), values);
    values.putAll(overrides);
    return new SettingsLoader(values);
  }

  /** このスナップショットへ明示上書き設定を重ねる。上書き設定が空なら同じスナップショットを使う。 */
  public SettingsLoader withOverrides(Map<String, String> overrides) {
    if (overrides.isEmpty()) {
      return this;
    }
    Map<String, String> merged = new LinkedHashMap<>(values);
    merged.putAll(overrides);
    return new SettingsLoader(merged);
  }

  /** ファイルを一度だけ読み、明示された上書き設定を重ねる。 */
  public static SettingsLoader load(Path path, Map<String, String> overrides) throws IOException {
    return parse(path.toString(), Files.readAllLines(path, StandardCharsets.UTF_8), overrides);
  }

  /** 同梱の既定値にも同じパーサーと優先順位を使う。 */
  public static SettingsLoader parse(
      String source, List<String> lines, Map<String, String> overrides) {
    Map<String, String> values = new LinkedHashMap<>();
    parseInto(source, lines, values);
    values.putAll(overrides);
    return new SettingsLoader(values);
  }

  /** 型付き設定の生成・検証はスナップショット内で一度だけ行い、不変レコードを共有する。 */
  public <T> T bind(Class<T> type) {
    return type.cast(bound.computeIfAbsent(type, this::bindUncached));
  }

  /**
   * 指定した設定キーの接頭辞からレコードを生成する。同じ設定型を異なる接頭辞で読み込める。
   *
   * <p>この経路はキャッシュせず、スナップショットの値をそのまま参照する。生成したレコードは呼び出し側が保持する。
   */
  public <T> T bind(Class<T> type, String prefix) {
    return bindRecord(type, prefix);
  }

  /** 接頭辞を含む設定キーと値の、変更できないマップを返す。 */
  public Map<String, String> values() {
    return values;
  }

  public static <T> T validate(T value) {
    if (value == null || !value.getClass().isRecord()) {
      throw new IllegalArgumentException("Settings validation requires a record value");
    }
    SettingsPrefix prefix = value.getClass().getAnnotation(SettingsPrefix.class);
    if (prefix == null || prefix.value().isBlank()) {
      throw new IllegalArgumentException(
          "Missing @SettingsPrefix on " + value.getClass().getName());
    }
    validateRecord(value, prefix.value().trim());
    return value;
  }

  private static void parseInto(String source, List<String> lines, Map<String, String> values) {
    String section = "";
    for (int i = 0; i < lines.size(); i++) {
      String line = stripComment(lines.get(i)).trim();
      if (line.isEmpty()) {
        continue;
      }
      if (line.startsWith("[") && line.endsWith("]")) {
        section = line.substring(1, line.length() - 1).trim();
        if (section.isEmpty()) {
          throw parseError(source, i, "empty section");
        }
        continue;
      }
      int equals = line.indexOf('=');
      if (equals <= 0) {
        throw parseError(source, i, "expected key = value");
      }
      String key = line.substring(0, equals).trim();
      String rawValue = line.substring(equals + 1).trim();
      if (key.isEmpty() || rawValue.isEmpty()) {
        throw parseError(source, i, "empty key or value");
      }
      String fullKey = section.isEmpty() ? key : section + "." + key;
      values.put(fullKey, scalarValue(rawValue));
    }
  }

  private static String stripComment(String line) {
    boolean inString = false;
    char quote = 0;
    for (int i = 0; i < line.length(); i++) {
      char c = line.charAt(i);
      if ((c == '"' || c == '\'') && (i == 0 || line.charAt(i - 1) != '\\')) {
        if (!inString) {
          inString = true;
          quote = c;
        } else if (quote == c) {
          inString = false;
        }
      } else if (c == '#' && !inString) {
        return line.substring(0, i);
      }
    }
    return line;
  }

  private static String scalarValue(String rawValue) {
    String value = rawValue.trim();
    if ((value.startsWith("\"") && value.endsWith("\""))
        || (value.startsWith("'") && value.endsWith("'"))) {
      return value.substring(1, value.length() - 1);
    }
    String lower = value.toLowerCase(Locale.ROOT);
    if ("true".equals(lower) || "false".equals(lower)) {
      return lower;
    }
    return value.replace("_", "");
  }

  private static IllegalArgumentException parseError(String source, int lineIndex, String detail) {
    return new IllegalArgumentException(
        "Invalid settings file " + source + " at line " + (lineIndex + 1) + ": " + detail);
  }

  private static IllegalArgumentException invalidSetting(
      String key, String value, String expected) {
    return new IllegalArgumentException(
        "Invalid setting " + key + "=" + value + "; expected " + expected);
  }

  private <T> T bindUncached(Class<T> type) {
    SettingsPrefix prefix = type.getAnnotation(SettingsPrefix.class);
    if (prefix == null || prefix.value().isBlank()) {
      throw new IllegalArgumentException("Missing @SettingsPrefix on " + type.getName());
    }
    return bindRecord(type, prefix.value().trim());
  }

  private <T> T bindRecord(Class<T> type, String prefix) {
    if (!type.isRecord()) {
      throw new IllegalArgumentException("Settings binding requires a record type: " + type);
    }
    RecordComponent[] components = type.getRecordComponents();
    Class<?>[] parameterTypes = new Class<?>[components.length];
    Object[] args = new Object[components.length];
    for (int i = 0; i < components.length; i++) {
      RecordComponent component = components[i];
      parameterTypes[i] = component.getType();
      args[i] = bindComponent(prefix, component);
    }
    try {
      Constructor<T> constructor = type.getDeclaredConstructor(parameterTypes);
      constructor.setAccessible(true);
      return constructor.newInstance(args);
    } catch (InvocationTargetException e) {
      Throwable cause = e.getCause();
      if (cause instanceof RuntimeException runtimeException) {
        throw runtimeException;
      }
      throw new IllegalStateException("Failed to bind settings record: " + type.getName(), cause);
    } catch (NoSuchMethodException | InstantiationException | IllegalAccessException e) {
      throw new IllegalStateException("Failed to bind settings record: " + type.getName(), e);
    }
  }

  private static void validateRecord(Object value, String prefix) {
    for (RecordComponent component : value.getClass().getRecordComponents()) {
      String key = settingKey(prefix, component);
      Object componentValue;
      try {
        var accessor = component.getAccessor();
        accessor.setAccessible(true);
        componentValue = accessor.invoke(value);
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException(
            "Failed to inspect settings record: " + value.getClass(), e);
      }
      if (component.getType().isRecord()) {
        if (componentValue == null) {
          throw invalidSetting(key, "null", "nested settings record");
        }
        validateRecord(componentValue, key);
      } else {
        validateValue(key, componentValue, component);
      }
    }
  }

  private Object bindComponent(String prefix, RecordComponent component) {
    Class<?> type = component.getType();
    if (type.isRecord()) {
      return bindNested(prefix, component);
    }

    String key = settingKey(prefix, component);
    String value = rawValue(key);
    Default defaultValue = component.getAnnotation(Default.class);
    if (value == null && defaultValue != null) {
      value = defaultValue.value();
    }
    if (value == null) {
      throw invalidSetting(key, "<missing>", "setting");
    }
    Object converted = convertValue(key, value, type);
    validateValue(key, converted, component);
    return converted;
  }

  private Object bindNested(String parentPrefix, RecordComponent component) {
    Class<?> type = component.getType();
    return bindRecord(type, settingKey(parentPrefix, component));
  }

  private static String settingKey(String prefix, RecordComponent component) {
    Setting setting = component.getAnnotation(Setting.class);
    String name =
        setting == null || setting.value().isBlank() ? component.getName() : setting.value();
    return prefix + "." + name;
  }

  private String rawValue(String key) {
    String value = values.get(key);
    return value == null ? null : normalizeRawValue(value);
  }

  private static String normalizeRawValue(String value) {
    String trimmed = value.trim();
    return trimmed.isEmpty() ? null : trimmed;
  }

  private static Object convertValue(String key, String value, Class<?> type) {
    if (type == String.class) {
      return value;
    }
    if (type == boolean.class || type == Boolean.class) {
      if ("true".equalsIgnoreCase(value) || "false".equalsIgnoreCase(value)) {
        return Boolean.parseBoolean(value);
      }
      throw invalidSetting(key, value, "boolean");
    }
    if (type == int.class || type == Integer.class) {
      try {
        return Integer.parseInt(value);
      } catch (NumberFormatException e) {
        throw invalidSetting(key, value, "integer");
      }
    }
    if (type == long.class || type == Long.class) {
      try {
        return Long.parseLong(value);
      } catch (NumberFormatException e) {
        throw invalidSetting(key, value, "long integer");
      }
    }
    if (type == float.class || type == Float.class) {
      try {
        return Float.parseFloat(value);
      } catch (NumberFormatException e) {
        throw invalidSetting(key, value, "float");
      }
    }
    if (type == double.class || type == Double.class) {
      try {
        return Double.parseDouble(value);
      } catch (NumberFormatException e) {
        throw invalidSetting(key, value, "double");
      }
    }
    if (type.isEnum()) {
      return enumValue(key, value, type);
    }
    throw invalidSetting(key, value, "supported setting type");
  }

  private static Object enumValue(String key, String value, Class<?> type) {
    String normalized = value.trim().toUpperCase(Locale.ROOT).replace('-', '_');
    Object[] constants = type.getEnumConstants();
    for (Object constant : constants) {
      if (((Enum<?>) constant).name().equals(normalized)) {
        return constant;
      }
    }
    throw invalidSetting(key, value, "one of " + enumNames(constants));
  }

  private static String enumNames(Object[] constants) {
    StringBuilder builder = new StringBuilder();
    for (int i = 0; i < constants.length; i++) {
      if (i > 0) {
        builder.append(", ");
      }
      builder.append(((Enum<?>) constants[i]).name());
    }
    return builder.toString();
  }

  private static void validateValue(String key, Object value, RecordComponent component) {
    if (value instanceof Number number) {
      if ((number instanceof Float floatValue && !Float.isFinite(floatValue))
          || (number instanceof Double doubleValue && !Double.isFinite(doubleValue))) {
        throw invalidSetting(key, value.toString(), "finite number");
      }
      double numeric = number.doubleValue();
      if (component.isAnnotationPresent(Positive.class) && numeric <= 0.0) {
        throw invalidSetting(key, value.toString(), "positive number");
      }
      if (component.isAnnotationPresent(NonNegative.class) && numeric < 0.0) {
        throw invalidSetting(key, value.toString(), "non-negative number");
      }
      Range range = component.getAnnotation(Range.class);
      if (range != null && outsideRange(number, range)) {
        throw invalidSetting(key, value.toString(), "number in " + rangeExpectation(range));
      }
      MultipleOf multipleOf = component.getAnnotation(MultipleOf.class);
      if (multipleOf != null) {
        if (multipleOf.value() <= 0) {
          throw new IllegalStateException(
              "Invalid @MultipleOf on " + component.getDeclaringRecord().getName());
        }
        if (!(number instanceof Byte
            || number instanceof Short
            || number instanceof Integer
            || number instanceof Long)) {
          throw new IllegalStateException(
              "@MultipleOf requires an integer component: "
                  + component.getDeclaringRecord().getName()
                  + "."
                  + component.getName());
        }
        if (number.longValue() % multipleOf.value() != 0) {
          throw invalidSetting(key, value.toString(), "integer multiple of " + multipleOf.value());
        }
      }
    }
  }

  private static boolean outsideRange(Number number, Range range) {
    if (!Double.isFinite(range.min())
        || !Double.isFinite(range.max())
        || range.min() > range.max()
        || (range.min() == range.max() && (!range.minInclusive() || !range.maxInclusive()))) {
      throw new IllegalStateException("Invalid @Range declaration");
    }
    int minimumComparison = compareToBound(number, range.min());
    int maximumComparison = compareToBound(number, range.max());
    boolean belowMinimum = range.minInclusive() ? minimumComparison < 0 : minimumComparison <= 0;
    boolean aboveMaximum = range.maxInclusive() ? maximumComparison > 0 : maximumComparison >= 0;
    return belowMinimum || aboveMaximum;
  }

  private static int compareToBound(Number number, double bound) {
    if (number instanceof Float floatValue) {
      return Float.compare(floatValue, (float) bound);
    }
    if (number instanceof Double doubleValue) {
      return Double.compare(doubleValue, bound);
    }
    return BigDecimal.valueOf(number.longValue()).compareTo(BigDecimal.valueOf(bound));
  }

  private static String rangeExpectation(Range range) {
    return (range.minInclusive() ? "[" : "(")
        + range.min()
        + ", "
        + range.max()
        + (range.maxInclusive() ? "]" : ")");
  }
}
