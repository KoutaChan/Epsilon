package com.epsilon.workflow;

import com.epsilon.config.settings.SettingsLoader;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/** コマンド名を除いた位置引数を一貫したエラー形式で型変換する。 */
public final class WorkflowArguments {

  private final String command;
  private final SettingsLoader settings;
  private final List<String> values;

  public WorkflowArguments(String command, String[] values, SettingsLoader settings) {
    this.settings = settings;
    this.command = command;
    this.values = List.copyOf(Arrays.asList(values));
  }

  public SettingsLoader settings() {
    return settings;
  }

  public int size() {
    return values.size();
  }

  public String text(int index, String name) {
    if (index < 0 || index >= values.size()) {
      throw new IllegalArgumentException(
          command + " is missing required argument " + name + " at position " + (index + 1));
    }
    String value = values.get(index);
    if (value.isBlank()) {
      throw invalid(name, value, "a non-blank value", null);
    }
    return value;
  }

  public String optionalText(int index, String fallback) {
    return index < values.size() ? values.get(index) : fallback;
  }

  public Path path(int index, String name) {
    return Path.of(text(index, name));
  }

  public Path optionalPath(int index, Path fallback) {
    return index < values.size() ? Path.of(values.get(index)) : fallback;
  }

  public int integer(int index, String name) {
    return parse(index, name, "an integer", Integer::parseInt);
  }

  public int optionalInteger(int index, String name, int fallback) {
    return index < values.size() ? integer(index, name) : fallback;
  }

  public long longValue(int index, String name) {
    return parse(index, name, "a long integer", Long::parseLong);
  }

  public long optionalLong(int index, String name, long fallback) {
    return index < values.size() ? longValue(index, name) : fallback;
  }

  public double doubleValue(int index, String name) {
    return parse(index, name, "a finite number", WorkflowArguments::finiteDouble);
  }

  public double optionalDouble(int index, String name, double fallback) {
    return index < values.size() ? doubleValue(index, name) : fallback;
  }

  public float floatValue(int index, String name) {
    return parse(index, name, "a finite number", WorkflowArguments::finiteFloat);
  }

  public <E extends Enum<E>> E optionalEnum(int index, String name, Class<E> type, E fallback) {
    if (index >= values.size()) {
      return fallback;
    }
    String value = values.get(index);
    try {
      return Enum.valueOf(type, value.trim().toUpperCase(Locale.ROOT));
    } catch (IllegalArgumentException error) {
      throw invalid(name, value, type.getSimpleName(), error);
    }
  }

  public List<String> remaining(int startIndex) {
    if (startIndex < 0 || startIndex > values.size()) {
      throw new IndexOutOfBoundsException("startIndex=" + startIndex);
    }
    return values.subList(startIndex, values.size());
  }

  private <T> T parse(int index, String name, String expected, Parser<T> parser) {
    String value = text(index, name);
    try {
      return parser.parse(value);
    } catch (RuntimeException error) {
      throw invalid(name, value, expected, error);
    }
  }

  private IllegalArgumentException invalid(
      String name, String value, String expected, RuntimeException cause) {
    return new IllegalArgumentException(
        command + " argument " + name + " must be " + expected + ": " + value, cause);
  }

  private static double finiteDouble(String value) {
    double parsed = Double.parseDouble(value);
    if (!Double.isFinite(parsed)) {
      throw new NumberFormatException("non-finite double");
    }
    return parsed;
  }

  private static float finiteFloat(String value) {
    float parsed = Float.parseFloat(value);
    if (!Float.isFinite(parsed)) {
      throw new NumberFormatException("non-finite float");
    }
    return parsed;
  }

  @FunctionalInterface
  private interface Parser<T> {
    T parse(String value);
  }
}
