package com.epsilon.reviewer.json;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * JsonObject を record へ厳密に読む。知らないフィールド、型の違い、整数に正確に直せない数は invalid_request にする（メッセージは
 * フィールドのパスを含む英語）。null はフィールドを省いたのと同じに扱う。
 *
 * <p>フィールドの型は、必須の String・int・boolean と、省いてよい Integer・Boolean・List&lt;…&gt;・入れ子の record（無ければ null）。
 * List の要素は省けない。既定値と、複数のフィールドにまたがる検査は、record のコンパクトコンストラクターが受け持つ。
 */
public final class StrictRecords {
  private static final ClassValue<Shape> SHAPES =
      new ClassValue<>() {
        @Override
        protected Shape computeValue(Class<?> type) {
          RecordComponent[] components = type.getRecordComponents();
          Class<?>[] types = new Class<?>[components.length];
          Set<String> names = new HashSet<>();
          for (int i = 0; i < components.length; i++) {
            types[i] = components[i].getType();
            names.add(components[i].getName());
          }
          try {
            Constructor<?> constructor = type.getDeclaredConstructor(types);
            constructor.setAccessible(true);
            return new Shape(components, constructor, Set.copyOf(names));
          } catch (NoSuchMethodException failure) {
            throw new IllegalStateException(failure);
          }
        }
      };

  private StrictRecords() {}

  /** @throws ReviewerException invalid_request */
  public static <R extends Record> R read(JsonObject object, Class<R> type) {
    return type.cast(record(object, type, null));
  }

  private static Object record(JsonObject object, Class<?> type, String path) {
    Shape shape = SHAPES.get(type);
    for (String key : object.keySet())
      if (!shape.names().contains(key))
        throw invalid(
            (path == null ? "The request" : path) + " has an unknown field: " + key + ".");
    Object[] values = new Object[shape.components().length];
    for (int i = 0; i < values.length; i++) {
      RecordComponent component = shape.components()[i];
      String name = path == null ? component.getName() : path + "." + component.getName();
      values[i] = value(object.get(component.getName()), component.getGenericType(), name);
    }
    try {
      return shape.constructor().newInstance(values);
    } catch (InvocationTargetException failure) {
      if (failure.getCause() instanceof RuntimeException cause) throw cause;
      throw new IllegalStateException(failure.getCause());
    } catch (ReflectiveOperationException failure) {
      throw new IllegalStateException(failure);
    }
  }

  private static Object value(JsonElement element, Type type, String path) {
    boolean absent = element == null || element.isJsonNull();
    if (absent) {
      if (type == int.class || type == boolean.class || type == String.class)
        throw invalid(path + " is required.");
      return null;
    }
    if (type == int.class || type == Integer.class) return integer(element, path);
    if (type == boolean.class || type == Boolean.class) {
      if (!element.isJsonPrimitive() || !element.getAsJsonPrimitive().isBoolean())
        throw invalid(path + " must be a boolean.");
      return element.getAsBoolean();
    }
    if (type == String.class) {
      if (!element.isJsonPrimitive() || !element.getAsJsonPrimitive().isString())
        throw invalid(path + " must be a string.");
      return element.getAsString();
    }
    if (type instanceof ParameterizedType list && list.getRawType() == List.class) {
      if (!element.isJsonArray()) throw invalid(path + " must be an array.");
      JsonArray array = element.getAsJsonArray();
      Type item = list.getActualTypeArguments()[0];
      List<Object> values = new ArrayList<>(array.size());
      for (int i = 0; i < array.size(); i++) {
        String itemPath = path + "[" + i + "]";
        if (array.get(i).isJsonNull()) throw invalid(itemPath + " is required.");
        values.add(value(array.get(i), item, itemPath));
      }
      return List.copyOf(values);
    }
    if (type instanceof Class<?> nested && nested.isRecord()) {
      if (!element.isJsonObject()) throw invalid(path + " must be an object.");
      return record(element.getAsJsonObject(), nested, path);
    }
    throw new IllegalArgumentException("Unsupported request field type: " + type);
  }

  private static int integer(JsonElement element, String path) {
    if (!element.isJsonPrimitive() || !element.getAsJsonPrimitive().isNumber())
      throw invalid(path + " must be an integer.");
    try {
      BigDecimal number = element.getAsBigDecimal();
      return number.intValueExact();
    } catch (ArithmeticException | NumberFormatException failure) {
      throw invalid(path + " must be an integer.");
    }
  }

  private static ReviewerException invalid(String message) {
    return new ReviewerException(ErrorCode.INVALID_REQUEST, message);
  }

  /** record のフィールドと標準コンストラクター（型ごとに 1 回だけ調べる）。 */
  private record Shape(
      RecordComponent[] components, Constructor<?> constructor, Set<String> names) {}
}
