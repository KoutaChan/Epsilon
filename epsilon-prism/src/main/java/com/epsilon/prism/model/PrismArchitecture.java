package com.epsilon.prism.model;

import com.google.gson.Gson;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;

/** ネットワークの形式。model.json の architecture に名前を書く。形の型（{@link PrismShape#architecture()}）が形式を決める。 */
public enum PrismArchitecture {
  /** 型ごとの FFN を持たない形（{@link PrismShape.V2}）。 */
  V2("epsilon-prism-v2", PrismShape.V2.class),
  /** 型ごとの FFN を持つ形（{@link PrismShape.V3}）。 */
  V3("epsilon-prism-v3", PrismShape.V3.class),
  /** ΣC と K を備え、幅と深さを指定する V4。 */
  V4("epsilon-prism-v4-core", PrismShape.V4.class);

  private final String id;
  private final Class<? extends PrismShape> type;
  private final Set<String> fields;

  PrismArchitecture(String id, Class<? extends PrismShape> type) {
    this.id = id;
    this.type = type;
    fields = fieldNames(type);
  }

  private static Set<String> fieldNames(Class<?> type) {
    return Arrays.stream(type.getRecordComponents())
        .map(RecordComponent::getName)
        .collect(Collectors.toUnmodifiableSet());
  }

  /** 学習計画に保存する形。各 record の成分名の集合は互いに異なる。 */
  public static final class ShapeJson
      implements JsonSerializer<PrismShape>, JsonDeserializer<PrismShape> {
    @Override
    public JsonElement serialize(PrismShape shape, Type type, JsonSerializationContext context) {
      return context.serialize(shape, shape.getClass());
    }

    @Override
    public PrismShape deserialize(JsonElement json, Type type, JsonDeserializationContext context) {
      JsonObject record = json.getAsJsonObject();
      for (PrismArchitecture architecture : values()) {
        if (architecture.matchesFields(record)) {
          architecture.validate(record);
          return context.deserialize(record, architecture.type);
        }
      }
      throw new JsonParseException("Unknown shape fields: " + record.keySet());
    }
  }

  /** 保存した形の成分を record の型と照合する。 */
  private void validate(JsonObject record) {
    for (RecordComponent field : type.getRecordComponents()) {
      JsonElement value = record.get(field.getName());
      if (!(value instanceof JsonPrimitive item))
        throw new JsonParseException("Invalid " + field.getName());
      if (field.getType() == int.class) {
        if (!item.isNumber()) throw new JsonParseException(field.getName() + " must be an integer");
        try {
          item.getAsBigDecimal().intValueExact();
        } catch (ArithmeticException error) {
          throw new JsonParseException(field.getName() + " must be a 32-bit integer", error);
        }
      } else if (field.getType() == boolean.class ? !item.isBoolean() : !item.isString()) {
        throw new JsonParseException("Invalid " + field.getName());
      }
    }
  }

  /** モデルの形式に従い、形を表すフィールドとその値を照合してから読む。 */
  PrismShape readShape(Gson json, JsonObject record) {
    if (!matchesFields(record))
      throw new JsonParseException("Invalid shape fields for " + id + ": " + record.keySet());
    validate(record);
    return json.fromJson(record, type);
  }

  boolean matchesFields(JsonObject record) {
    return fields.equals(record.keySet());
  }

  /**
   * model.json に書く名前を返す。
   *
   * @return 名前
   */
  public String id() {
    return id;
  }

  /**
   * 名前から形式を返す。
   *
   * @param id model.json の architecture
   * @return 形式
   * @throws IllegalArgumentException 名前がどの形式でもないとき
   */
  public static PrismArchitecture byId(String id) {
    for (PrismArchitecture architecture : values()) {
      if (architecture.id.equals(id)) {
        return architecture;
      }
    }
    throw new IllegalArgumentException("unknown PRISM architecture " + id);
  }
}
