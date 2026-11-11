package com.epsilon.reviewer.json;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * JSON の厳密な読み取りを確かめる。見るのは、知らないフィールド、型の違い、整数として受け取れない数（小数・範囲外）、入れ子、
 * 省いたフィールド、null、JSON の形。
 */
public class StrictRecordsTest {
  record Item(int index, String tile) {}

  record Sample(
      String name,
      int count,
      Integer optional,
      boolean flag,
      Boolean maybe,
      List<Integer> numbers,
      List<Item> items,
      Item nested) {
    Sample {
      numbers = numbers == null ? List.of() : numbers;
      items = items == null ? List.of() : items;
    }
  }

  @Test
  public void fullAndMinimalObjectsAreRead() {
    Sample full =
        read(
            "{\"name\":\"a\",\"count\":2.0,\"optional\":3,\"flag\":true,\"maybe\":false,"
                + "\"numbers\":[1,2],\"items\":[{\"index\":0,\"tile\":\"5mr\"}],"
                + "\"nested\":{\"index\":1,\"tile\":\"E\"}}");
    Assert.assertEquals(
        full,
        new Sample(
            "a", 2, 3, true, false, List.of(1, 2), List.of(new Item(0, "5mr")), new Item(1, "E")));
    Sample minimal = read("{\"name\":\"a\",\"count\":1,\"flag\":false,\"optional\":null}");
    Assert.assertNull(minimal.optional());
    Assert.assertNull(minimal.maybe());
    Assert.assertNull(minimal.nested());
    Assert.assertEquals(minimal.numbers(), List.of());
    Assert.assertEquals(minimal.items(), List.of());
  }

  @Test
  public void malformedFieldsNameTheirPath() {
    String base = "\"name\":\"a\",\"count\":1,\"flag\":true";
    rejected("{" + base + ",\"extra\":1}", "The request has an unknown field: extra.");
    rejected("{\"count\":1,\"flag\":true}", "name is required.");
    rejected("{\"name\":\"a\",\"flag\":true}", "count is required.");
    rejected("{\"name\":\"a\",\"count\":null,\"flag\":true}", "count is required.");
    rejected("{\"name\":5,\"count\":1,\"flag\":true}", "name must be a string.");
    rejected("{\"name\":\"a\",\"count\":\"1\",\"flag\":true}", "count must be an integer.");
    rejected("{\"name\":\"a\",\"count\":1.5,\"flag\":true}", "count must be an integer.");
    rejected("{\"name\":\"a\",\"count\":1e10,\"flag\":true}", "count must be an integer.");
    rejected("{\"name\":\"a\",\"count\":1,\"flag\":1}", "flag must be a boolean.");
    rejected("{" + base + ",\"numbers\":{}}", "numbers must be an array.");
    rejected("{" + base + ",\"numbers\":[1,null]}", "numbers[1] is required.");
    rejected("{" + base + ",\"numbers\":[1,\"2\"]}", "numbers[1] must be an integer.");
    rejected("{" + base + ",\"nested\":[]}", "nested must be an object.");
    rejected("{" + base + ",\"nested\":{\"index\":1}}", "nested.tile is required.");
    rejected(
        "{" + base + ",\"items\":[{\"index\":0,\"tile\":\"E\",\"x\":1}]}",
        "items[0] has an unknown field: x.");
  }

  @Test
  public void onlyOneStrictJsonObjectIsRead() {
    for (String text :
        List.of("{} {}", "[1]", "{'a':1}", "{a:1}", "{\"a\":1,}", "", "nul", "{\"a\":NaN}"))
      Assert.assertEquals(
          Assert.expectThrows(ReviewerException.class, () -> object(text)).code(),
          ErrorCode.INVALID_REQUEST,
          text);
    Assert.expectThrows(
        ReviewerException.class,
        () -> Json.readObject("{\"a\":\"bbbbbbbbbb\"}".getBytes(StandardCharsets.UTF_8), 8));
    Assert.assertEquals(object(" {\"a\":1} ").get("a").getAsInt(), 1);
  }

  @Test
  public void treesKeepTheirNullsAndRecordsDropThem() {
    JsonObject tree = JsonParser.parseString("{\"draw\":null}").getAsJsonObject();
    Assert.assertEquals(new String(Json.write(tree), StandardCharsets.UTF_8), "{\"draw\":null}");
    Assert.assertEquals(
        new String(Json.write(new Item(1, null)), StandardCharsets.UTF_8), "{\"index\":1}");
    Assert.assertEquals(
        new String(Json.write(new Item(1, "<&>")), StandardCharsets.UTF_8),
        "{\"index\":1,\"tile\":\"<&>\"}");
  }

  private static JsonObject object(String text) {
    return Json.readObject(text.getBytes(StandardCharsets.UTF_8), 1024);
  }

  private static Sample read(String text) {
    return StrictRecords.read(object(text), Sample.class);
  }

  private static void rejected(String text, String message) {
    ReviewerException failure = Assert.expectThrows(ReviewerException.class, () -> read(text));
    Assert.assertEquals(failure.code(), ErrorCode.INVALID_REQUEST, text);
    Assert.assertEquals(failure.getMessage(), message, text);
  }
}
