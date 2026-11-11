package com.epsilon.mahjong.engine;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Random;
import org.testng.Assert;

/**
 * 状態のフィールドをリフレクションで読んで比べる、テスト用のヘルパー。
 */
final class Fields {

  private Fields() {}

  /**
   * 2 つの状態の、static でない全てのフィールドを比べる。配列は要素ごとに、状態の部品はフィールドごとにたどる。値が等しいことに加えて、
   * 2 つが配列や部品を共有していないこと（片方を変えても他方が変わらないこと）も確かめる。enum は同じ定数を指していればよい。Random は
   * 牌山が使う直前に種を入れ直すので比べない。
   */
  static void assertSameFields(Object expected, Object actual) {
    compareObjects(expected, actual, expected.getClass().getSimpleName());
  }

  /**
   * フィールドの値を読む。どのビューからも届かない部品を、テストで読むために使う。
   *
   * @param owner フィールドを持つオブジェクト
   * @param name フィールドの名前
   * @return フィールドの値
   */
  static Object read(Object owner, String name) {
    try {
      Field field = owner.getClass().getDeclaredField(name);
      field.setAccessible(true);
      return field.get(owner);
    } catch (ReflectiveOperationException e) {
      throw new AssertionError("Cannot read " + owner.getClass().getSimpleName() + "." + name, e);
    }
  }

  private static void compareObjects(Object expected, Object actual, String path) {
    Assert.assertSame(actual.getClass(), expected.getClass(), path);
    for (Class<?> type = expected.getClass(); type != Object.class; type = type.getSuperclass()) {
      for (Field field : type.getDeclaredFields()) {
        if (Modifier.isStatic(field.getModifiers())) {
          continue;
        }
        field.setAccessible(true);
        try {
          compareValues(field.get(expected), field.get(actual), path + "." + field.getName());
        } catch (IllegalAccessException e) {
          throw new AssertionError("Cannot read " + path + "." + field.getName(), e);
        }
      }
    }
  }

  private static void compareValues(Object expected, Object actual, String path) {
    if (expected == null || actual == null || isValue(expected)) {
      Assert.assertEquals(actual, expected, path);
      return;
    }
    if (expected instanceof Random) {
      return;
    }
    Assert.assertNotSame(actual, expected, path + " is shared");
    if (expected.getClass().isArray()) {
      int length = Array.getLength(expected);
      Assert.assertEquals(Array.getLength(actual), length, path + ".length");
      for (int index = 0; index < length; index++) {
        compareValues(Array.get(expected, index), Array.get(actual, index), path + "[" + index + "]");
      }
      return;
    }
    compareObjects(expected, actual, path);
  }

  /** 値で比べる型か（基本型のラッパーと enum）。 */
  private static boolean isValue(Object value) {
    return value instanceof Number || value instanceof Boolean || value instanceof Character || value instanceof Enum;
  }
}
