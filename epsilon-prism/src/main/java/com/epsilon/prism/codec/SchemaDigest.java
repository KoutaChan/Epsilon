package com.epsilon.prism.codec;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.Objects;

/**
 * 形式の指紋を作る。enum の名前と定数の順、ビットの配置、名前つきの値を順に書いた記述から SHA-256 を作り、先頭 8 B を 16 進で返す。
 *
 * <p>版の番号は持たない。形式を変えれば指紋が変わり、合わなければ読む側が止めるだけにする。型の名前はパッケージを除いて書くので、
 * パッケージを移しても指紋は変わらない。
 */
public final class SchemaDigest {

  private final StringBuilder text;

  private SchemaDigest(String name) {
    text = new StringBuilder(Objects.requireNonNull(name, "name"));
  }

  /**
   * 指紋の記述を始める。
   *
   * @param name 形式の名前
   * @return 記述
   */
  public static SchemaDigest named(String name) {
    return new SchemaDigest(name);
  }

  /**
   * enum の名前と、定数の名前を宣言の順に加える。
   *
   * @param type enum
   * @return この記述
   */
  public SchemaDigest values(Class<? extends Enum<?>> type) {
    text.append(';').append(name(type)).append('=');
    text.append(Arrays.toString(type.getEnumConstants()));
    return this;
  }

  /**
   * ビットの配置を加える。
   *
   * @param layout 配置
   * @return この記述
   */
  public SchemaDigest layout(BitLayout<?> layout) {
    text.append(';').append(layout.describe());
    return this;
  }

  /**
   * 名前つきの値を加える。
   *
   * @param key 名前
   * @param value 値（{@code toString} で書く）
   * @return この記述
   */
  public SchemaDigest with(String key, Object value) {
    text.append(';').append(key).append('=').append(value);
    return this;
  }

  /**
   * 指紋を返す。
   *
   * @return SHA-256 の先頭 8 B の 16 進（16 文字）
   */
  public String hex() {
    try {
      byte[] hash =
          MessageDigest.getInstance("SHA-256").digest(text.toString().getBytes(StandardCharsets.UTF_8));
      return HexFormat.of().formatHex(hash, 0, 8);
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException("SHA-256 is unavailable", impossible);
    }
  }

  /**
   * 記述の文字列を返す（試験と診断のため）。
   *
   * @return 記述
   */
  @Override
  public String toString() {
    return text.toString();
  }

  /**
   * パッケージを除いた型の名前を返す。入れ子の型は {@code Outer$Inner} になる。
   *
   * @param type 型
   * @return 名前
   */
  public static String name(Class<?> type) {
    String packageName = type.getPackageName();
    return packageName.isEmpty() ? type.getName() : type.getName().substring(packageName.length() + 1);
  }

  /**
   * 型を名前の順に並べた写しを返す（sealed の permits のように順が決まらない一覧を、決まった順で書くため）。
   *
   * @param types 型
   * @return 並べた写し
   */
  public static Class<?>[] sorted(Class<?>[] types) {
    Class<?>[] copy = types.clone();
    Arrays.sort(copy, Comparator.comparing(Class::getName));
    return copy;
  }
}
