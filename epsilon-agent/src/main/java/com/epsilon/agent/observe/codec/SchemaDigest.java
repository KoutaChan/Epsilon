package com.epsilon.agent.observe.codec;

import com.epsilon.mahjong.rules.Vocabulary;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.Objects;

/**
 * 形式のフィンガープリントを作る。語彙（enum の名前と、そのルールで使う定数を宣言の順に並べた列）、ビットの配置、名前つきの値を
 * 順に記述へ書き、その SHA-256 の先頭 8 バイトを 16 進で返す。
 *
 * <p>バージョン番号は持たない。形式を変えればフィンガープリントが変わり、合わなければ読む側が止める。型の名前はパッケージを除いた
 * Outer$Inner で書く。そのため、パッケージを移してもフィンガープリントは変わらないが、外側のクラスの名前は変えられない。
 */
public final class SchemaDigest {

  /** フィンガープリントに使う、ハッシュの先頭のバイト数。 */
  private static final int FINGERPRINT_BYTES = 8;

  private final StringBuilder text;

  private SchemaDigest(String name) {
    text = new StringBuilder(Objects.requireNonNull(name, "name"));
  }

  /** 名前を付けて、形式の記述を始める。 */
  public static SchemaDigest named(String name) {
    return new SchemaDigest(name);
  }

  /**
   * 語彙の型の名前と列を加える。列は {@code Arrays.toString} と同じ形で書くので、全ての定数を持つ列は、enum の {@code values()} を
   * 文字列にしたものと同じになる。
   */
  public SchemaDigest values(Vocabulary<?> vocabulary) {
    text.append(';').append(name(vocabulary.type())).append('=').append(vocabulary);
    return this;
  }

  public SchemaDigest layout(BitLayout<?> layout) {
    text.append(';').append(layout.describe());
    return this;
  }

  /** 名前つきの値を加える（値は {@code toString} で書く）。 */
  public SchemaDigest with(String key, Object value) {
    text.append(';').append(key).append('=').append(value);
    return this;
  }

  /** 記述の SHA-256 を取り、先頭 8 バイトを 16 進の 16 文字で返す。 */
  public String hex() {
    try {
      byte[] hash = MessageDigest.getInstance("SHA-256").digest(text.toString().getBytes(StandardCharsets.UTF_8));
      return HexFormat.of().formatHex(hash, 0, FINGERPRINT_BYTES);
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException("SHA-256 is unavailable", impossible);
    }
  }

  /** 記述の文字列（テストと診断に使う）。 */
  @Override
  public String toString() {
    return text.toString();
  }

  /** パッケージを除いた型の名前。入れ子の型は Outer$Inner。 */
  public static String name(Class<?> type) {
    String packageName = type.getPackageName();
    return packageName.isEmpty() ? type.getName() : type.getName().substring(packageName.length() + 1);
  }

  /** 型を名前の順に並べたコピー。sealed の permits のように順が決まらない一覧を、決まった順で書くのに使う。 */
  public static Class<?>[] sorted(Class<?>[] types) {
    Class<?>[] copy = types.clone();
    Arrays.sort(copy, Comparator.comparing(Class::getName));
    return copy;
  }
}
