package com.epsilon.prism.input;

import java.util.Objects;

/**
 * 入力の欄の書き方・語彙・繰り返し。値 0 は常に「なし・0 個・偽」。
 *
 * @param coding 値の書き方（書き手が {@link Ids} のどの変換を通すか）。指紋に入るので、語彙が同じでも書き方を変えれば指紋が変わる
 * @param vocabulary 値の数（0 を含む）
 * @param repeats 同じ語彙を並べる数（席・牌ごと）。繰り返しごとに別の名前空間を持つ
 * @param values 値の enum（{@link Coding#ENUM} の欄だけ。ほかは null）。指紋はここから値の enum の名前と順を集めるので、一覧を手で書かない
 */
public record FieldSpec(Coding coding, int vocabulary, int repeats, Class<? extends Enum<?>> values) {

  /** 値の書き方。 */
  public enum Coding {
    /** 2 値（0 は偽、1 は真）。 */
    FLAG,
    /** 0〜max の数え上げ（値はそのまま、0 は 0 個）。 */
    COUNT,
    /** n 種の 0 始まりの番号（{@link Ids#tile}・{@link Ids#seat}・{@link Ids#index} で番号 + 1、0 はなし）。 */
    ID,
    /** enum の値（{@link Ids#of} で ordinal + 1、0 はなし）。 */
    ENUM
  }

  /**
   * 書き方・語彙・繰り返しを確かめる。
   *
   * @throws IllegalArgumentException 語彙が 2〜{@link InputSchema#MAX_VOCABULARY} の外か、繰り返しが 1 未満か、2 値の語彙が 2 でない
   *     か、値の enum が enum の欄だけにないか、値の enum の定数の数と語彙が合わないとき
   */
  public FieldSpec {
    Objects.requireNonNull(coding, "coding");
    if (vocabulary < 2 || vocabulary > InputSchema.MAX_VOCABULARY || repeats < 1) {
      throw new IllegalArgumentException("invalid field spec: " + vocabulary + " x " + repeats);
    }
    if (coding == Coding.FLAG && vocabulary != 2) {
      throw new IllegalArgumentException("flag vocabulary must be 2: " + vocabulary);
    }
    if ((coding == Coding.ENUM) != (values != null)) {
      throw new IllegalArgumentException("value enum must be given exactly for enum fields");
    }
    if (values != null && values.getEnumConstants().length + 1 != vocabulary) {
      throw new IllegalArgumentException("vocabulary does not match " + values.getName());
    }
  }

  /**
   * 2 値の欄（0 は偽、1 は真）。
   *
   * @return 語彙 2
   */
  public static FieldSpec flag() {
    return new FieldSpec(Coding.FLAG, 2, 1, null);
  }

  /**
   * 0〜max の数え上げ（値はそのまま、0 は 0 個）。
   *
   * @param max 最大の数
   * @return 語彙 max + 1
   */
  public static FieldSpec count(int max) {
    return new FieldSpec(Coding.COUNT, max + 1, 1, null);
  }

  /**
   * n 種の番号（値は番号 + 1、0 はなし）。牌・席・出来事の番号・向聴に使う。
   *
   * @param kinds 番号の種類の数
   * @return 語彙 kinds + 1
   */
  public static FieldSpec ids(int kinds) {
    return new FieldSpec(Coding.ID, kinds + 1, 1, null);
  }

  /**
   * enum の値（ordinal + 1、0 はなし）。
   *
   * @param type 値の enum
   * @return 語彙は定数の数 + 1
   */
  public static FieldSpec of(Class<? extends Enum<?>> type) {
    return new FieldSpec(Coding.ENUM, type.getEnumConstants().length + 1, 1, type);
  }

  /**
   * 同じ語彙を count 回並べる。
   *
   * @param count 繰り返しの数
   * @return 繰り返しだけを替えた欄
   */
  public FieldSpec times(int count) {
    return new FieldSpec(coding, vocabulary, count, values);
  }
}
