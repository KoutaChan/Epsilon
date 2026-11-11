package com.epsilon.prism.input;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;
import com.epsilon.prism.FourPlayerFormat;
import java.util.Objects;

/**
 * 入力のフィールドの値の書き方・語彙・繰り返し。値 0 は常に「なし・0 個・偽」を表す。
 *
 * @param coding 値の書き方（ライターが {@link Ids} のどの変換を通すか）。フィンガープリントに入るので、語彙が同じでも書き方を変えれば
 *     フィンガープリントが変わる
 * @param vocabulary 取りうる値の数（0 を含む）
 * @param repeats 同じ語彙を並べる数（席ごと・牌ごとなど）。繰り返しごとに別の名前空間を持つ
 * @param values 値の語彙（{@link Coding#ENUM} のフィールドだけ。ほかは null）。フィンガープリントは値の enum の名前と並びをここから集める
 *     ので、一覧を手で書かない
 */
public record FieldSpec(Coding coding, int vocabulary, int repeats, Vocabulary<?> values) {

  /** 語彙の上限。値は u8 で送るので、語彙はこれ以下にする。 */
  public static final int MAX_VOCABULARY = 256;

  /** 語彙で値の範囲が決まるフィールド（埋め込むフィールドと、関係の属性）。 */
  public interface BoundedField {

    /**
     * 語彙と繰り返しを返す。
     *
     * @return 語彙と繰り返し
     */
    FieldSpec spec();
  }

  /** 値の書き方。 */
  public enum Coding {
    /** 真偽の 2 値（0 は偽、1 は真）。 */
    FLAG,
    /** 0〜max の個数（値はそのまま書き、0 は 0 個）。 */
    COUNT,
    /** n 種類の 0 始まりの番号（{@link Ids#tile}・{@link Ids#seat}・{@link Ids#index} で番号 + 1 を書き、0 はなし）。 */
    ID,
    /** enum の値（{@link Ids#of} で語彙の中の位置 + 1 を書き、0 はなし）。 */
    ENUM
  }

  /**
   * 書き方・語彙・繰り返しが正しいか確かめる。
   *
   * @throws IllegalArgumentException 語彙が 2〜{@link #MAX_VOCABULARY} の外のとき、繰り返しが 1 未満のとき、真偽の 2 値なのに語彙が
   *     2 でないとき、enum のフィールドなのに値の語彙が無いか enum でないのに値の語彙があるとき、または値の語彙の大きさが語彙と
   *     合わないとき
   */
  public FieldSpec {
    Objects.requireNonNull(coding, "coding");
    if (vocabulary < 2 || vocabulary > MAX_VOCABULARY || repeats < 1) {
      throw new IllegalArgumentException("invalid field spec: " + vocabulary + " x " + repeats);
    }
    if (coding == Coding.FLAG && vocabulary != 2) {
      throw new IllegalArgumentException("flag vocabulary must be 2: " + vocabulary);
    }
    if ((coding == Coding.ENUM) != (values != null)) {
      throw new IllegalArgumentException("value vocabulary must be given exactly for enum fields");
    }
    if (values != null && values.size() + 1 != vocabulary) {
      throw new IllegalArgumentException("vocabulary does not match " + values.type().getName());
    }
  }

  /** 真偽の 2 値のフィールド（語彙 2）。 */
  public static FieldSpec flag() {
    return new FieldSpec(Coding.FLAG, 2, 1, null);
  }

  /** 0〜max の個数のフィールド（語彙 max + 1）。 */
  public static FieldSpec count(int max) {
    return new FieldSpec(Coding.COUNT, max + 1, 1, null);
  }

  /** n 種類の番号のフィールド（値は番号 + 1、0 はなし。語彙は種類の数 + 1）。牌・席・イベントの番号・向聴に使う。 */
  public static FieldSpec ids(int kinds) {
    return new FieldSpec(Coding.ID, kinds + 1, 1, null);
  }

  /**
   * 四人打ちの語彙で値を書く enum のフィールドを返す。フィールドの enum の初期化から呼ばれるので、入力の形式のクラスの初期化には頼らない
   * （頼ると、フィールドの enum を先に使ったときに初期化が循環する）。
   *
   * @param type 値の enum
   * @return フィールドの仕様
   */
  public static FieldSpec oneOf(Class<? extends Enum<?>> type) {
    return of(FourPlayerFormat.values(type));
  }

  /**
   * ルールごとの語彙で値を書く enum のフィールドを返す。語彙は、そのルールにある定数だけを宣言の順に並べたもの。フィールドの enum の
   * 初期化から呼んでよい。
   *
   * @param type 値の enum
   * @param variant ルール（四人打ち・三人打ち）
   * @return フィールドの仕様
   */
  @SuppressWarnings({"unchecked", "rawtypes"})
  public static FieldSpec oneOf(Class<? extends Enum<?>> type, Variant variant) {
    return of(Vocabulary.of((Class) type, variant));
  }

  /** enum の値のフィールド（値は語彙の中の位置 + 1、0 はなし。語彙は定数の数 + 1）。 */
  public static FieldSpec of(Vocabulary<?> values) {
    return new FieldSpec(Coding.ENUM, values.size() + 1, 1, values);
  }

  /** 同じ語彙を {@code count} 回並べた仕様を返す。 */
  public FieldSpec times(int count) {
    return new FieldSpec(coding, vocabulary, count, values);
  }
}
