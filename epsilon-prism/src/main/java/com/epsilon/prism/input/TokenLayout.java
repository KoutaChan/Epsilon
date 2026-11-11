package com.epsilon.prism.input;

import java.util.Objects;

/**
 * フィールドの enum から、トークンの列と埋め込みの名前空間を一度だけ作る。入力のライター、ネットワーク、HIP の表はどれもこれを引き、
 * オフセットを手で並べない。
 *
 * <p>列はフィールドの宣言の順に置き、繰り返しのあるフィールドは繰り返しの数だけ列を続ける。名前空間は列ごとの語彙の累積和で、列 c の値 v
 * は埋め込みの表の {@code columnOffsets()[c] + v} 行を引く。列にも名前空間にも、隙間や重なりはない。
 *
 * <p>作った後は変わらないので、複数のスレッドから読んでよい。列を引くメソッドはメモリを割り当てない。
 *
 * @param <E> フィールドの enum
 */
public final class TokenLayout<E extends Enum<E> & FieldSpec.BoundedField> {

  private final Class<E> type;
  private final E[] fields;
  private final int[] firstColumn;
  private final int[] columnField;
  private final int[] columnVocabulary;
  private final int[] columnOffset;
  private final FieldSpec.Coding[] columnCoding;
  private final int tableSize;

  private TokenLayout(Class<E> type) {
    this.type = Objects.requireNonNull(type, "type");
    fields = type.getEnumConstants();
    firstColumn = new int[fields.length];
    int columns = 0;
    for (E field : fields) {
      firstColumn[field.ordinal()] = columns;
      columns = Math.addExact(columns, field.spec().repeats());
    }
    columnField = new int[columns];
    columnVocabulary = new int[columns];
    columnOffset = new int[columns];
    columnCoding = new FieldSpec.Coding[columns];
    int column = 0;
    int offset = 0;
    for (E field : fields) {
      FieldSpec spec = field.spec();
      for (int repeat = 0; repeat < spec.repeats(); repeat++, column++) {
        columnField[column] = field.ordinal();
        columnVocabulary[column] = spec.vocabulary();
        columnOffset[column] = offset;
        columnCoding[column] = spec.coding();
        offset = Math.addExact(offset, spec.vocabulary());
      }
    }
    tableSize = offset;
  }

  /**
   * フィールドの enum から配置を作る。
   *
   * @param type フィールドの enum
   * @param <E> フィールドの enum
   * @return 配置
   */
  public static <E extends Enum<E> & FieldSpec.BoundedField> TokenLayout<E> of(Class<E> type) {
    return new TokenLayout<>(type);
  }

  /**
   * フィールドの enum を返す。
   *
   * @return フィールドの enum のクラス
   */
  public Class<E> type() {
    return type;
  }

  /**
   * 1 トークンの列の数を返す。
   *
   * @return 列の数
   */
  public int width() {
    return columnVocabulary.length;
  }

  /**
   * 埋め込みの表の行の数（列ごとの語彙の和）を返す。
   *
   * @return 行の数
   */
  public int tableSize() {
    return tableSize;
  }

  /**
   * 繰り返しのないフィールドの列を返す。
   *
   * @param field フィールド
   * @return 列
   * @throws IllegalArgumentException フィールドに繰り返しがあるとき（そのときは {@link #column(Enum, int)} を使う）
   */
  public int column(E field) {
    if (field.spec().repeats() != 1) {
      throw new IllegalArgumentException(type.getSimpleName() + "." + field.name() + " repeats");
    }
    return firstColumn[field.ordinal()];
  }

  /**
   * フィールドの何番目かの繰り返しの列を返す。
   *
   * @param field フィールド
   * @param index 繰り返しの番号（0 始まり）
   * @return 列
   */
  public int column(E field, int index) {
    return firstColumn[field.ordinal()] + Objects.checkIndex(index, field.spec().repeats());
  }

  /**
   * フィールドの何番目かの繰り返しについて、埋め込みの表での先頭の行を返す。値 v は、この行 + v の行を引く。
   *
   * @param field フィールド
   * @param index 繰り返しの番号（0 始まり）
   * @return 表の行
   */
  public int offset(E field, int index) {
    return columnOffset[column(field, index)];
  }

  /**
   * 列ごとの語彙の写しを返す（ネットワークの表と HIP の定数を作るときに使う）。
   *
   * @return 列の順の語彙
   */
  public int[] columnVocabularies() {
    return columnVocabulary.clone();
  }

  /**
   * 列の値の書き方を返す（ネットワークの表と HIP の定数を作るときに使う）。
   *
   * @param column 列
   * @return 書き方
   */
  public FieldSpec.Coding coding(int column) {
    return columnCoding[column];
  }

  /**
   * 列ごとの名前空間の先頭の行の写しを返す（ネットワークの表と HIP の定数を作るときに使う）。
   *
   * @return 列の順の表の行
   */
  public int[] columnOffsets() {
    return columnOffset.clone();
  }

  /**
   * 1 トークン分の値（u8）が、どれも列の語彙の中にあるかを確かめる。行ごとに 1 回呼ぶ。
   *
   * @param values 値を並べた配列
   * @param from このトークンの最初の列の添字
   * @throws IllegalStateException 語彙の外の値があるとき
   */
  public void requireValid(byte[] values, int from) {
    for (int column = 0; column < columnVocabulary.length; column++) {
      int value = values[from + column] & 0xFF;
      if (value >= columnVocabulary[column]) {
        E field = fields[columnField[column]];
        throw new IllegalStateException(
            type.getSimpleName()
                + "."
                + field.name()
                + "["
                + (column - firstColumn[field.ordinal()])
                + "] out of range: "
                + value);
      }
    }
  }
}
