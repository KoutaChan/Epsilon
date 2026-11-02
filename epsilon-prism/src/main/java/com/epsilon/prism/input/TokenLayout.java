package com.epsilon.prism.input;

import java.util.Objects;

/**
 * 欄の enum から、トークンの列と埋め込みの名前空間を一度だけ作る。書き手・網・HIP の表は全てこれを引き、番地を手で並べない。
 *
 * <p>列は欄の宣言の順に置き、繰り返しのある欄は繰り返しの数だけ続ける。名前空間は列ごとの語彙の累積和で、列 c の値 v は埋め込みの表の
 * {@code columnOffsets()[c] + v} 行を引く。列と名前空間は隙間も重なりもない。
 *
 * <p>作った後は変わらないので、複数のスレッドから読んでよい。列を引く口は割り当てをしない。
 *
 * @param <E> 欄の enum
 */
public final class TokenLayout<E extends Enum<E> & InputSchema.BoundedField> {

  private final Class<E> type;
  private final E[] fields;
  private final int[] firstColumn;
  private final int[] columnField;
  private final int[] columnVocabulary;
  private final int[] columnOffset;
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
    int column = 0;
    int offset = 0;
    for (E field : fields) {
      FieldSpec spec = field.spec();
      for (int repeat = 0; repeat < spec.repeats(); repeat++, column++) {
        columnField[column] = field.ordinal();
        columnVocabulary[column] = spec.vocabulary();
        columnOffset[column] = offset;
        offset = Math.addExact(offset, spec.vocabulary());
      }
    }
    tableSize = offset;
  }

  /**
   * 欄の enum の配置を作る。
   *
   * @param type 欄の enum
   * @param <E> 欄の enum
   * @return 配置
   */
  public static <E extends Enum<E> & InputSchema.BoundedField> TokenLayout<E> of(Class<E> type) {
    return new TokenLayout<>(type);
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
   * 繰り返しのない欄の列を返す。
   *
   * @param field 欄
   * @return 列
   * @throws IllegalArgumentException 欄に繰り返しがあるとき（{@link #column(Enum, int)} を使う）
   */
  public int column(E field) {
    if (field.spec().repeats() != 1) {
      throw new IllegalArgumentException(type.getSimpleName() + "." + field.name() + " repeats");
    }
    return firstColumn[field.ordinal()];
  }

  /**
   * 欄の index 番目の繰り返しの列を返す。
   *
   * @param field 欄
   * @param index 繰り返しの番号（0 始まり）
   * @return 列
   */
  public int column(E field, int index) {
    return firstColumn[field.ordinal()] + Objects.checkIndex(index, field.spec().repeats());
  }

  /**
   * 欄の index 番目の繰り返しの、埋め込みの表での先頭を返す。値 v はこの行 + v を引く。
   *
   * @param field 欄
   * @param index 繰り返しの番号（0 始まり）
   * @return 表の行
   */
  public int offset(E field, int index) {
    return columnOffset[column(field, index)];
  }

  /**
   * 列ごとの語彙の写しを返す（網の表と HIP の定数を作るときに使う）。
   *
   * @return 列の順の語彙
   */
  public int[] columnVocabularies() {
    return columnVocabulary.clone();
  }

  /**
   * 列ごとの名前空間の先頭の写しを返す（網の表と HIP の定数を作るときに使う）。
   *
   * @return 列の順の表の行
   */
  public int[] columnOffsets() {
    return columnOffset.clone();
  }

  /**
   * 1 トークン分の値（u8）を語彙で確かめる。行ごとに 1 回呼ぶ。
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
