package com.epsilon.prism.hip;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * K1 の 1 つの型の埋め込み（BF16 にした W と E の積を FP32 で畳んだ組の表の和）。
 *
 * <pre>
 * x = ⌊Σ_表 T[Σ_i 値(列_i)·刻み_i] + Σ_k W_d[k]·特徴_k + b⌉
 * </pre>
 *
 * 表 T は FP32 の {@code [行, H]}、W_d は 2 値の欄・数値・厳密な欄を Linear の入力の順に並べた BF16 の {@code [特徴, H]}（転置）、
 * b は BF16 に丸めた偏りを FP32 で持つ。位置は全て WeightPack の中のバイトの位置。
 *
 * @param category 欄の区画
 * @param scalar 数値の区画（BF16、なければ null）
 * @param exact 厳密な区画（FP32 を BF16 に丸めて読む、なければ null）
 * @param tables 畳んだ組の表（{@link #MAX_TABLES} 以下）
 * @param dense 表に畳まない特徴（{@link #MAX_DENSE} 以下、Linear の入力の順）
 * @param denseOffset W_d の位置（特徴がなければ 0）
 * @param biasOffset b の位置
 */
public record HipEmbedding(
    HipSection category,
    HipSection scalar,
    HipSection exact,
    List<Table> tables,
    List<Dense> dense,
    int denseOffset,
    int biasOffset) {

  /** 1 トークンが引く表の上限。 */
  public static final int MAX_TABLES = 8;

  /** 1 つの表に畳む欄の上限。 */
  public static final int MAX_MEMBERS = 8;

  /** 表に畳まない特徴の上限。 */
  public static final int MAX_DENSE = 24;

  /** 1 つの表の int の数（位置・行・欄の数・列・刻み）。 */
  public static final int TABLE_INTS = 3 + 2 * MAX_MEMBERS;

  /** 1 つの型の int の数（native の {@code EmbeddingSpec} と同じ）。 */
  public static final int INTS = 8 + MAX_TABLES * TABLE_INTS + MAX_DENSE;

  /** 置き場（native の順と同じ）。出力はトークンの型の区画（局・席・牌・出来事）か遷移の静的な部分。 */
  public enum Slot {
    /** 局のトークン → 残差の流れの局の区画。 */
    ROUND,
    /** 席のトークン → 席の区画。 */
    SEAT,
    /** 牌のトークン → 牌の区画。 */
    TILE,
    /** 出来事のトークン → 出来事の区画。 */
    EVENT,
    /** 遷移 → 遷移の静的な部分。 */
    TRANSITION
  }

  /** 表に畳まない特徴の元。 */
  public enum Source {
    /** 欄の区画の列（2 値）。 */
    FLAG,
    /** 数値の区画の列（BF16）。 */
    SCALAR,
    /** 厳密な区画の列（FP32 を BF16 に丸める）。 */
    EXACT
  }

  /**
   * 畳んだ組の表。
   *
   * @param offset 表の位置（FP32 の {@code [rows, H]}）
   * @param rows 行の数（欄の語彙の積）
   * @param columns 欄の列
   * @param strides 欄ごとの刻み（行 = Σ 値·刻み）
   */
  public record Table(int offset, int rows, int[] columns, int[] strides) {

    /**
     * 形を確かめる。
     *
     * @throws IllegalArgumentException 欄の数が合わないか、範囲を外れるとき
     */
    public Table {
      if (offset < 0 || rows <= 0 || columns.length == 0 || columns.length > MAX_MEMBERS) {
        throw new IllegalArgumentException("invalid embedding table at " + offset + " with " + rows + " rows");
      }
      if (columns.length != strides.length) {
        throw new IllegalArgumentException("embedding table columns and strides differ in length");
      }
      columns = columns.clone();
      strides = strides.clone();
    }

    @Override
    public int[] columns() {
      return columns.clone();
    }

    @Override
    public int[] strides() {
      return strides.clone();
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Table that
          && offset == that.offset
          && rows == that.rows
          && Arrays.equals(columns, that.columns)
          && Arrays.equals(strides, that.strides);
    }

    @Override
    public int hashCode() {
      return Objects.hash(offset, rows, Arrays.hashCode(columns), Arrays.hashCode(strides));
    }

    @Override
    public String toString() {
      return "Table[offset=" + offset + ", rows=" + rows + ", columns=" + Arrays.toString(columns) + "]";
    }
  }

  /**
   * 表に畳まない特徴。
   *
   * @param source 元
   * @param index 元の区画の列
   */
  public record Dense(Source source, int index) {

    /**
     * 形を確かめる。
     *
     * @throws IllegalArgumentException 列が負か大きすぎるとき
     */
    public Dense {
      Objects.requireNonNull(source, "source");
      if (index < 0 || index > 0xFFFF) {
        throw new IllegalArgumentException("dense feature column out of range: " + index);
      }
    }
  }

  /**
   * 形を確かめる。
   *
   * @throws IllegalArgumentException 上限を超えるか、元の区画がない特徴があるとき
   */
  public HipEmbedding {
    Objects.requireNonNull(category, "category");
    tables = List.copyOf(tables);
    dense = List.copyOf(dense);
    if (tables.isEmpty() || tables.size() > MAX_TABLES) {
      throw new IllegalArgumentException("embedding needs 1.." + MAX_TABLES + " tables, got " + tables.size());
    }
    if (dense.size() > MAX_DENSE) {
      throw new IllegalArgumentException("embedding has more than " + MAX_DENSE + " dense features: " + dense.size());
    }
    for (Dense feature : dense) {
      if ((feature.source() == Source.SCALAR && scalar == null)
          || (feature.source() == Source.EXACT && exact == null)) {
        throw new IllegalArgumentException("dense feature " + feature + " has no source section");
      }
    }
    if (denseOffset < 0 || biasOffset < 0) {
      throw new IllegalArgumentException("embedding offsets must not be negative");
    }
  }

  /**
   * native の {@code EmbeddingSpec} の並びで書く。
   *
   * @param into 頭の int の列
   * @param at 書き始める位置
   */
  void encode(int[] into, int at) {
    int[] spec = new int[INTS];
    spec[0] = category.ordinal();
    spec[1] = scalar == null ? -1 : scalar.ordinal();
    spec[2] = exact == null ? -1 : exact.ordinal();
    spec[3] = tables.size();
    spec[4] = dense.size();
    spec[5] = denseOffset;
    spec[6] = biasOffset;
    int cursor = 8;
    for (int t = 0; t < MAX_TABLES; t++, cursor += TABLE_INTS) {
      if (t >= tables.size()) {
        continue;
      }
      Table table = tables.get(t);
      spec[cursor] = table.offset();
      spec[cursor + 1] = table.rows();
      spec[cursor + 2] = table.columns.length;
      for (int m = 0; m < table.columns.length; m++) {
        spec[cursor + 3 + m] = table.columns[m];
        spec[cursor + 3 + MAX_MEMBERS + m] = table.strides[m];
      }
    }
    for (int k = 0; k < dense.size(); k++) {
      spec[cursor + k] = dense.get(k).source().ordinal() << 16 | dense.get(k).index();
    }
    System.arraycopy(spec, 0, into, at, INTS);
  }
}
