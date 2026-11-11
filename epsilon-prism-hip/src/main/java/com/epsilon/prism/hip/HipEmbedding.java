package com.epsilon.prism.hip;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * K1 が 1 つの型のトークンを埋め込む方法。フィールドの値の組ごとに、BF16 にした W と E の積を FP32 で前もって計算して表にしておき
 * （組の表）、その表を引いて足す。
 *
 * <pre>
 * x = ⌊acc⌉、acc = Σ_表 T[Σ_i 値(列_i)·ストライド_i] + Σ_k W_d[k]·特徴_k + b
 * </pre>
 *
 * 表 T は FP32 の {@code [行, H]}。W_d は、2 値のフィールド・数値・厳密な値を Linear の入力の順に並べた BF16 の {@code [特徴, H]}（転置）。
 * b は BF16 に丸めたバイアスを FP32 で持つ。位置は全て WeightPack の中のバイト位置。
 *
 * <p>局・席・牌・イベントの型は、層 0 の Attention の q・k・v を折り込んだ表も持つ（遷移の型は持たず、位置は −1）:
 *
 * <pre>
 * qkv0 = ⌊r·(Σ_表 T'[行] + Σ_k W'_d[k]·特徴_k + b') + b_qkv⌉、r = 1/√(mean(acc²) + ε)
 * </pre>
 *
 * T' = T·diag(g0)·W_qkv0ᵀ（表ごとに FP32 の {@code [行, 3A]}）、W'_d = W_d·diag(g0)·W_qkv0ᵀ（FP32 の {@code [特徴, 3A]}）、b' と
 * b_qkv（FP32 の {@code [2, 3A]}）。列はヘッドの対ごとに並べる（{@link #foldColumn}）。
 *
 * @param category カテゴリの領域
 * @param scalar 数値の領域（BF16。なければ null）
 * @param exact 厳密な値の領域（FP32 を BF16 に丸めて読む。なければ null）
 * @param tables 折り込んだ組の表（{@link #MAX_TABLES} 個以下）
 * @param dense 表に折り込まない特徴（{@link #MAX_DENSE} 個以下、Linear の入力の順）
 * @param denseOffset W_d の位置（特徴がなければ 0）
 * @param biasOffset b の位置
 * @param foldOffset W'_d の位置。後ろに b' と b_qkv が続く（層 0 の QKV を折り込まない型は −1）
 */
public record HipEmbedding(
    HipSection category,
    HipSection scalar,
    HipSection exact,
    List<Table> tables,
    List<Dense> dense,
    int denseOffset,
    int biasOffset,
    int foldOffset) {

  /** 1 つのトークンが引く表の数の上限。 */
  public static final int MAX_TABLES = 8;

  /** 1 つの表に折り込むフィールドの数の上限。 */
  public static final int MAX_MEMBERS = 8;

  /** 表に折り込まない特徴の数の上限。 */
  public static final int MAX_DENSE = 24;

  /** 1 つの表を書くのに使う int の数（位置・行の数・フィールドの数・列・ストライド・T' の位置）。 */
  public static final int TABLE_INTS = 4 + 2 * MAX_MEMBERS;

  /** 1 つの型を書くのに使う int の数（native の {@code EmbeddingSpec} と同じ）。 */
  public static final int INTS = 8 + MAX_TABLES * TABLE_INTS + MAX_DENSE;

  /** 埋め込みの書き先。並びは native と同じ。トークンの型ごとの領域（局・席・牌・イベント）か、遷移の静的な部分に書く。 */
  public enum Slot {
    /** 局のトークン → 残差ストリームの局の領域。 */
    ROUND,
    /** 席のトークン → 席の領域。 */
    SEAT,
    /** 牌のトークン → 牌の領域。 */
    TILE,
    /** イベントのトークン → イベントの領域。 */
    EVENT,
    /** 遷移 → 遷移の静的な部分。 */
    TRANSITION
  }

  /** 表に折り込まない特徴の読み元。 */
  public enum Source {
    /** カテゴリの領域の列（2 値）。 */
    FLAG,
    /** 数値の領域の列（BF16）。 */
    SCALAR,
    /** 厳密な値の領域の列（FP32 を BF16 に丸める）。 */
    EXACT
  }

  /**
   * 層 0 の QKV の列が、折り込んだ表の中でどの位置に来るかを返す（native の foldColumn と同じ）。折り込んだ表はヘッドの対ごとに列を
   * 並べる。対 p の q・k・v の 32 列ずつ（ヘッド 2p と 2p + 1 の 16 次元ずつ）が 96p から続くので、K2 でヘッドの対を受け持つ
   * ワークグループは、1 トークンにつき連続した 192 バイトを読む。
   *
   * @param column q・k・v を {@code [3A]} に並べたときの列（{@code < 3A}）
   * @param attention Attention の幅 A
   * @return ヘッドの対ごとの並びでの位置
   */
  public static int foldColumn(int column, int attention) {
    return 96 * ((column % attention) / 32) + 32 * (column / attention) + column % 32;
  }

  /**
   * 折り込んだ組の表。
   *
   * @param offset 表の位置（FP32 の {@code [rows, H]}）
   * @param rows 行の数（フィールドごとの語彙の大きさの積）
   * @param columns フィールドの列
   * @param strides フィールドごとのストライド（行 = Σ 値·ストライド）
   * @param foldOffset T' の位置（FP32 の {@code [rows, 3A]}。層 0 の QKV を折り込まない型は −1）
   */
  public record Table(int offset, int rows, int[] columns, int[] strides, int foldOffset) {

    /**
     * 形を確かめる。
     *
     * @throws IllegalArgumentException 位置が負か、行の数が 0 以下か、T' の位置が −1 より小さいとき。フィールドが 0 個か
     *     {@link #MAX_MEMBERS} 個を超えるとき。列とストライドの数が違うとき
     */
    public Table {
      if (offset < 0 || rows <= 0 || columns.length == 0 || columns.length > MAX_MEMBERS || foldOffset < -1) {
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
          && Arrays.equals(strides, that.strides)
          && foldOffset == that.foldOffset;
    }

    @Override
    public int hashCode() {
      return Objects.hash(offset, rows, Arrays.hashCode(columns), Arrays.hashCode(strides), foldOffset);
    }

    @Override
    public String toString() {
      return "Table[offset=" + offset + ", rows=" + rows + ", columns=" + Arrays.toString(columns) + ", foldOffset="
          + foldOffset + "]";
    }
  }

  /**
   * 表に折り込まない特徴。
   *
   * @param source 読み元
   * @param index 読み元の領域の列
   */
  public record Dense(Source source, int index) {

    /**
     * 形を確かめる。
     *
     * @throws IllegalArgumentException 列が負か 0xFFFF を超えるとき
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
   * @throws IllegalArgumentException 表が 0 個か上限を超えるとき、特徴が上限を超えるとき、読み元の領域がない特徴があるとき、位置が
   *     負のとき（W'_d の位置は −1 より小さいとき）。また、表の T' と W'_d の片方だけに位置があるとき
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
    if (denseOffset < 0 || biasOffset < 0 || foldOffset < -1) {
      throw new IllegalArgumentException("embedding offsets must not be negative");
    }
    for (Table table : tables) {
      if ((table.foldOffset() < 0) != (foldOffset < 0)) {
        throw new IllegalArgumentException("the folded tables and the folded dense weights must be present together");
      }
    }
  }

  /**
   * native の {@code EmbeddingSpec} と同じ並びで書く。
   *
   * @param into 書き先のヘッダーの int の列
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
    spec[7] = foldOffset;
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
      spec[cursor + 3 + 2 * MAX_MEMBERS] = table.foldOffset();
    }
    for (int k = 0; k < dense.size(); k++) {
      spec[cursor + k] = dense.get(k).source().ordinal() << 16 | dense.get(k).index();
    }
    System.arraycopy(spec, 0, into, at, INTS);
  }
}
