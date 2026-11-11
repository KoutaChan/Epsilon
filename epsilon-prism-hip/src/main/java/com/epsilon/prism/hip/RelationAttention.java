package com.epsilon.prism.hip;

import ai.djl.ndarray.NDArray;

/**
 * 関係のバイアスつきの Attention の順伝播と逆伝播。学習のとき DJL の自動微分の外で計算する。GPU の実装は {@link HipAttention}。
 *
 * <p>数式: ℓ_ij = (q_i·k_j)·scale + Σ_s ρ[entries·s + c_s(i, j), h]（キー j があるときだけ）、o_i = Σ_j softmax_j(ℓ_i) v_j。区分 s の
 * コード c_s は 4 ビットで、1 組のコードのうちワード s/4 のビット 4(s%4) から入っている。逆伝播は出力の勾配から q・kv・ρ の勾配を作る。q・kv の
 * 勾配は FP32 で計算し、q・kv の型へ最近接偶数で 1 回丸める。ρ の勾配はワークグループ（行 × ヘッド）ごとの部分和で、呼び出し元が行に
 * ついて足す。配列の形は {@link Dimensions} に書いたとおり。
 */
public interface RelationAttention {

  /** 1 行のトークンの数の上限（native の kAttentionMaxTokens）。 */
  int MAX_TOKENS = 160;

  /** ヘッドの幅（native の kAttentionFeatures）。 */
  int FEATURES = 16;

  /** 関係の区分の数の上限（native の kAttentionMaxSegments）。 */
  int MAX_SEGMENTS = 8;

  /** 1 つの区分がとる値の数の上限（コードが 4 ビットのため）。 */
  int MAX_ENTRIES = 16;

  /**
   * Attention の次元。配列の形は、q・out・outFloat・gradOut・gradQuery が [rows, queries, width]、kv・gradKeyValue が
   * [rows, keys, 2·width]（キーの後にバリュー）、mask が [rows, keys]、codes が [rows, codeQueries, keys, words]（先頭の queries 個を
   * 読む）、ρ が [tableRows, heads]、logSumExp が [rows, heads, queries]、gradTable が [rows·heads, tableRows]。
   *
   * @param rows 行の数
   * @param queries クエリの数
   * @param keys キーの数
   * @param codeQueries 関係のコードの配列で、クエリの軸の長さ
   * @param heads ヘッドの数（幅は heads·{@value #FEATURES}。GPU の順伝播はヘッドの対ごとにワークグループを作るので偶数）
   * @param words 1 組のコードを入れる INT16 のワードの数（{@link RelationCodes#WORDS}。GPU は 1 組のコードを 32 bit の 1 ワードとして読む）
   * @param segments 関係の区分の数
   * @param entries 1 つの区分がとる値の数（ρ の行は segments·entries）
   * @param scale 内積に掛ける倍率
   */
  record Dimensions(
      int rows,
      int queries,
      int keys,
      int codeQueries,
      int heads,
      int words,
      int segments,
      int entries,
      float scale) {

    /**
     * 次元を確かめる。
     *
     * @throws IllegalArgumentException 値が範囲の外のとき、ヘッドの数が奇数のとき、ワードの数が {@link RelationCodes#WORDS} でないとき
     */
    public Dimensions {
      if (rows <= 0
          || queries <= 0
          || queries > MAX_TOKENS
          || keys <= 0
          || keys > MAX_TOKENS
          || codeQueries < queries
          || heads <= 0
          || heads % 2 != 0
          || words != RelationCodes.WORDS
          || segments <= 0
          || segments > MAX_SEGMENTS
          || entries <= 0
          || entries > MAX_ENTRIES
          || !(scale > 0)) {
        throw new IllegalArgumentException(
            "invalid relation attention dimensions: rows=" + rows + " queries=" + queries + " keys=" + keys
                + " codeQueries=" + codeQueries + " heads=" + heads + " words=" + words + " segments=" + segments
                + " entries=" + entries + " scale=" + scale);
      }
    }

    /** クエリと出力の幅。 */
    public int width() {
      return heads * FEATURES;
    }

    /** ρ の行の数。 */
    public int tableRows() {
      return segments * entries;
    }
  }

  /**
   * 順伝播。
   *
   * @param d 次元
   * @param query クエリ
   * @param keyValue キーとバリュー
   * @param mask キーがあるか（0 でなければある）
   * @param codes 関係のコード（INT16）
   * @param table ρ（FP32）
   * @param out 出力の書き先（クエリと同じ型）
   * @param outFloat 逆伝播で使う FP32 の出力の書き先
   * @param logSumExp 逆伝播で使う log Σ exp の書き先（FP32）
   */
  void forward(
      Dimensions d,
      NDArray query,
      NDArray keyValue,
      NDArray mask,
      NDArray codes,
      NDArray table,
      NDArray out,
      NDArray outFloat,
      NDArray logSumExp);

  /**
   * 逆伝播。
   *
   * @param d 次元
   * @param query 順伝播のクエリ
   * @param keyValue 順伝播のキーとバリュー
   * @param mask 順伝播で使った、キーがあるかの印
   * @param codes 順伝播の関係のコード
   * @param table 順伝播の ρ
   * @param outFloat 順伝播の FP32 の出力
   * @param logSumExp 順伝播の log Σ exp
   * @param gradOut 出力の勾配（出力と同じ型）
   * @param gradQuery クエリの勾配の書き先（クエリと同じ型）
   * @param gradKeyValue キーとバリューの勾配の書き先（キーとバリューと同じ型）
   * @param gradTable ρ の勾配の部分和の書き先（FP32）
   */
  void backward(
      Dimensions d,
      NDArray query,
      NDArray keyValue,
      NDArray mask,
      NDArray codes,
      NDArray table,
      NDArray outFloat,
      NDArray logSumExp,
      NDArray gradOut,
      NDArray gradQuery,
      NDArray gradKeyValue,
      NDArray gradTable);
}
