package com.epsilon.prism.hip;

import ai.djl.ndarray.NDArray;

/**
 * 関係の偏りつきの注意の、順伝播と逆伝播の演算（学習で DJL の自動微分の外に置く演算）。GPU は {@link HipAttention}。
 *
 * <p>数式: ℓ_ij = (q_i·k_j)·scale + Σ_s ρ[entries·s + c_s(i, j), h]（鍵 j が在るときだけ）、o_i = Σ_j softmax_j(ℓ_i) v_j。区画 s の符号
 * c_s は、組の語 s/4 の 4(s%4) ビット目から 4 ビット。逆伝播は出力の勾配から q・kv・ρ の勾配を作る（q・kv の勾配は FP32 で作って q・kv の
 * 型に最近接偶数で 1 回丸める。ρ は作業群（行 × 頭）ごとの部分和で、呼び手が行で足す）。配列の形は {@link Dimensions} の約束どおり。
 */
public interface RelationAttention {

  /** 行のトークンの上限（native の kAttentionMaxTokens）。 */
  int MAX_TOKENS = 160;

  /** 頭の幅（native の kAttentionFeatures）。 */
  int FEATURES = 16;

  /** 関係の区画の上限（native の kAttentionMaxSegments）。 */
  int MAX_SEGMENTS = 8;

  /** 区画あたりの値の上限（4 ビットの符号）。 */
  int MAX_ENTRIES = 16;

  /**
   * 注意の数。配列の形: q・out・outFloat・gradOut・gradQuery は [rows, queries, width]、kv・gradKeyValue は [rows, keys, 2·width]
   * （鍵の後に値）、mask は [rows, keys]、codes は [rows, codeQueries, keys, words]（先頭の queries 個を読む）、ρ は
   * [tableRows, heads]、logSumExp は [rows, heads, queries]、gradTable は [rows·heads, tableRows]。
   *
   * @param rows 行の数
   * @param queries 問い合わせの数
   * @param keys 鍵の数
   * @param codeQueries 関係の符号の問い合わせの軸の長さ
   * @param heads 頭の数（幅は heads·{@value #FEATURES}。GPU の順伝播は頭の対で作業群を作るので偶数）
   * @param words 1 組の符号の INT16 の語の数（{@link RelationCodes#WORDS}。GPU は 1 組の符号を 32 bit の 1 語として読む）
   * @param segments 関係の区画の数
   * @param entries 区画あたりの値の数（ρ の行は segments·entries）
   * @param scale 内積の倍率
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
     * 数を確かめる。
     *
     * @throws IllegalArgumentException 範囲の外か、頭の数が奇数か、語の数が {@link RelationCodes#WORDS} でないとき
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
        throw new IllegalArgumentException("invalid relation attention dimensions " + this);
      }
    }

    /** 問い合わせ・出力の幅。 */
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
   * @param d 数
   * @param query 問い合わせ
   * @param keyValue 鍵と値
   * @param mask 鍵の在否（0 でなければ在る）
   * @param codes 関係の符号（INT16）
   * @param table ρ（FP32）
   * @param out 出力（問い合わせと同じ型、書く）
   * @param outFloat 逆伝播のための FP32 の出力（書く）
   * @param logSumExp 逆伝播のための log Σ exp（FP32、書く）
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
   * @param d 数
   * @param query 順伝播の問い合わせ
   * @param keyValue 順伝播の鍵と値
   * @param mask 順伝播の鍵の在否
   * @param codes 順伝播の関係の符号
   * @param table 順伝播の ρ
   * @param outFloat 順伝播の FP32 の出力
   * @param logSumExp 順伝播の log Σ exp
   * @param gradOut 出力の勾配（出力と同じ型）
   * @param gradQuery 問い合わせの勾配（問い合わせと同じ型、書く）
   * @param gradKeyValue 鍵と値の勾配（鍵と値と同じ型、書く）
   * @param gradTable ρ の勾配の部分和（FP32、書く）
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
