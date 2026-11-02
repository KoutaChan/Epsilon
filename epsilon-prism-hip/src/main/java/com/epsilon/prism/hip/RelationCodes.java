package com.epsilon.prism.hip;

import ai.djl.ndarray.NDArray;

/**
 * 関係の偏りの符号を、入力の関係の属性から作る演算（学習の順伝播で {@link RelationAttention} が読む符号）。GPU は {@link HipAttention}
 * （{@code t0_relation.hip}）。
 *
 * <p>式は推論の K0 と同じ 7 区画。行の全ての対 (問い合わせ i, 鍵 j)（0 ≤ i, j &lt; S）の符号を区画 s を bit 4s から 1 語に
 * 詰め、[行, S, S, {@value #WORDS}] の INT16（下位の語に区画 0〜3、上位の語に区画 4〜6）として書く。S は行のトークンの軸の長さで、行の
 * トークン（局・席・牌と出来事）より後ろの詰め物の位置は、属性を全て 0、型を位置（出来事）で決めて同じ式で作る（網の eager の符号と
 * 同じビット。K0 の行の外の鍵の印は使わない）。
 */
public interface RelationCodes {

  /** 1 組の符号の INT16 の語の数。 */
  int WORDS = 2;

  /**
   * 入力の並び（native の TokenLayout・RelationColumns と行の索引の列。推論の WeightPack の頭の {@link HipSchema} と同じ値）。
   *
   * @param seats 行ごとの席のトークンの数
   * @param tiles 行ごとの牌のトークンの数（牌の関係の表は tiles² 個）
   * @param rowWidth 行の索引の列の数
   * @param rowFirstEvent 行の索引の列: 最初の出来事
   * @param rowEvents 行の索引の列: 出来事の数
   * @param width 関係の属性の列の数
   * @param tile 関係の属性の列: 牌
   * @param owner 関係の属性の列: 持ち主
   * @param event 関係の属性の列: 出来事の番号
   * @param riichi 関係の属性の列: 持ち主の立直宣言の番号
   * @param lastDiscard 関係の属性の列: 持ち主の最後の打牌の番号
   * @param link 関係の属性の列: 繋がる出来事
   * @param safety 関係の属性の列: 相手から見た安全の最初の列
   * @param safeties 相手から見た安全の列の数
   */
  record Layout(
      int seats,
      int tiles,
      int rowWidth,
      int rowFirstEvent,
      int rowEvents,
      int width,
      int tile,
      int owner,
      int event,
      int riichi,
      int lastDiscard,
      int link,
      int safety,
      int safeties) {

    /** 全ての行にある局・席・牌のトークンの数。 */
    public int fixed() {
      return 1 + seats + tiles;
    }
  }

  /**
   * 符号を作る。
   *
   * @param layout 入力の並び
   * @param rows 行の数
   * @param events 出来事のトークンの数
   * @param tokens 行のトークンの軸の長さ S（どの行の局・席・牌と出来事の数より小さくない）
   * @param relation 関係の属性 [rows·{@link Layout#fixed()} + events, 幅]（UINT8。トークンの軸は型ごとの区画: 局・席・牌・出来事）
   * @param rowColumn 行の索引 [rows, 行の幅]（INT32）
   * @param tileRelations 牌の関係の類 [牌·牌]（INT32）
   * @param codes 符号 [rows, S, S, {@value #WORDS}]（INT16、書く）
   */
  void codes(
      Layout layout,
      int rows,
      int events,
      int tokens,
      NDArray relation,
      NDArray rowColumn,
      NDArray tileRelations,
      NDArray codes);
}
