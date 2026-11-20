package com.epsilon.prism.hip;

import ai.djl.ndarray.NDArray;

/**
 * 入力の関係の属性から、関係のバイアスのコードを作る演算。作ったコードは学習の順伝播で {@link RelationAttention} が読む。GPU の実装は
 * {@link HipAttention}（{@code t0_relation.hip}）。
 *
 * <p>式は推論の K0 と同じで、7 つの区分からなる。行の中の全ての対（クエリ i、キー j。0 ≤ i, j &lt; S）について、区分 s のコードを
 * bit 4s から詰めて、[行, S, S, {@value #WORDS}] の INT16 として書く（下位のワードに区分 0〜3、上位のワードに区分 4〜6）。S は行の
 * トークンの軸の長さ。行のトークン（局・席・牌とイベント）より後ろのパディングの位置も同じ式で作る。そのとき属性は全て 0 とし、型は
 * 位置から決める（イベントの型になる）。こうするとネットワークの eager のコードとビットまで同じになる。K0 が使う、行の外のキーの印は使わない。
 */
public interface RelationCodes {

  /** 1 組のコードを入れる INT16 のワードの数。 */
  int WORDS = 2;

  /**
   * 入力の並び（native の TokenLayout・RelationColumns と、行のインデックスの列）。値は推論で WeightPack のヘッダーに書く
   * {@link HipSchema} と同じ。
   *
   * @param seats 1 行の席のトークンの数
   * @param tiles 1 行の牌のトークンの数（牌の関係の表は tiles² 個）
   * @param rowWidth 行のインデックスの列の数
   * @param rowFirstEvent 行のインデックスのうち、最初のイベントの列
   * @param rowEvents 行のインデックスのうち、イベントの数の列
   * @param width 関係の属性の列の数
   * @param tile 関係の属性のうち、牌の列
   * @param owner 関係の属性のうち、持ち主の列
   * @param event 関係の属性のうち、イベントの番号の列
   * @param riichi 関係の属性のうち、持ち主が立直を宣言したイベントの番号の列
   * @param lastDiscard 関係の属性のうち、持ち主の最後の打牌のイベントの番号の列
   * @param link 関係の属性のうち、繋がるイベントの列
   * @param safety 関係の属性のうち、相手から見た安全の列の先頭
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
   * コードを作る。
   *
   * @param layout 入力の並び
   * @param rows 行の数
   * @param events イベントのトークンの数
   * @param tokens 行のトークンの軸の長さ S（どの行でも、局・席・牌とイベントのトークンの数以上）
   * @param relation 関係の属性 [rows·{@link Layout#fixed()} + events, 幅]（UINT8。トークンの軸は型ごとの領域に分かれる: 局・席・牌・イベント）
   * @param rowColumn 行のインデックス [rows, 行の幅]（INT32）
   * @param tileRelations 牌どうしの関係の種類 [牌·牌]（INT32）
   * @param codes コードの書き先 [rows, S, S, {@value #WORDS}]（INT16）
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
