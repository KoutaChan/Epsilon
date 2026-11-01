package com.epsilon.prism.hip;

/**
 * 打つ網のカーネル（K1〜K7 と関係の符号の K0）。{@link HipPlan#kernelMillis()} はこの順に並び、native の順と同じ（読み込みで
 * 数を照らす）。
 */
public enum HipKernel {
  /** K1: トークンの埋め込みと遷移の静的な部分（畳んだ組の表の和＋数値の Linear＋型）。 */
  EMBED,
  /** K0: 行の全ての対の関係の符号（{@link HipRegion#RELATION_CODE}）。K2・K4 の全ての頭の対と層が読む。 */
  RELATION,
  /** K2: 最後でない層の注意（問い合わせは全てのトークン）。層ごとに 1 回。 */
  ATTENTION0,
  /** K3: 最後でない層の O 射影＋残差 → RMSNorm → FFN → 残差（全てのトークン）。層ごとに 1 回。 */
  MLP0,
  /** K4: 最後の層の注意（問い合わせは局・席・牌）。 */
  ATTENTION1,
  /** K5: 最後の層の MLP、最後の正規化、交差注意の鍵と値。 */
  MLP1_KV,
  /** K6: 候補（遷移ごとの交差注意と FFN）と B。 */
  CANDIDATE,
  /** K7: 価値グラフ（集約・門・合成）。s を要求の順に詰める。 */
  VALUE_GRAPH;

  /** 深さ L の網で、この kernel を回す層（K2・K3 は最後でない層、K4・K5 は最後の層、ほかは層を持たない 0）か。 */
  public boolean acceptsLayer(int layer, int depth) {
    return switch (this) {
      case ATTENTION0, MLP0 -> layer >= 0 && layer < depth - 1;
      case ATTENTION1, MLP1_KV -> layer == depth - 1;
      case EMBED, RELATION, CANDIDATE, VALUE_GRAPH -> layer == 0;
    };
  }
}
