package com.epsilon.prism.hip;

/**
 * 推論のカーネル。並びは native と同じ。{@link HipPlan#kernelMillis()} もこの順に並び、層ごとに回すカーネルの時間は全ての層の和になる。
 * 順伝播は、形の記述から次の順に組む: 埋め込み → 関係のコード → 全トークンの層（Attention → MLP）→ 局・席・牌の層（Attention →
 * MLP。最後の層だけは最後の MLP）→ 2 段目のキーとバリュー → 候補 → 2 段目 → 価値グラフ。2 段目のない形では、2 段目の 2 つを飛ばす。
 */
public enum HipKernel {
  /** K1: トークンの埋め込み、遷移の静的な部分（折り込んだ組の表の和＋数値の Linear＋型）、層 0 の q・k・v を作る。 */
  EMBED,
  /**
   * K0: 行の中の全てのトークンの対について、関係のコード（{@link HipRegion#RELATION_CODE}）を作る。K2・K4 が使う、行の組ごとに
   * 並べた行の番号も書く（その時間もこのカーネルに含める）。
   */
  RELATION,
  /** K2: 全トークンの層の Attention（全てのトークンがクエリになる）。行の組の数だけ起動する。 */
  ATTENTION_ALL,
  /**
   * K3: 全トークンの層の O 射影＋残差 → RMSNorm → FFN → 残差を計算し、次の層の Attention の入力 n を全てのトークンについて書く。
   * 最後の全トークンの層では、x は局・席・牌の分だけ書き、2 つ目以降の局・席・牌の層と 2 段目が使うイベントのキーとバリューも書く。
   */
  MLP_ALL,
  /**
   * K4: 局・席・牌の層の Attention（クエリは局・席・牌のトークン）。最初の局・席・牌の層は、イベントのキーとバリューも n から射影する。
   * 後の層は、最後の全トークンの層の K3 が作ったものを写す。行の組の数だけ起動する。
   */
  ATTENTION_FIXED,
  /** K3: 最後の層を除く局・席・牌の層の MLP。局・席・牌のトークンだけを処理し、型ごとの FFN なら組ごとの範囲に分けて回す。 */
  MLP_FIXED,
  /** K5: 最後の層の MLP、最後の正規化、候補層の Cross-Attention のキーとバリュー。 */
  MLP_LAST,
  /** K8: 2 段目が読む局・席・牌のキーとバリュー（最後の正規化の後の y から作る）。 */
  REREAD_KEY_VALUE,
  /** K6: 候補層（遷移ごとの Cross-Attention と FFN）。1 段の形では B も書く。 */
  CANDIDATE,
  /** K9: 候補層の 2 段目（遷移ごとに局・席・牌とイベントを読み直す Attention）と B。 */
  REREAD,
  /** K7: 価値グラフ（集約・ゲート・合成）。s を要求の順に詰めて書く。 */
  VALUE_GRAPH;

  /**
   * この形で、このカーネルがその層を回すかを返す。層を持たないカーネルは層 0 だけを受ける。
   *
   * @param layer 層
   * @param architecture 形の記述
   * @return 回すなら true
   */
  public boolean acceptsLayer(int layer, HipArchitecture architecture) {
    boolean inside = layer >= 0 && layer < architecture.depth();
    return switch (this) {
      case ATTENTION_ALL, MLP_ALL -> inside && architecture.layer(layer).role() == LayerRole.ALL_TOKENS;
      case ATTENTION_FIXED -> inside && architecture.layer(layer).role() == LayerRole.FIXED_TOKENS;
      case MLP_FIXED ->
          inside && architecture.layer(layer).role() == LayerRole.FIXED_TOKENS && layer < architecture.depth() - 1;
      case MLP_LAST -> layer == architecture.depth() - 1;
      case REREAD_KEY_VALUE, REREAD -> layer == 0 && architecture.reread();
      case EMBED, RELATION, CANDIDATE, VALUE_GRAPH -> layer == 0;
    };
  }
}
