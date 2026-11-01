package com.epsilon.prism.model;

/**
 * 当てはめの段の時計の印。学習のレーンの 1 微小バッチの順伝播・損失・逆伝播の区切りで {@link #mark} を呼ぶと、前の印からの時間がその段に
 * 入る（計測の同期する歩では先にレーンの計算ストリームを待つので、段の時間はホストの積み込みとデバイスの計算の長い方になる）。実体を渡すのは
 * 当てはめの計測（系の性質 {@code prism.fitProfile}）だけで、ほかは {@link #NONE} で何もしない。網は {@link Pass} から、
 * 注意の記録（{@link AttentionTape}）は作るときに受け取る。
 */
public interface FitProbe {

  /** 何もしない印（計測しない学習と推論）。 */
  FitProbe NONE =
      new FitProbe() {
        @Override
        public void mark(Stage stage, int layer) {}

        @Override
        public boolean cuts() {
          return false;
        }
      };

  /**
   * 段の区切り（前の印からの時間を段に足す）。
   *
   * @param stage 段
   * @param layer 層の段（{@link Stage#layered()}）は背骨の層の番号、ほかは 0
   */
  void mark(Stage stage, int layer);

  /**
   * 層を持たない段の区切り。
   *
   * @param stage 段
   */
  default void mark(Stage stage) {
    mark(stage, 0);
  }

  /**
   * 背骨の出力も葉にして、損失の逆伝播を頭の側（{@link Stage#HEADS_BACKWARD}）と背骨の側に分けるなら true（計測のときだけ）。
   *
   * @return 分けるなら true
   */
  boolean cuts();

  /** 段（1 微小バッチの中の順）。層の段は背骨の層ごとに数える。 */
  enum Stage {
    /** 前もって並べた網の入力・索引・損失の入力の非同期の写しを積む（学習のレーン。ホストは写しを待たない）。 */
    TRANSFER(false),
    /** 入力の索引（{@link BatchIndex}。前もって送った索引は受け取るだけ、ほかはホストの組み立てと同期の送り）。 */
    BATCH_INDEX(false),
    /** トークンの埋め込みと、出来事の行への並べ替え。 */
    EMBED(false),
    /** 関係の符号（GPU の学習は T0、ほかは eager の {@link RelationClass}）。 */
    RELATION(false),
    /** 層の入力の葉・注意の前の正規化・q と kv の射影。 */
    NORM_QKV(true),
    /** 層の注意（自前の注意の記録があれば t1 の順伝播、なければ djl-rocm の演算）。 */
    ATTENTION(true),
    /** 層の O 射影・残差・FFN（正規化・up・SiLU・down・残差）。 */
    O_FFN(true),
    /** 背骨の最後の正規化。 */
    FINAL_NORM(false),
    /** 遷移ごとの交差注意（候補層）。 */
    CANDIDATE(false),
    /** 打牌の値と価値グラフの頭（s）。 */
    VALUE_GRAPH(false),
    /** 補助の頭（候補の結果と読み）。 */
    AUX_HEADS(false),
    /** 損失（差分整合・行の平均の錨・補助の項）。 */
    LOSS(false),
    /** 逆伝播: 損失から背骨の出力の葉まで（損失・補助の頭・価値グラフ・候補層）。 */
    HEADS_BACKWARD(false),
    /** 逆伝播: 層の O 射影・残差・FFN（最後の層は背骨の最後の正規化も含む）。 */
    O_FFN_BACKWARD(true),
    /** 逆伝播: 層の注意の自前の逆伝播（t1 の演算だけ）。 */
    ATTENTION_BACKWARD(true),
    /** 逆伝播: ρ の部分和の和、q・kv・ρ への注入（q と kv の射影）、注意の前の正規化。 */
    NORM_QKV_BACKWARD(true),
    /** 逆伝播: 埋め込み。 */
    EMBED_BACKWARD(false),
    /** 逆伝播の全体（自前の注意の記録のないレーン。段に分けない）。 */
    BACKWARD(false),
    /** 微小バッチの配列と勾配の集め手を閉じる（損失を回収する歩は回収の待ちも）。 */
    RELEASE(false);

    private final boolean layered;

    Stage(boolean layered) {
      this.layered = layered;
    }

    /**
     * 背骨の層ごとに数える段なら true を返す。
     *
     * @return 層の段なら true
     */
    public boolean layered() {
      return layered;
    }
  }
}
