package com.epsilon.prism.model;

/**
 * 学習の段ごとの時間を測る計測点。学習のレーンで、マイクロバッチ 1 つの順伝播・損失・逆伝播の段の切れ目ごとに {@link #mark} を呼ぶと、
 * 前回の呼び出しからの時間がその段に足される。計測するステップでは先にレーンの計算ストリームを待つので、段の時間は、ホストが処理を積む
 * 時間とデバイスの計算時間の長い方になる。実際に測る実装を渡すのは学習の計測（システムプロパティ {@code prism.fitProfile}）のときだけで、
 * ほかは何もしない {@link #NONE} を使う。ネットワークは {@link Pass} からこれを受け取り、Attention の記録（{@link AttentionTape}）は
 * 作るときに受け取る。
 */
public interface FitProbe {

  /** 何もしない計測点（計測しない学習と推論で使う）。 */
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
   * 段の切れ目を記録し、前回の呼び出しからの時間をその段に足す。
   *
   * @param stage 段
   * @param layer 層ごとの段（{@link Stage#layered()}）ならバックボーンの層の番号、ほかは 0
   */
  void mark(Stage stage, int layer);

  /**
   * 層を持たない段の切れ目を記録する。
   *
   * @param stage 段
   */
  default void mark(Stage stage) {
    mark(stage, 0);
  }

  /**
   * 計測用の葉（{@link Context#probeCut}。ヘッドの中の葉と、埋め込みのフィールドを並べた配列）を作り、逆伝播をヘッドの中と埋め込みの段に
   * 分けるかを返す。分けるのは計測のときだけ。
   *
   * @return 分けるなら true
   */
  boolean cuts();

  /** 段。マイクロバッチ 1 つの中で実行する順に並ぶ。層ごとの段は、バックボーンの層ごとに別に数える。 */
  enum Stage {
    /**
     * 次のマイクロバッチのネットワークの入力・インデックス・損失の入力を、非同期のコピーとしてコピー用のストリームに積み、計算ストリームには
     * このマイクロバッチのコピーが終わるのを待つ処理を積む（学習のレーン）。ホストはコピーを待たない。
     */
    TRANSFER(false),
    /** 入力のインデックス（{@link BatchIndex}）。前もって送ったインデックスは受け取るだけで、そうでなければホストで組み立てて同期して送る。 */
    BATCH_INDEX(false),
    /** トークンの埋め込みと、イベントの行への並べ替え。 */
    EMBED(false),
    /** 関係のコード（GPU の学習では T0、ほかは eager の {@link RelationClass}）。 */
    RELATION(false),
    /** 層の入力の葉、Attention の前の正規化、q と kv の射影。 */
    NORM_QKV(true),
    /** 層の Attention（自前の Attention の記録があれば t1 の順伝播、なければ djl-rocm の演算）。 */
    ATTENTION(true),
    /** 層の O 射影・残差・FFN（正規化・up・SiLU・down・残差）。 */
    O_FFN(true),
    /** バックボーンの最後の正規化。 */
    FINAL_NORM(false),
    /** 遷移ごとの Cross-Attention（候補層）。 */
    CANDIDATE(false),
    /** 打牌の値と価値グラフのヘッド（s）。 */
    VALUE_GRAPH(false),
    /** 補助のヘッド（候補の結果と読み）。 */
    AUX_HEADS(false),
    /** 損失（差分整合、行の平均のアンカー、補助の項）。 */
    LOSS(false),
    /**
     * 逆伝播のうち、損失からヘッドの中の計測用の葉とバックボーンの出力の葉までの部分。損失・補助のヘッド・価値グラフ・打牌の値と、候補層の
     * O 射影と FFN を含む。計測用の葉から先は下の 4 つの段に入る。
     */
    HEADS_BACKWARD(false),
    /** 逆伝播のうち、候補層の Cross-Attention の演算だけ（出力の葉へ勾配を注入する。q と K/V の FP32 の入力も葉にする）。 */
    CROSS_ATTENTION_BACKWARD(false),
    /** 逆伝播のうち、Cross-Attention の q（t0 の正規化と射影）と K/V（バックボーンの出力の射影）。FP32 への型の変換も含む。 */
    CROSS_PROJECTION_BACKWARD(false),
    /** 逆伝播のうち、候補層の t0（遷移のフィールドを並べる処理、W_X・W_E、τ の有無の掛け算）。 */
    CANDIDATE_START_BACKWARD(false),
    /**
     * 逆伝播のうち、バックボーンの出力から行を集める gather（候補層の y_T[τ]、補助のヘッドの判断の行、読みの行）から、バックボーンの出力の
     * 葉までの部分。
     */
    GATHER_BACKWARD(false),
    /** 逆伝播のうち、層の O 射影・残差・FFN（最後の層ではバックボーンの最後の正規化も含む）。 */
    O_FFN_BACKWARD(true),
    /** 逆伝播のうち、層の Attention の自前の逆伝播（t1 の演算だけ）。 */
    ATTENTION_BACKWARD(true),
    /** 逆伝播のうち、ρ の部分和の合計、q・kv・ρ への勾配の注入（q と kv の射影）、Attention の前の正規化。 */
    NORM_QKV_BACKWARD(true),
    /** 逆伝播のうち、埋め込み（型ごとの Linear、イベントの行への並べ替え、トークンの連結）。フィールドを並べる処理は次の段に入る。 */
    EMBED_BACKWARD(false),
    /** 逆伝播のうち、埋め込みのフィールドを並べる処理（型ごとの表の行の勾配への足し込み）。 */
    EMBED_PACK_BACKWARD(false),
    /** 逆伝播の全体。自前の Attention の記録がないレーンで使い、段に分けない。 */
    BACKWARD(false),
    /** マイクロバッチの配列と勾配のコレクターを閉じる。損失を回収するステップでは、回収を待つ時間もここに入る。 */
    RELEASE(false);

    private final boolean layered;

    Stage(boolean layered) {
      this.layered = layered;
    }

    /**
     * バックボーンの層ごとに数える段かを返す。
     *
     * @return 層ごとの段なら true
     */
    public boolean layered() {
      return layered;
    }
  }
}
