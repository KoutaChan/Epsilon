package com.epsilon.prism.hip;

/**
 * 1 つの計画の形。幅はカーネルの特化に合わせ、容量は {@link HipPlan#create} で一度だけ取る作業域と重みの置き場の大きさを決める。
 *
 * @param hidden 流れの幅 H。カーネルは {@link #HIDDEN} だけを特化している
 * @param ffn FFN の幅 F（{@link #FFN_CHUNK} の倍数。背骨と候補層で共通）
 * @param gate 頭の隠れ幅 G（{@link #GATE_STEP} の倍数。B と門で共通）
 * @param depth 背骨の層の数 L（{@link #MIN_DEPTH}〜{@link #MAX_DEPTH}。最後の層だけ問い合わせを局・席・牌に限る）
 * @param epsilon RMSNorm の ε（網の約束では 1e-5·256/H）
 * @param fixedTokens 行ごとに必ずあるトークン（局・席・牌）の数
 * @param capacity 1 塊の容量
 * @param weightBytes WeightPack の容量（バイト）
 */
public record HipShape(
    int hidden, int ffn, int gate, int depth, float epsilon, int fixedTokens, Capacity capacity, long weightBytes) {

  /** カーネルが特化している H。 */
  public static final int HIDDEN = 256;

  public static final int HEAD_WIDTH = 16;

  /** K3・K5・K6 が F を回す段の幅（隠れ h を LDS に置く列）。 */
  public static final int FFN_CHUNK = 128;

  /** G の倍数の決まり（B と門の WMMA の列の段）。 */
  public static final int GATE_STEP = 64;

  public static final int MIN_DEPTH = 2;

  /** 背骨の層の数の上限（WeightPack の層ごとの置き場の数）。 */
  public static final int MAX_DEPTH = 3;

  /** 1 行のトークンの上限（K2・K4 の LDS の行。局・席・牌 39 と公開の出来事 110 の和 149 を入れる）。 */
  public static final int MAX_ROW_TOKENS = 160;

  /** 1 行の局・席・牌のトークンの上限（K4 の LDS の問い合わせの行。39 を 16 の倍数に）。 */
  public static final int MAX_FIXED_TOKENS = 48;

  /** 行ごとの関係の符号の語（16 トークンの塊の対ごとに 256 語）。native の codeIndex と同じ置き方。 */
  public static final int RELATION_CODE_WORDS = (MAX_ROW_TOKENS / 16) * (MAX_ROW_TOKENS / 16) * 256;

  /**
   * 1 塊の容量。
   *
   * @param rows 行
   * @param events 出来事のトークン
   * @param transitions 遷移
   * @param actions 合法手
   * @param inputBytes blob のバイト数
   */
  public record Capacity(int rows, int events, int transitions, int actions, long inputBytes) {

    /**
     * 容量を確かめる。
     *
     * @throws IllegalArgumentException 正でない容量のとき
     */
    public Capacity {
      if (rows <= 0 || events < 0 || transitions < rows || actions < rows || inputBytes <= 0) {
        throw new IllegalArgumentException(
            "invalid capacity: rows="
                + rows
                + " events="
                + events
                + " transitions="
                + transitions
                + " actions="
                + actions
                + " inputBytes="
                + inputBytes);
      }
    }
  }

  /**
   * 形を確かめる。
   *
   * @throws IllegalArgumentException 特化していない幅か、倍数の決まり・深さ・容量を外れるとき
   */
  public HipShape {
    if (hidden != HIDDEN) {
      throw new IllegalArgumentException("HIP kernels are specialised for hidden " + HIDDEN + " only, got " + hidden);
    }
    if (ffn <= 0 || ffn % FFN_CHUNK != 0) {
      throw new IllegalArgumentException("ffn must be a positive multiple of " + FFN_CHUNK + ", got " + ffn);
    }
    if (gate <= 0 || gate % GATE_STEP != 0) {
      throw new IllegalArgumentException("gate must be a positive multiple of " + GATE_STEP + ", got " + gate);
    }
    if (depth < MIN_DEPTH || depth > MAX_DEPTH) {
      throw new IllegalArgumentException("depth must be " + MIN_DEPTH + ".." + MAX_DEPTH + ", got " + depth);
    }
    if (!(epsilon > 0f) || !Float.isFinite(epsilon)) {
      throw new IllegalArgumentException("epsilon must be positive and finite, got " + epsilon);
    }
    if (fixedTokens <= 0 || fixedTokens > MAX_FIXED_TOKENS) {
      throw new IllegalArgumentException(
          "fixed tokens per row must be 1.." + MAX_FIXED_TOKENS + ", got " + fixedTokens);
    }
    if (capacity == null) {
      throw new IllegalArgumentException("capacity is required");
    }
    if (weightBytes <= 0 || weightBytes > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "weight pack capacity must be 1.." + Integer.MAX_VALUE + " bytes, got " + weightBytes);
    }
  }

  /** 注意の幅 A = H/2（頭の幅 16 × H/32 頭）。 */
  public int attention() {
    return hidden / 2;
  }

  /** 頭の数 H/32。 */
  public int heads() {
    return attention() / HEAD_WIDTH;
  }

  /** K2・K4 の作業群が受け持つ頭の対の数 H/64。 */
  public int headPairs() {
    return heads() / 2;
  }

  /** 1 塊のトークンの容量（局・席・牌の全ての行と出来事）。 */
  public long tokens() {
    return (long) fixedTokens * capacity.rows() + capacity.events();
  }

  /** 1 塊の局・席・牌のトークンの容量。 */
  public long fixedTokenCapacity() {
    return (long) fixedTokens * capacity.rows();
  }

  /**
   * 塊がこの計画の容量に入ることを確かめる。
   *
   * @param batch 塊
   * @throws IllegalArgumentException 容量を超えるとき
   */
  public void requireFits(HipBatch batch) {
    if (batch.rows() > capacity.rows()
        || batch.events() > capacity.events()
        || batch.transitions() > capacity.transitions()
        || batch.actions() > capacity.actions()
        || batch.bytes() > capacity.inputBytes()) {
      throw new IllegalArgumentException(batch + " exceeds the plan capacity " + capacity);
    }
  }
}
