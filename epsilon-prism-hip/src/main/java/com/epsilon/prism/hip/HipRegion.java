package com.epsilon.prism.hip;

/**
 * 計画の作業域の区画（native の順と同じ）。段の境目で全体メモリに残すものだけを持つ。トークンの区画の並びは入力のトークンの軸と
 * 同じ（局 R・席 4R・牌 34R・出来事 N_E）。
 */
public enum HipRegion {
  /** 入力の blob（UINT8）。forward は H2D をここへ 1 回。 */
  INPUT(HipElement.U8),
  /** 残差の流れ x {@code [トークン, H]}（BF16）。K1 が書き、K3・K5 が読み、K3 が上書きする。 */
  RESIDUAL(HipElement.BF16),
  /**
   * 次の層の注意の入力 n = ⌊(x·r)·g⌉ {@code [トークン, H]}（BF16、RMSNorm をかけた残差の流れ）。K1（層 0）と K3（次の層）が書き、K2・K4 が
   * A の断片として読む。
   */
  NORMALIZED(HipElement.BF16),
  /** 注意の出力 o {@code [トークン, A]}（BF16）。K2・K4 が書き、K3・K5 が読む。 */
  ATTENTION(HipElement.BF16),
  /**
   * 関係の符号 {@code [行, 塊, 塊, 256]}（7 区画を 1 語に詰めた INT32。行ごとに {@link HipShape#RELATION_CODE_WORDS} 語で、16 トークンの
   * 塊の対ごとに 256 語）。K0 が書き、K2・K4 が読む。
   */
  RELATION_CODE(HipElement.I32),
  /** 最後の正規化の後の局・席・牌 y {@code [39R, H]}（BF16、トークンの軸の先頭 39R と同じ並び）。K5 が書き、K6（牌）・K7（局）が読む。 */
  STATE(HipElement.BF16),
  /** 候補層の交差注意の鍵と値 {@code [R, 39, 2A]}（BF16、行ごとに局・席・牌の順）。K5 が書き、K6 が読む。 */
  MEMORY(HipElement.BF16),
  /** 遷移の静的な部分 t_s = ⌊W_X·pack + b_X⌉ {@code [遷移, H]}（BF16）。K1 が書き、K6 が読む。 */
  STATIC(HipElement.BF16),
  /** 遷移の表現 z {@code [遷移, H]}（BF16）。K6 が書き、K7 が読む。 */
  TRANSITION(HipElement.BF16),
  /** 打牌の値 B {@code [遷移]}（FP32）。K6 が書き、K7 が読む。 */
  VALUE(HipElement.FP32),
  /** s {@code [手]}（FP32、要求の順）。K7 が書き、forward は D2H をここから 1 回。 */
  SCORES(HipElement.FP32);

  private final HipElement element;

  HipRegion(HipElement element) {
    this.element = element;
  }

  public HipElement element() {
    return element;
  }

  /** 計画の容量でのバイト数。 */
  public long capacityBytes(HipShape shape) {
    HipShape.Capacity capacity = shape.capacity();
    return switch (this) {
      case INPUT -> capacity.inputBytes();
      case RESIDUAL, NORMALIZED -> shape.tokens() * shape.hidden() * element.bytes();
      case ATTENTION -> shape.tokens() * shape.attention() * element.bytes();
      case RELATION_CODE -> (long) capacity.rows() * HipShape.RELATION_CODE_WORDS * element.bytes();
      case STATE -> shape.fixedTokenCapacity() * shape.hidden() * element.bytes();
      case MEMORY -> shape.fixedTokenCapacity() * 2L * shape.attention() * element.bytes();
      case STATIC, TRANSITION -> (long) capacity.transitions() * shape.hidden() * element.bytes();
      case VALUE -> (long) capacity.transitions() * element.bytes();
      case SCORES -> (long) capacity.actions() * element.bytes();
    };
  }

  /** 塊が使うバイト数（区画の先頭から）。 */
  public long bytes(HipShape shape, HipBatch batch) {
    long fixed = (long) shape.fixedTokens() * batch.rows();
    long tokens = fixed + batch.events();
    return switch (this) {
      case INPUT -> batch.bytes();
      case RESIDUAL, NORMALIZED -> tokens * shape.hidden() * element.bytes();
      case ATTENTION -> tokens * shape.attention() * element.bytes();
      case RELATION_CODE -> (long) batch.rows() * HipShape.RELATION_CODE_WORDS * element.bytes();
      case STATE -> fixed * shape.hidden() * element.bytes();
      case MEMORY -> fixed * 2L * shape.attention() * element.bytes();
      case STATIC, TRANSITION -> (long) batch.transitions() * shape.hidden() * element.bytes();
      case VALUE -> (long) batch.transitions() * element.bytes();
      case SCORES -> (long) batch.actions() * element.bytes();
    };
  }
}
