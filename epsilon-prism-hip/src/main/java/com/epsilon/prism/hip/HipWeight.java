package com.epsilon.prism.hip;

/**
 * カーネルが読む重みの置き場（native の順と同じ）。行列は BF16 で DJL の Linear と同じ {@code [out, in]}、偏りは FP32（eager の
 * Linear に合わせて BF16 に丸めた値）、利得・関係の偏り・符号の表・頭の最後の層は FP32 のまま。
 *
 * <p>層ごとの置き場（{@link #layered()}）は背骨の層ごとに 1 つずつ持つ。埋め込みの表と型ごとの Linear は畳んだ組の表として
 * {@link HipEmbedding} が置き場を持つ。
 */
public enum HipWeight {
  /** 牌の関係の類 {@code [34·34]}（UINT8、MahjongTileRelation の ordinal）。 */
  TILE_RELATION(HipElement.U8, false),
  /** 注意の前の RMSNorm の利得 {@code [H]}。 */
  ATTENTION_NORM(HipElement.FP32, true),
  /** q・k・v の射影 {@code [3A, H]}（q・k・v の塊を頭の順に）。 */
  QKV(HipElement.BF16, true),
  /** q・k・v の偏り {@code [3A]}。 */
  QKV_BIAS(HipElement.FP32, true),
  /** 関係の偏りの表 ρ {@code [112, 頭]}。 */
  RELATION_BIAS(HipElement.FP32, true),
  /** O 射影 {@code [H, A]}。 */
  OUTPUT(HipElement.BF16, true),
  /** O 射影の偏り {@code [H]}。 */
  OUTPUT_BIAS(HipElement.FP32, true),
  /** FFN の前の RMSNorm の利得 {@code [H]}。 */
  FFN_NORM(HipElement.FP32, true),
  /** 上り {@code [F, H]}。 */
  UP(HipElement.BF16, true),
  /** 上りの偏り {@code [F]}。 */
  UP_BIAS(HipElement.FP32, true),
  /** 下り {@code [H, F]}。 */
  DOWN(HipElement.BF16, true),
  /** 下りの偏り {@code [H]}。 */
  DOWN_BIAS(HipElement.FP32, true),
  /** 最後の RMSNorm の利得 {@code [H]}。 */
  FINAL_NORM(HipElement.FP32, false),
  /** 候補の牌の項 W_E {@code [H, H]}（偏りなし）。 */
  TILE_ENTRY(HipElement.BF16, false),
  /** 候補の問い合わせの前の RMSNorm の利得 {@code [H]}。 */
  QUERY_NORM(HipElement.FP32, false),
  /** 候補の問い合わせ {@code [A, H]}。 */
  QUERY(HipElement.BF16, false),
  /** 候補の問い合わせの偏り {@code [A]}。 */
  QUERY_BIAS(HipElement.FP32, false),
  /** 交差注意の鍵と値の射影 {@code [2A, H]}。 */
  KEY_VALUE(HipElement.BF16, false),
  /** 交差注意の鍵と値の偏り {@code [2A]}。 */
  KEY_VALUE_BIAS(HipElement.FP32, false),
  /** 鍵の符号の鍵の差分 E_k {@code [符号, A]}。 */
  CODE_KEY(HipElement.FP32, false),
  /** 鍵の符号の値の差分 E_v {@code [符号, A]}。 */
  CODE_VALUE(HipElement.FP32, false),
  /** 候補の O 射影 {@code [H, A]}。 */
  CANDIDATE_OUTPUT(HipElement.BF16, false),
  /** 候補の O 射影の偏り {@code [H]}。 */
  CANDIDATE_OUTPUT_BIAS(HipElement.FP32, false),
  /** 候補の FFN の前の RMSNorm の利得 {@code [H]}。 */
  CANDIDATE_FFN_NORM(HipElement.FP32, false),
  /** 候補の上り {@code [F, H]}。 */
  CANDIDATE_UP(HipElement.BF16, false),
  /** 候補の上りの偏り {@code [F]}。 */
  CANDIDATE_UP_BIAS(HipElement.FP32, false),
  /** 候補の下り {@code [H, F]}。 */
  CANDIDATE_DOWN(HipElement.BF16, false),
  /** 候補の下りの偏り {@code [H]}。 */
  CANDIDATE_DOWN_BIAS(HipElement.FP32, false),
  /** B の RMSNorm の利得 {@code [H]}。 */
  DISCARD_NORM(HipElement.FP32, false),
  /** B の隠れ W_B {@code [G, H]}。 */
  DISCARD_HIDDEN(HipElement.BF16, false),
  /** B の隠れの偏り c_B {@code [G]}。 */
  DISCARD_HIDDEN_BIAS(HipElement.FP32, false),
  /** B の最後の層 wB {@code [G]}。 */
  DISCARD_OUTPUT(HipElement.FP32, false),
  /** B の道の係数 uBx {@code [B の道]}。 */
  DISCARD_EXACT(HipElement.FP32, false),
  /** 門の RMSNorm の利得 {@code [H]}（e_a と e_base で共通）。 */
  GATE_NORM(HipElement.FP32, false),
  /** 門の隠れ W_g {@code [G, 3H]}。 */
  GATE_HIDDEN(HipElement.BF16, false),
  /** 門の隠れの偏り c {@code [G]}。 */
  GATE_HIDDEN_BIAS(HipElement.FP32, false),
  /** 門の種類ごとの偏り β {@code [種類, G]}。 */
  GATE_KIND_OFFSET(HipElement.FP32, false),
  /** 門の道の射影 U {@code [種類, G, 門の道]}。 */
  GATE_EXACT_PROJECTION(HipElement.FP32, false),
  /** 門の最後の層 w {@code [種類, G]}。 */
  GATE_OUTPUT(HipElement.FP32, false),
  /** 門の最後の偏り b {@code [種類]}。 */
  GATE_OUTPUT_BIAS(HipElement.FP32, false),
  /** 門の道の係数 u {@code [種類, 門の道]}。 */
  GATE_EXACT_OUTPUT(HipElement.FP32, false);

  private final HipElement element;
  private final boolean layered;

  HipWeight(HipElement element, boolean layered) {
    this.element = element;
    this.layered = layered;
  }

  public HipElement element() {
    return element;
  }

  /** 背骨の層ごとに置き場を持つなら true。 */
  public boolean layered() {
    return layered;
  }

  public long elements(HipDimensions d) {
    long h = d.hidden();
    long a = d.attention();
    long f = d.ffn();
    long g = d.gate();
    long kinds = d.gateKinds();
    return switch (this) {
      case TILE_RELATION -> (long) d.tiles() * d.tiles();
      case ATTENTION_NORM,
          OUTPUT_BIAS,
          FFN_NORM,
          DOWN_BIAS,
          FINAL_NORM,
          QUERY_NORM,
          CANDIDATE_OUTPUT_BIAS,
          CANDIDATE_FFN_NORM,
          CANDIDATE_DOWN_BIAS,
          DISCARD_NORM,
          GATE_NORM -> h;
      case QKV -> 3 * a * h;
      case QKV_BIAS -> 3 * a;
      case RELATION_BIAS -> (long) d.relationRows() * d.heads();
      case OUTPUT, CANDIDATE_OUTPUT -> h * a;
      case UP, CANDIDATE_UP -> f * h;
      case UP_BIAS, CANDIDATE_UP_BIAS -> f;
      case DOWN, CANDIDATE_DOWN -> h * f;
      case TILE_ENTRY -> h * h;
      case QUERY -> a * h;
      case QUERY_BIAS -> a;
      case KEY_VALUE -> 2 * a * h;
      case KEY_VALUE_BIAS -> 2 * a;
      case CODE_KEY, CODE_VALUE -> d.codes() * a;
      case DISCARD_HIDDEN -> g * h;
      case DISCARD_HIDDEN_BIAS, DISCARD_OUTPUT, GATE_HIDDEN_BIAS -> g;
      case DISCARD_EXACT -> d.discardExact();
      case GATE_HIDDEN -> g * 3 * h;
      case GATE_KIND_OFFSET, GATE_OUTPUT -> kinds * g;
      case GATE_EXACT_PROJECTION -> kinds * g * d.actionExact();
      case GATE_OUTPUT_BIAS -> kinds;
      case GATE_EXACT_OUTPUT -> kinds * d.actionExact();
    };
  }

  public long bytes(HipDimensions d) {
    return elements(d) * element.bytes();
  }
}
