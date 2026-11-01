package com.epsilon.prism.hip;

/** 実行系が読み書きする要素の型。 */
public enum HipElement {
  U8(1),
  /** bfloat16（最近接偶数で丸めた値）。 */
  BF16(2),
  FP32(4),
  I32(4);

  private final int bytes;

  HipElement(int bytes) {
    this.bytes = bytes;
  }

  /** 1 要素のバイト数。 */
  public int bytes() {
    return bytes;
  }
}
