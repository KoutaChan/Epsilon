package com.epsilon.prism.kernel;

/** 収集カーネルが読み書きする要素の型。 */
public enum ElementType {
  U8(1),
  /** bfloat16（最近接偶数で丸めた値）。 */
  BF16(2),
  FP32(4),
  I32(4);

  private final int bytes;

  ElementType(int bytes) {
    this.bytes = bytes;
  }

  /** 1 要素のバイト数。 */
  public int bytes() {
    return bytes;
  }
}
