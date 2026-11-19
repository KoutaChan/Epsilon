package com.epsilon.prism.model;

/** 設定と fit-cohorts で選べるネットワークの形。 */
public enum ShapeArm {
  /** 型ごとの FFN を持たない形（H 256・F 512・G 256、深さは選べる）。 */
  V2(PrismShape.standard()),
  /** 型ごとの FFN を持つ形（局・席の FFN 1536、候補層の 2 段目あり）。 */
  V3(new PrismShape.V3(256, 512, 512, 768, 1536, 1024, true)),
  /** V3 から候補層の 2 段目を外した形。 */
  V3_ONE_STAGE(new PrismShape.V3(256, 512, 512, 768, 1536, 1024, false)),
  /** V3 の局・席の FFN を 512 にした形。 */
  V3_CONTEXT_512(new PrismShape.V3(256, 512, 512, 768, 512, 1024, true)),
  /** 幅 256 の V4。 */
  V4_S(new PrismShape.V4()),
  /** 幅 384 の V4。 */
  V4_M(new PrismShape.V4(384));

  /** V3 で始まるアームの深さ。 */
  public static final int V3_DEPTH = PrismShape.V3.DEPTH;

  private final PrismShape.TokenTrunk fixed;

  /** 設定や当てはめで幹の深さを選べる腕。 */
  public boolean selectableDepth() {
    return this == V2;
  }

  ShapeArm(PrismShape.TokenTrunk fixed) {
    this.fixed = fixed;
  }

  /** 深さを指定せずに選んだ形。V2 は既定の深さを使う。 */
  public PrismShape shape() {
    return fixed;
  }

  /** トークンの幹の深さを指定した形。 */
  public PrismShape.TokenTrunk shape(int depth) {
    if (selectableDepth()) {
      return PrismShape.standard(depth);
    }
    if (depth != fixed.depth()) {
      throw new IllegalArgumentException(this + " needs depth " + fixed.depth() + ", got " + depth);
    }
    return fixed;
  }
}
