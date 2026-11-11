package com.epsilon.prism.model;

/**
 * 形のアーム。型ごとの FFN を持たない形（V2）と、型ごとの FFN を持つ形（V3 で始まるアーム）を比べるために、形を名前で選べるようにする。
 * 名前は定数の綴りのまま使う（設定の network.shape でも fit-cohorts の --shape でも V2・V3・V3_ONE_STAGE・V3_CONTEXT_512）。V3 で始まる
 * アームは全て H 256・F 512（層 0）・G 512・深さ 3・牌の FFN 768・候補層の FFN 1024。
 */
public enum ShapeArm {
  /** 型ごとの FFN を持たない形（H 256・F 512・G 256、深さは選べる）。 */
  V2(null),
  /** 型ごとの FFN を持つ形（局・席の FFN 1536、候補層の 2 段目あり）。 */
  V3(new PrismShape.Typed(768, 1536, 1024, true)),
  /** V3 から候補層の 2 段目を外した形。 */
  V3_ONE_STAGE(new PrismShape.Typed(768, 1536, 1024, false)),
  /** V3 の局・席の FFN を 512 にした形。 */
  V3_CONTEXT_512(new PrismShape.Typed(768, 512, 1024, true));

  /** V3 で始まるアームの深さ。 */
  public static final int V3_DEPTH = 3;

  private final PrismShape.Typed typed;

  ShapeArm(PrismShape.Typed typed) {
    this.typed = typed;
  }

  /**
   * アームの形を返す。
   *
   * @param depth バックボーンの層の数（V3 で始まるアームは 3 だけ）
   * @return 形
   * @throws IllegalArgumentException V3 で始まるアームに 3 以外の深さを渡したとき
   */
  public PrismShape shape(int depth) {
    if (typed == null) {
      return PrismShape.standard(depth);
    }
    if (depth != V3_DEPTH) {
      throw new IllegalArgumentException(this + " needs depth " + V3_DEPTH + ", got " + depth);
    }
    return new PrismShape(256, 512, 512, V3_DEPTH, typed);
  }
}
