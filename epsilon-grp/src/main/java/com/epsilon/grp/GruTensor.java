package com.epsilon.grp;

/**
 * GruSinkhorn の重みのテンソル。weights.f32 と GruSinkhorn の重みの配列には、テンソルを宣言の順に並べる。層ごとのテンソルは、
 * 層 0 から順に層の数だけ並ぶ。W は [入力][出力] の行優先で、Dense が出力に足していく順に読める。ゲートの列は r・z・n の順に
 * 状態の幅ずつ並び、PyTorch の GRU の重み [3 × 状態の幅][入力] を転置した形になる。ロジットの列は [絶対席 × S ＋ 順位] の順。
 */
public enum GruTensor {
  /** 層の入力からゲートへの重み。層 0 の入力は RoundField の行、ほかの層の入力は 1 つ下の層の新しい状態。 */
  INPUT_WEIGHTS(Axis.LAYER_INPUT, Axis.GATES, true),
  /** 層の前の状態からゲートへの重み。 */
  STATE_WEIGHTS(Axis.HIDDEN, Axis.GATES, true),
  INPUT_BIAS(Axis.ONE, Axis.GATES, true),
  STATE_BIAS(Axis.ONE, Axis.GATES, true),
  /** 最上層の最後の状態から射影の層への重み（ReLU の前）。 */
  PROJECTION_WEIGHTS(Axis.HIDDEN, Axis.HIDDEN, false),
  PROJECTION_BIAS(Axis.ONE, Axis.HIDDEN, false),
  /** 射影の層から、周辺確率のロジットへの重み。 */
  LOGIT_WEIGHTS(Axis.HIDDEN, Axis.MARGINALS, false),
  LOGIT_BIAS(Axis.ONE, Axis.MARGINALS, false);

  private enum Axis {
    ONE,
    LAYER_INPUT,
    HIDDEN,
    GATES,
    MARGINALS;

    private int length(GruShape shape, int layer) {
      return switch (this) {
        case ONE -> 1;
        case LAYER_INPUT -> layer == 0 ? shape.inputs() : shape.hidden();
        case HIDDEN -> shape.hidden();
        case GATES -> shape.gates();
        case MARGINALS -> shape.marginals();
      };
    }
  }

  private static final GruTensor[] VALUES = values();

  private final Axis rows;
  private final Axis columns;
  private final boolean perLayer;

  GruTensor(Axis rows, Axis columns, boolean perLayer) {
    this.rows = rows;
    this.columns = columns;
    this.perLayer = perLayer;
  }

  /** 指定した層でのテンソルの行の数を返す。層ごとでないテンソルでは層を見ない。 */
  public int rows(GruShape shape, int layer) {
    return rows.length(shape, layer);
  }

  public int columns(GruShape shape) {
    return columns.length(shape, 0);
  }

  /** このテンソルがいくつあるかを返す。層ごとのテンソルなら層の数、ほかは 1。 */
  public int layers(GruShape shape) {
    return perLayer ? shape.layers() : 1;
  }

  /**
   * テンソルと層ごとの、重みの配列の中での開始位置を返す。添字は [テンソル（宣言の順）][層] で、層ごとでないテンソルは層 0 だけを
   * 持つ。最後の {@code [values().length][0]} は重みの数。
   */
  public static int[][] starts(GruShape shape) {
    int[][] starts = new int[VALUES.length + 1][];
    int next = 0;
    for (GruTensor tensor : VALUES) {
      starts[tensor.ordinal()] = new int[tensor.layers(shape)];
      for (int layer = 0; layer < starts[tensor.ordinal()].length; layer++) {
        starts[tensor.ordinal()][layer] = next;
        next += tensor.rows(shape, layer) * tensor.columns(shape);
      }
    }
    starts[VALUES.length] = new int[] {next};
    return starts;
  }

  /** 重みの数を返す（{@link #starts} の最後の値）。 */
  public static int count(GruShape shape) {
    return starts(shape)[VALUES.length][0];
  }
}
