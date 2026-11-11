package com.epsilon.grp;

/**
 * OutcomeNetwork の重みのテンソル。重みの配列には、テンソルを宣言の順に並べる。各テンソルの開始位置も、宣言の順に大きさを足して
 * 決める。W は [入力][出力] の行優先で、Dense が出力に足していく順に読める。
 */
public enum OutcomeTensor {
  W1(Axis.INPUT, Axis.HIDDEN),
  B1(Axis.ONE, Axis.HIDDEN),
  W2(Axis.HIDDEN, Axis.HIDDEN),
  B2(Axis.ONE, Axis.HIDDEN),
  W3(Axis.HIDDEN, Axis.LOGITS),
  B3(Axis.ONE, Axis.LOGITS);

  private enum Axis {
    ONE,
    INPUT,
    HIDDEN,
    LOGITS;

    private int length(OutcomeShape shape) {
      return switch (this) {
        case ONE -> 1;
        case INPUT -> shape.width();
        case HIDDEN -> shape.hidden();
        case LOGITS -> shape.logits();
      };
    }
  }

  private static final OutcomeTensor[] VALUES = values();

  private final Axis rows;
  private final Axis columns;

  OutcomeTensor(Axis rows, Axis columns) {
    this.rows = rows;
    this.columns = columns;
  }

  public int rows(OutcomeShape shape) {
    return rows.length(shape);
  }

  public int columns(OutcomeShape shape) {
    return columns.length(shape);
  }

  public int size(OutcomeShape shape) {
    return rows(shape) * columns(shape);
  }

  /** 隠れ層の重み（出力の幅が隠れ層の幅である W）なら true。学習では、この重みを He の初期値で初期化し、重み減衰を掛ける。 */
  public boolean hidden() {
    return rows != Axis.ONE && columns == Axis.HIDDEN;
  }

  /** テンソルごとの、重みの配列の中での開始位置を返す（宣言の順）。最後の要素は全体の要素の数。 */
  public static int[] starts(OutcomeShape shape) {
    int[] starts = new int[VALUES.length + 1];
    for (OutcomeTensor tensor : VALUES) {
      starts[tensor.ordinal() + 1] = starts[tensor.ordinal()] + tensor.size(shape);
    }
    return starts;
  }
}
