package com.epsilon.grp;

/**
 * GrpNetwork の重みのテンソル。weights.f32 と GrpNetwork の重みの配列には、テンソルを宣言の順に並べる。各テンソルの開始位置も、
 * 宣言の順に大きさを足して決める。W は [入力][出力] の行優先で、Dense が出力に足していく順に読める。段階ごとの出力ヘッドを
 * 持つときは、W3・B3 が GrpPhase の順に段階の数だけ並ぶ。
 */
public enum GrpTensor {
  W1(Axis.INPUT, Axis.HIDDEN, false),
  B1(Axis.ONE, Axis.HIDDEN, false),
  W2(Axis.HIDDEN, Axis.HIDDEN, false),
  B2(Axis.ONE, Axis.HIDDEN, false),
  W3(Axis.HIDDEN, Axis.OUTPUT, true),
  B3(Axis.ONE, Axis.OUTPUT, true);

  private enum Axis {
    ONE,
    INPUT,
    HIDDEN,
    OUTPUT;

    private int length(GrpShape shape) {
      return switch (this) {
        case ONE -> 1;
        case INPUT -> shape.width();
        case HIDDEN -> shape.hidden();
        case OUTPUT -> shape.outputs();
      };
    }
  }

  private static final GrpTensor[] VALUES = values();

  private final Axis rows;
  private final Axis columns;
  private final boolean perHead;

  GrpTensor(Axis rows, Axis columns, boolean perHead) {
    this.rows = rows;
    this.columns = columns;
    this.perHead = perHead;
  }

  /** 出力ヘッド 1 つ分の行の数。 */
  public int rows(GrpShape shape) {
    return rows.length(shape);
  }

  /** 出力ヘッド 1 つ分の列の数。 */
  public int columns(GrpShape shape) {
    return columns.length(shape);
  }

  /** 要素の数を返す。段階ごとの出力ヘッドを持つなら、全ての段階の分を足す。 */
  public int size(GrpShape shape) {
    return rows(shape) * columns(shape) * (perHead ? shape.heads() : 1);
  }

  /** 隠れ層の重み（出力の幅が隠れ層の幅である W）なら true。学習では、この重みを He の初期値で初期化し、重み減衰を掛ける。 */
  public boolean hidden() {
    return rows != Axis.ONE && columns == Axis.HIDDEN;
  }

  /** テンソルごとの、重みの配列の中での開始位置を返す（宣言の順）。最後の要素は全体の要素の数。 */
  public static int[] starts(GrpShape shape) {
    int[] starts = new int[VALUES.length + 1];
    for (GrpTensor tensor : VALUES) {
      starts[tensor.ordinal() + 1] = starts[tensor.ordinal()] + tensor.size(shape);
    }
    return starts;
  }
}
