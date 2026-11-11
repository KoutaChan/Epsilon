package com.epsilon.grp;

import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;

/**
 * GRU と Sinkhorn の出力ヘッドを使う GRP（推論だけ）。入力は、各局が始まったときの卓を局の順に並べたもの。RoundHistory の行
 * （終わった局まで）の後に、局末の卓を 1 行足す。1 ステップの入力は、RoundField の行の値を列ごとのスケールで割ったもの。
 *
 * <p>GRU は各層の状態を 0 で始める。各ステップでは層 0 から順に、層の入力 x と前の状態 h から次の状態を求める（PyTorch の
 * GRU と同じ式）。r = σ(W_ir x ＋ b_ir ＋ W_hr h ＋ b_hr)、z = σ(W_iz x ＋ b_iz ＋ W_hz h ＋ b_hz)、
 * n = tanh(W_in x ＋ b_in ＋ r ⊙ (W_hn h ＋ b_hn))、h' = (1 − z) ⊙ n ＋ z ⊙ h。上の層の入力は、下の層の新しい状態。
 * 最後のステップで最上層が持つ状態を射影して ReLU に通し、さらに S² 個のロジット（[絶対席 × S ＋ 順位]）にする。Sinkhorn の出力
 * ヘッドは、それを行の和も列の和も 1 の周辺確率にする。
 *
 * <p>GRU と射影は float、ロジットから先は double で求め、指数と tanh は StrictMath で求める。重みは書き換えない。作業領域は
 * このモデルで作ったものを使い、スレッドごとに持つ。値は履歴と卓だけで決まり、ビット単位で再現する。
 */
public final class GruSinkhorn implements GrpModel {

  /** 行と列の正規化を交互に繰り返す回数。 */
  static final int ITERATIONS = 32;

  /** 交互作用を c·tanh(a / c) でクリップするときの c。 */
  static final double INTERACTION_CLIP = 2;

  private final GruShape shape;
  private final float[] weights;

  /** テンソルと層ごとの、重みの配列の中での開始位置。添字は [テンソル（宣言の順）][層]。 */
  private final int[][] starts;

  /** 入力の列ごとのスケール（RoundField）。 */
  private final float[] scales;

  private final int seats;
  private final int inputs;
  private final int hidden;
  private final int gates;
  private final int layers;

  /** 周辺確率の数 S²。 */
  private final int outputs;

  private GruSinkhorn(GruShape shape, float[] weights) {
    this.shape = shape;
    this.weights = weights;
    starts = GruTensor.starts(shape);
    scales = RoundField.scales(shape.variant());
    seats = shape.variant().seats();
    inputs = shape.inputs();
    hidden = shape.hidden();
    gates = shape.gates();
    layers = shape.layers();
    outputs = shape.marginals();
  }

  /**
   * 重みのコピーを持つモデルを作る。重みの数が形に合わないか、有限でない値があれば IllegalArgumentException を投げる。
   *
   * @param weights GruTensor の宣言の順に並べた重み
   */
  public static GruSinkhorn of(GruShape shape, float[] weights) {
    int expected = GruTensor.count(shape);
    if (weights.length != expected) {
      throw new IllegalArgumentException(shape + " needs " + expected + " weights, got " + weights.length);
    }
    for (int index = 0; index < weights.length; index++) {
      if (!Float.isFinite(weights[index])) {
        throw new IllegalArgumentException("Weight " + index + " is not finite: " + weights[index]);
      }
    }
    return new GruSinkhorn(shape, weights.clone());
  }

  public GruShape shape() {
    return shape;
  }

  @Override
  public Variant variant() {
    return shape.variant();
  }

  /** 重みのコピーを返す。並びは GruTensor の宣言の順。 */
  public float[] weights() {
    return weights.clone();
  }

  /**
   * 半荘が続く卓の、席ごとの順位の周辺確率を書く。
   *
   * @param history 終わった局までの、各局が始まったときの卓（1 局以上）
   * @param out 書き先（長さ S²。並びは [絶対席 × S ＋ 順位]）
   */
  public void marginals(RoundHistory history, TableState table, GruWorkspace workspace, double[] out) {
    requireOutput(out, outputs);
    forward(history, table, workspace);
    System.arraycopy(workspace.marginals, 0, out, 0, outputs);
  }

  /** {@link #marginals(RoundHistory, TableState, GruWorkspace, double[])} と同じことを GrpState で行う。 */
  public void marginals(RoundHistory history, GrpState state, GruWorkspace workspace, double[] out) {
    requireOutput(out, outputs);
    forward(history, state, workspace);
    System.arraycopy(workspace.marginals, 0, out, 0, outputs);
  }

  /**
   * 半荘が続く卓の、絶対席ごとの Φ_i = Σ_r P[i][r]·u_r を書く。P は周辺確率。
   *
   * @param out 書き先（長さ S）
   */
  public void phi(RoundHistory history, TableState table, RankUtility utility, GruWorkspace workspace, double[] out) {
    requireOutput(out, seats);
    forward(history, table, workspace);
    expectedUtilities(variant(), workspace.marginals, utility, out);
  }

  /** {@link #phi(RoundHistory, TableState, RankUtility, GruWorkspace, double[])} と同じことを GrpState で行う。 */
  public void phi(RoundHistory history, GrpState state, RankUtility utility, GruWorkspace workspace, double[] out) {
    requireOutput(out, seats);
    forward(history, state, workspace);
    expectedUtilities(variant(), workspace.marginals, utility, out);
  }

  /**
   * 周辺確率 P から Φ_i = Σ_r P[i][r]·u_r を求めて書く。r の小さい順に足す。
   *
   * @param marginals 周辺確率（並びは [絶対席 × S ＋ 順位]）
   * @param out 書き先（長さ S）
   */
  static void expectedUtilities(Variant variant, double[] marginals, RankUtility utility, double[] out) {
    int seats = variant.seats();
    for (int seat = 0; seat < seats; seat++) {
      double value = 0;
      for (int rank = 0; rank < seats; rank++) {
        value += marginals[seat * seats + rank] * utility.value(variant, rank);
      }
      out[seat] = value;
    }
  }

  /** 履歴が null のとき、空のとき、ルールが違うときは IllegalArgumentException を投げる。 */
  RoundHistory requireHistory(RoundHistory history) {
    if (history == null) {
      throw new IllegalArgumentException("GruSinkhorn reads the round history: " + this);
    }
    if (history.variant() != variant()) {
      throw new IllegalArgumentException("Variant mismatch: " + this + ", " + history);
    }
    if (history.size() == 0) {
      throw new IllegalArgumentException("The round history is empty: " + history);
    }
    return history;
  }

  /** 周辺確率を求めて作業領域に書く。 */
  void forward(RoundHistory history, TableState table, GruWorkspace workspace) {
    if (table.variant() != variant()) {
      throw new IllegalArgumentException("Variant mismatch: " + this + ", " + table);
    }
    if (table.gameOver()) {
      throw new IllegalArgumentException("A finished game has no next hand: " + table);
    }
    RoundField.write(table, check(workspace).row, 0);
    forward(requireHistory(history), workspace);
  }

  /** 周辺確率を求めて作業領域に書く。 */
  void forward(RoundHistory history, GrpState state, GruWorkspace workspace) {
    if (state.variant() != variant()) {
      throw new IllegalArgumentException("Variant mismatch: " + this + ", " + state);
    }
    RoundField.write(state, check(workspace).row, 0);
    forward(requireHistory(history), workspace);
  }

  /**
   * 履歴を読んでから局末の卓の行で 1 ステップ進め、射影の層、ロジットの層、Sinkhorn の順に通す。同じ履歴を読んだ後の状態が
   * 作業領域に残っていれば、履歴は読み直さずにそれを使う。
   */
  private void forward(RoundHistory history, GruWorkspace workspace) {
    long stamp = history.stamp();
    if (!workspace.restore(stamp)) {
      for (float[] layer : workspace.state) {
        Arrays.fill(layer, 0f);
      }
      for (int index = 0; index < history.size(); index++) {
        for (int column = 0; column < inputs; column++) {
          workspace.input[column] = history.value(index, column) / scales[column];
        }
        step(workspace);
      }
      workspace.save(stamp);
    }
    for (int column = 0; column < inputs; column++) {
      workspace.input[column] = workspace.row[column] / scales[column];
    }
    step(workspace);
    Dense.relu(
        weights,
        start(GruTensor.PROJECTION_WEIGHTS, 0),
        start(GruTensor.PROJECTION_BIAS, 0),
        workspace.state[layers - 1],
        hidden,
        workspace.projected,
        hidden);
    Dense.logits(
        weights,
        start(GruTensor.LOGIT_WEIGHTS, 0),
        start(GruTensor.LOGIT_BIAS, 0),
        workspace.projected,
        hidden,
        workspace.logits,
        outputs);
    sinkhorn(workspace);
  }

  /** 作業領域の入力で 1 ステップ進め、各層の状態をその場で更新する。 */
  private void step(GruWorkspace workspace) {
    float[] x = workspace.input;
    int width = inputs;
    float[] fromInput = workspace.inputGates;
    float[] fromState = workspace.stateGates;
    for (int layer = 0; layer < layers; layer++) {
      float[] h = workspace.state[layer];
      Dense.affine(
          weights, start(GruTensor.INPUT_WEIGHTS, layer), start(GruTensor.INPUT_BIAS, layer), x, width, fromInput,
          gates);
      Dense.affine(
          weights, start(GruTensor.STATE_WEIGHTS, layer), start(GruTensor.STATE_BIAS, layer), h, hidden, fromState,
          gates);
      // h' の各ユニットは自分の前の値しか読まず、ゲートは前の状態から計算済みなので、その場で書き換えてよい。
      for (int unit = 0; unit < hidden; unit++) {
        float reset = sigmoid(fromInput[unit] + fromState[unit]);
        float update = sigmoid(fromInput[hidden + unit] + fromState[hidden + unit]);
        float candidate =
            (float) StrictMath.tanh(fromInput[2 * hidden + unit] + reset * fromState[2 * hidden + unit]);
        h[unit] = (1f - update) * candidate + update * h[unit];
      }
      x = h;
      width = hidden;
    }
  }

  /**
   * ロジット z を周辺確率にする。交互作用 a = z − 行の平均 − 列の平均 ＋ 全体の平均 を c·tanh(a / c) でクリップし、
   * exp(a) の行と列を交互に {@code ITERATIONS} 回ずつ正規化する。最後に行を正規化してから、列ごとに不足分 (1 − 列の和) / S を
   * その列の全ての席に足す。これで行の和を保ったまま、列の和が 1 になる。|a| < c なので exp(a) は (e^−c, e^c) に入り、対数を
   * 取らずに割ってもアンダーフローしない（対数で正規化するのと同じ写像になる）。補正の後に、正の有限な値でない
   * 確率が 1 つでも残れば IllegalStateException を投げる。
   */
  private void sinkhorn(GruWorkspace workspace) {
    double[] z = workspace.logits;
    double[] p = workspace.marginals;
    double[] rowMeans = workspace.rowMeans;
    double[] columnMeans = workspace.columnMeans;
    Arrays.fill(rowMeans, 0);
    Arrays.fill(columnMeans, 0);
    double mean = 0;
    for (int seat = 0; seat < seats; seat++) {
      for (int rank = 0; rank < seats; rank++) {
        double value = z[seat * seats + rank];
        rowMeans[seat] += value;
        columnMeans[rank] += value;
        mean += value;
      }
    }
    for (int index = 0; index < seats; index++) {
      rowMeans[index] /= seats;
      columnMeans[index] /= seats;
    }
    mean /= outputs;
    for (int seat = 0; seat < seats; seat++) {
      for (int rank = 0; rank < seats; rank++) {
        double interaction = z[seat * seats + rank] - rowMeans[seat] - columnMeans[rank] + mean;
        p[seat * seats + rank] =
            StrictMath.exp(INTERACTION_CLIP * StrictMath.tanh(interaction / INTERACTION_CLIP));
      }
    }
    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
      normalizeRows(p);
      normalizeColumns(p);
    }
    normalizeRows(p);
    for (int rank = 0; rank < seats; rank++) {
      double sum = 0;
      for (int seat = 0; seat < seats; seat++) {
        sum += p[seat * seats + rank];
      }
      double deficit = (1 - sum) / seats;
      for (int seat = 0; seat < seats; seat++) {
        p[seat * seats + rank] += deficit;
      }
    }
    for (double value : p) {
      if (!(value > 0) || !Double.isFinite(value)) {
        throw new IllegalStateException("The column repair left a marginal of " + value + ": " + this);
      }
    }
  }

  private void normalizeRows(double[] p) {
    for (int seat = 0; seat < seats; seat++) {
      double sum = 0;
      for (int rank = 0; rank < seats; rank++) {
        sum += p[seat * seats + rank];
      }
      for (int rank = 0; rank < seats; rank++) {
        p[seat * seats + rank] /= sum;
      }
    }
  }

  private void normalizeColumns(double[] p) {
    for (int rank = 0; rank < seats; rank++) {
      double sum = 0;
      for (int seat = 0; seat < seats; seat++) {
        sum += p[seat * seats + rank];
      }
      for (int seat = 0; seat < seats; seat++) {
        p[seat * seats + rank] /= sum;
      }
    }
  }

  private int start(GruTensor tensor, int layer) {
    return starts[tensor.ordinal()][layer];
  }

  private GruWorkspace check(GruWorkspace workspace) {
    if (workspace.model != this) {
      throw new IllegalArgumentException("A workspace of another model used with " + this);
    }
    return workspace;
  }

  private static float sigmoid(float x) {
    return (float) (1 / (1 + StrictMath.exp(-x)));
  }

  private static void requireOutput(double[] out, int length) {
    if (out.length != length) {
      throw new IllegalArgumentException("Output needs " + length + " values, got " + out.length);
    }
  }

  @Override
  public String toString() {
    return "GruSinkhorn[" + shape + "]";
  }
}
