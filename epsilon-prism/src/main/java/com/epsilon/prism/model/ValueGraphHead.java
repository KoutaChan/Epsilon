package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;

/**
 * 価値グラフのヘッド（HIP の K7 に当たる）。ゲートの値 Δ を求め、手ごとの s を組み立てる。
 *
 * <pre>
 * u^G_a  = fp32(W_g·[N(e_a); N(e_base); h] + c) + β_k + U_k·x^G_a     （U_k は [G, 厳密なフィールドの列]、FP32 の要素積）
 * Δ_k(a) = Σ_g w_k,g·silu(u^G_a,g) + b_k + u_k·x^G_a
 * e_a    = Σ_t sg(softmax_t(B(t)/τ))·z_t（t は手の遷移、τ = 0.01、FP32。遷移が 1 つなら z_t）
 *
 * 手番:       s[打牌 x] = B(t_x)、s[立直 x] = A(s[ダマ x]) + Δ_RIICHI(e_立直x, e_ダマx)
 *             M = A(max_{打牌∪立直} s)、s[暗槓・加槓・九種・ツモ] = M + Δ_k(e_a, ē)、ē = Σ_代表の打牌 sg(softmax(B/τ))·e
 * 応答・槍槓: s[PASS] = 0、s[a] = Δ_k(e_a, e_PASS)
 * 鳴いた後:   s[打牌 x] = B(t_x)
 * A(v): 学習では v の元の手（ダマ x・最大の手）が C にあれば v、なければ sg(v)。推論では v（値は同じ）。
 * </pre>
 *
 * <p>ゲートの種類 k は、{@link InputFormat.Role#gate()} が真になる扱いを、扱いの表の順に並べたもの。w_k・b_k・u_k・U_k・β_k は 0 で
 * 始まるので、初期化したままのネットワークでは s ≡ 0 になる。集約の重みには勾配を流さず、中心化もしない。e_base と ē（基準の手の表現）
 * には Δ の勾配が流れる。
 */
final class ValueGraphHead extends Module {

  /** 集約の温度。 */
  static final float TAU = 0.01f;

  /** M の候補にならない手に置く値。最大を取るときに選ばれないようにする。 */
  private static final float FLOOR = -1e30f;

  private final double epsilon;
  private final Parameter norm;
  private final Parameter hidden;
  private final Parameter hiddenBias;
  private final Parameter kindOffset;
  private final Parameter exactProjection;
  private final Parameter output;
  private final Parameter outputBias;
  private final Parameter exactOutput;

  /**
   * パラメーターを登録する。
   *
   * @param shape ネットワークの形
   * @param format 入力の形式（ゲートの種類の数と、厳密なフィールドの列の数を決める）
   */
  ValueGraphHead(PrismShape shape, InputFormat format) {
    int gateKinds = gates(format).length;
    int exactWidth = format.width(Tensor.ACTION_EXACT);
    int width = shape.hidden();
    int gate = shape.gate();
    epsilon = shape.epsilon();
    norm = gain("norm", width);
    hidden = weight("hidden", new Shape(gate, 3L * width));
    hiddenBias = bias("hiddenBias", gate);
    kindOffset = zeros("kindOffset", new Shape(gateKinds, gate));
    exactProjection = zeros("exactProjection", new Shape(gateKinds, gate, exactWidth));
    output = zeros("output", new Shape(gateKinds, gate));
    outputBias = zeros("outputBias", new Shape(gateKinds));
    exactOutput = zeros("exactOutput", new Shape(gateKinds, exactWidth));
  }

  /** ゲートの値を出す手の扱いを、扱いの表の順に返す。 */
  private static InputFormat.Role[] gates(InputFormat format) {
    return format.roles().stream().filter(InputFormat.Role::gate).toArray(InputFormat.Role[]::new);
  }

  /**
   * ゲートの種類の番号を返す。番号は、ゲートの扱いだけを扱いの表の順に並べたときの位置。
   *
   * @param format 入力の形式
   * @param role ゲートの扱い
   * @return 番号
   * @throws IllegalArgumentException ゲートでない扱いのとき
   */
  static int gateIndex(InputFormat format, InputFormat.Role role) {
    int kind = 0;
    for (InputFormat.Role other : format.roles()) {
      if (other.ordinal() == role.ordinal()) {
        if (!other.gate()) {
          break;
        }
        return kind;
      }
      if (other.gate()) {
        kind++;
      }
    }
    throw new IllegalArgumentException("not a gate: " + role);
  }

  /**
   * 手ごとの s を求める。ゲートのある手は、先に Δ を求めてから組み立てる。
   *
   * @param context 文脈
   * @param index インデックス
   * @param transitions 遷移の表現 z [遷移, H]（FP32）
   * @param values B [遷移]（FP32）
   * @param states h = y_G [行, H]（計算の型）
   * @param actionExact ゲートのヘッドが読む厳密なフィールド [手, 列]（FP32）
   * @return s [行, 手の容量]（FP32）
   */
  NDArray scores(
      Context context,
      BatchIndex index,
      NDArray transitions,
      NDArray values,
      NDArray states,
      NDArray actionExact) {
    NDArray delta = null;
    if (index.gateCount > 0) {
      delta =
          gate(
              context,
              aggregate(transitions, values, index.gateChosen),
              aggregate(transitions, values, index.gateBase),
              NDArrays.gatherRows(states, index.gateRows.array()),
              index.gateKinds.array(),
              NDArrays.gatherRows(actionExact, index.gateActions.array()));
    }
    return compose(context.working(), index, values, delta);
  }

  /**
   * 手の表現 e_a = Σ_t sg(softmax_t(B(t)/τ))·z_t を求める（FP32）。遷移が 1 つの手は重みがちょうど 1 なので、
   * z_t をそのまま読む。詰めて和を取っても、0 の符号のほかはビットまで同じになる。遷移が 2 つ以上の手は、
   * 詰めた組で和を取ってから手の順に並べ直す。
   *
   * @param transitions z [遷移, H]（FP32）
   * @param values B [遷移]（FP32）
   * @param aggregation 集約に使う添字
   * @return e [手, H]（FP32）
   */
  static NDArray aggregate(
      NDArray transitions, NDArray values, BatchIndex.Aggregation aggregation) {
    NDArray direct =
        aggregation.directCount == 0
            ? null
            : NDArrays.gatherRows(transitions, aggregation.directTransitions.array());
    if (aggregation.spreadCount == 0) {
      return direct;
    }
    NDArray chosen = aggregation.spreadTransitions.array();
    long count = aggregation.spreadCount;
    long width = aggregation.width;
    NDArray logits = NDArrays.gatherRows(values, chosen).reshape(count, width).div(TAU);
    NDArray weights =
        NDArrays.maskedSoftmax(logits, aggregation.spreadMask.array(), 1).stopGradient();
    long hiddenWidth = transitions.getShape().get(1);
    NDArray rows = NDArrays.gatherRows(transitions, chosen).reshape(count, width, hiddenWidth);
    NDArray spread = rows.mul(weights.expandDims(2)).sum(new int[] {1});
    if (direct == null) {
      return spread;
    }
    return NDArrays.gatherRows(
        NDArrays.concat(new NDList(direct, spread), 0), aggregation.order.array());
  }

  /**
   * ゲートの値 Δ を求める。
   *
   * @param context 文脈
   * @param chosen e_a [ゲート, H]（FP32）
   * @param base e_base [ゲート, H]（FP32）
   * @param states ゲートのある行の h [ゲート, H]（計算の型）
   * @param kinds ゲートの種類 [ゲート]（INT64）
   * @param exact 厳密なフィールド [ゲート, 列]（FP32）
   * @return Δ [ゲート]（FP32）
   */
  NDArray gate(
      Context context,
      NDArray chosen,
      NDArray base,
      NDArray states,
      NDArray kinds,
      NDArray exact) {
    NDArray input =
        NDArrays.concat(
            new NDList(
                context.norm(chosen, norm, epsilon), context.norm(base, norm, epsilon), states),
            1);
    NDArray units =
        context
            .toFloat(context.linear(input, hidden, hiddenBias))
            .add(context.rows(kindOffset, kinds))
            .add(
                context
                    .rows(exactProjection, kinds)
                    .mul(exact.expandDims(1))
                    .sum(new int[] {2}));
    NDArray delta = Context.silu(units).mul(context.rows(output, kinds)).sum(new int[] {1});
    return delta
        .add(context.rows(outputBias, kinds))
        .add(exact.mul(context.rows(exactOutput, kinds)).sum(new int[] {1}));
  }

  /**
   * 手ごとの s を、クラスの Javadoc に書いた扱いごとの式で組み立てる。A(v) で元の手が C にないときは、勾配を止める。
   *
   * @param working 作業用のマネージャー
   * @param index インデックス
   * @param values B [遷移]（FP32）
   * @param delta Δ [ゲート]（FP32。ゲートが無ければ null）
   * @return s [行, 手の容量]（FP32）
   */
  static NDArray compose(NDManager working, BatchIndex index, NDArray values, NDArray delta) {
    long actions = index.actionSlots.array().size();
    long slots = (long) index.rows * index.actionCapacity;
    NDArray sum = working.zeros(new Shape(actions));
    if (index.discardCount > 0) {
      NDArray discards = NDArrays.gatherRows(values, index.discardTransitions.array());
      sum = sum.add(spread(discards, index.discardActions.array(), actions));
    }
    if (index.riichiCount > 0) {
      NDArray dama = NDArrays.gatherRows(values, index.riichiBaseTransitions.array());
      if (index.riichiAnchors != null) {
        dama = NDArrays.where(index.riichiAnchors.array(), dama, dama.stopGradient());
      }
      NDArray riichi = dama.add(NDArrays.gatherRows(delta, index.riichiGates.array()));
      sum = sum.add(spread(riichi, index.riichiActions.array(), actions));
    }
    if (index.responseCount > 0) {
      NDArray responses = NDArrays.gatherRows(delta, index.responseGates.array());
      sum = sum.add(spread(responses, index.responseActions.array(), actions));
    }
    if (index.turnCount > 0) {
      NDArray dense =
          spread(sum, index.actionSlots.array(), slots).reshape(index.rows, index.actionCapacity);
      NDArray candidates =
          NDArrays.where(index.pivotMask.array(), dense, working.full(dense.getShape(), FLOOR));
      NDArray pivot = candidates.argMax(1).expandDims(1);
      NDArray peak = dense.gather(pivot, 1);
      if (index.measured != null) {
        peak = NDArrays.where(index.measured.array().gather(pivot, 1), peak, peak.stopGradient());
      }
      peak = peak.mul(index.pivotPresent.array()).reshape(index.rows);
      NDArray turn =
          NDArrays.gatherRows(peak, index.turnRows.array())
              .add(NDArrays.gatherRows(delta, index.turnGates.array()));
      sum = sum.add(spread(turn, index.turnActions.array(), actions));
    }
    return spread(sum, index.actionSlots.array(), slots).reshape(index.rows, index.actionCapacity);
  }

  /** 長さ {@code size} の 0 の列を作り、値を決まった位置に置く。 */
  private static NDArray spread(NDArray values, NDArray positions, long size) {
    return NDArrays.scatterRows(values, positions, size);
  }
}
