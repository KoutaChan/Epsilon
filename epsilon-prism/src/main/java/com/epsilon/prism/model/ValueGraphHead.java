package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputSchema.Tensor;
import com.epsilon.prism.model.WidthPlan.Axis;
import com.epsilon.prism.model.WidthPlan.Role;
import java.util.Arrays;

/**
 * 価値グラフの頭（HIP の K7 に当たる）。門の値 Δ と、手ごとの s の合成を持つ。
 *
 * <pre>
 * u^G_a  = fp32(W_g·[N(e_a); N(e_base); h] + c) + β_k + U_k·x^G_a     （U_k は [G, 10]、FP32 の要素積）
 * Δ_k(a) = Σ_g w_k,g·silu(u^G_a,g) + b_k + u_k·x^G_a
 * e_a    = Σ_t sg(softmax_t(B(t)/τ))·z_t（手の遷移の上、τ = 0.01、FP32。遷移が 1 つなら z_t）
 *
 * 手番:       s[打牌 x] = B(t_x)、s[立直 x] = A(s[ダマ x]) + Δ_RIICHI(e_立直x, e_ダマx)
 *             M = A(max_{打牌∪立直} s)、s[暗槓・加槓・九種・ツモ] = M + Δ_k(e_a, ē)、ē = Σ_代表の打牌 sg(softmax(B/τ))·e
 * 応答・槍槓: s[PASS] = 0、s[a] = Δ_k(e_a, e_PASS)
 * 鳴いた後:   s[打牌 x] = B(t_x)
 * A(v): 学習では v の元の手（ダマ x・最大の手）が C にあれば v、なければ sg(v)。推論では v（値は同じ）。
 * </pre>
 *
 * <p>門の種類 k は {@link ActionRole#gate()} が真の役割を宣言の順に並べたもの。w_k・b_k・u_k・U_k・β_k は 0 で始めるので、白紙では
 * s ≡ 0。集約の重みには勾配を流さず、中心化も使わない。e_base・ē（基準の手の遷移の z）には Δ の勾配が流れる。
 */
final class ValueGraphHead extends Module {

  /** 集約の温度。 */
  static final float TAU = 0.01f;

  /** 門の役割（宣言の順）。 */
  private static final ActionRole[] GATES =
      Arrays.stream(ActionRole.values()).filter(ActionRole::gate).toArray(ActionRole[]::new);

  static final int GATE_KINDS = GATES.length;

  private static final int EXACT = Tensor.ACTION_EXACT.width();

  /** M を取る手の印の外の値。 */
  private static final float FLOOR = -1e30f;

  /** 手の役割ごとの s の作り方。 */
  enum Composition {
    /** s = B(t)。 */
    DISCARD_VALUE,
    /** s = A(s[ダマ]) + Δ_RIICHI（基準はダマの遷移）。 */
    RIICHI_GATE,
    /** s = Δ（基準は PASS の遷移、PASS ≡ 0）。 */
    RESPONSE_GATE,
    /** s = A(M) + Δ（基準は ē）。 */
    TURN_GATE,
    /** s = 0。 */
    ZERO;

    /**
     * 役割から作り方を返す。default を書かないので、役割を足すと組み立てで落ちる。
     *
     * @param role 役割
     * @return 作り方
     */
    static Composition of(ActionRole role) {
      return switch (role) {
        case DISCARD -> DISCARD_VALUE;
        case RIICHI -> RIICHI_GATE;
        case CALL, RON -> RESPONSE_GATE;
        case KAN, TSUMO, KYUSHU -> TURN_GATE;
        case PASS -> ZERO;
      };
    }
  }

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
   * @param shape 網の形
   */
  ValueGraphHead(PrismShape shape) {
    int width = shape.hidden();
    int gate = shape.gate();
    epsilon = shape.epsilon();
    norm = gain("norm", width);
    hidden =
        weight(
            "hidden",
            new Shape(gate, 3L * width),
            Axis.of(Role.INNER_OUT),
            Axis.of(Role.STREAM_IN, 3));
    hiddenBias = bias("hiddenBias", gate, Axis.of(Role.INNER_OUT));
    kindOffset =
        zeros("kindOffset", new Shape(GATE_KINDS, gate), Axis.of(Role.FIXED), Axis.of(Role.INNER_OUT));
    exactProjection =
        zeros(
            "exactProjection",
            new Shape(GATE_KINDS, gate, EXACT),
            Axis.of(Role.FIXED),
            Axis.of(Role.INNER_OUT),
            Axis.of(Role.FIXED));
    output = zeros("output", new Shape(GATE_KINDS, gate), Axis.of(Role.FIXED), Axis.of(Role.INNER_IN));
    outputBias = zeros("outputBias", new Shape(GATE_KINDS), Axis.of(Role.FIXED));
    exactOutput =
        zeros("exactOutput", new Shape(GATE_KINDS, EXACT), Axis.of(Role.FIXED), Axis.of(Role.FIXED));
  }

  /**
   * 門の種類の番号を返す。
   *
   * @param role 門の役割
   * @return 番号
   * @throws IllegalArgumentException 門でない役割のとき
   */
  static int gateIndex(ActionRole role) {
    for (int k = 0; k < GATES.length; k++) {
      if (GATES[k] == role) {
        return k;
      }
    }
    throw new IllegalArgumentException("not a gate: " + role);
  }

  /**
   * s を作る。
   *
   * @param context 文脈
   * @param index 索引
   * @param transitions 遷移の表現 z [遷移, H]（FP32）
   * @param values B [遷移]（FP32）
   * @param states h = y_G [行, H]（計算の型）
   * @param actionExact 門の道 [手, 10]（FP32）
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
      NDArray chosen =
          aggregate(
              transitions,
              values,
              index.gateTransitions,
              index.gateTransitionMask,
              index.gateCount,
              index.gateWidth);
      NDArray base = NDArrays.gatherRows(transitions, index.gateBaseTransitions);
      if (index.barWidth > 0) {
        NDArray bar =
            aggregate(
                transitions, values, index.barTransitions, index.barMask, index.rows, index.barWidth);
        NDArray usesBar = index.gateUsesBar.broadcast(base.getShape());
        base = NDArrays.where(usesBar, NDArrays.gatherRows(bar, index.gateRows), base);
      }
      delta =
          gate(
              context,
              chosen,
              base,
              NDArrays.gatherRows(states, index.gateRows),
              index.gateKinds,
              NDArrays.gatherRows(actionExact, index.gateActions));
    }
    return compose(context.working(), index, values, delta);
  }

  /**
   * 手の表現 e_a = Σ_t sg(softmax_t(B(t)/τ))·z_t（FP32）。
   *
   * @param transitions z [遷移, H]（FP32）
   * @param values B [遷移]（FP32）
   * @param chosen 手ごとの遷移 [手 × 幅]（INT64、詰め物は 0）
   * @param mask 遷移の在否 [手, 幅]
   * @param count 手の数
   * @param width 幅
   * @return e [手, H]（FP32）
   */
  static NDArray aggregate(
      NDArray transitions, NDArray values, NDArray chosen, NDArray mask, long count, long width) {
    NDArray logits = NDArrays.gatherRows(values, chosen).reshape(count, width).div(TAU);
    NDArray weights = NDArrays.maskedSoftmax(logits, mask, 1).stopGradient();
    long hiddenWidth = transitions.getShape().get(1);
    NDArray rows = NDArrays.gatherRows(transitions, chosen).reshape(count, width, hiddenWidth);
    return rows.mul(weights.expandDims(2)).sum(new int[] {1});
  }

  /**
   * 門の値 Δ。
   *
   * @param context 文脈
   * @param chosen e_a [門, H]（FP32）
   * @param base e_base [門, H]（FP32）
   * @param states 門の行の h [門, H]（計算の型）
   * @param kinds 門の種類 [門]（INT64）
   * @param exact 門の道 [門, 10]（FP32）
   * @return Δ [門]（FP32）
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
   * 手ごとの s を合成する（{@link Composition}、錨の勾配の扱いを含む）。
   *
   * @param working 作業の管理元
   * @param index 索引
   * @param values B [遷移]（FP32）
   * @param delta Δ [門]（FP32。門がなければ null）
   * @return s [行, 手の容量]（FP32）
   */
  static NDArray compose(NDManager working, BatchIndex index, NDArray values, NDArray delta) {
    long actions = index.actionSlots.size();
    long slots = (long) index.rows * index.actionCapacity;
    NDArray sum = working.zeros(new Shape(actions));
    if (index.discardCount > 0) {
      sum = sum.add(spread(NDArrays.gatherRows(values, index.discardTransitions), index.discardActions, actions));
    }
    if (index.riichiCount > 0) {
      NDArray dama = NDArrays.gatherRows(values, index.riichiBaseTransitions);
      if (index.riichiAnchors != null) {
        dama = NDArrays.where(index.riichiAnchors, dama, dama.stopGradient());
      }
      NDArray riichi = dama.add(NDArrays.gatherRows(delta, index.riichiGates));
      sum = sum.add(spread(riichi, index.riichiActions, actions));
    }
    if (index.responseCount > 0) {
      sum = sum.add(spread(NDArrays.gatherRows(delta, index.responseGates), index.responseActions, actions));
    }
    if (index.turnCount > 0) {
      NDArray dense = spread(sum, index.actionSlots, slots).reshape(index.rows, index.actionCapacity);
      NDArray candidates =
          NDArrays.where(index.pivotMask, dense, working.full(dense.getShape(), FLOOR));
      NDArray pivot = candidates.argMax(1).expandDims(1);
      NDArray peak = dense.gather(pivot, 1);
      if (index.measured != null) {
        peak = NDArrays.where(index.measured.gather(pivot, 1), peak, peak.stopGradient());
      }
      peak = peak.mul(index.pivotPresent).reshape(index.rows);
      NDArray turn =
          NDArrays.gatherRows(peak, index.turnRows).add(NDArrays.gatherRows(delta, index.turnGates));
      sum = sum.add(spread(turn, index.turnActions, actions));
    }
    return spread(sum, index.actionSlots, slots).reshape(index.rows, index.actionCapacity);
  }

  /** 値を 0 の列の決まった位置へ散らす。 */
  private static NDArray spread(NDArray values, NDArray positions, long size) {
    return NDArrays.scatterRows(values, positions, size);
  }
}
