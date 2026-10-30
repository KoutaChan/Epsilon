package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.input.FrontierField;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.input.InputSchema.DiscardExact;
import com.epsilon.prism.input.InputSchema.KeyBlock;
import com.epsilon.prism.input.InputSchema.Tensor;
import com.epsilon.prism.input.InputSchema.TransitionField;
import com.epsilon.prism.input.InputSchema.TransitionScalar;
import com.epsilon.prism.model.WidthPlan.Axis;
import com.epsilon.prism.model.WidthPlan.Role;

/**
 * 候補層（HIP の K6 に当たる）。遷移ごとに 1 段で、候補どうしは見ない。
 *
 * <pre>
 * t0 = W_X·pack_t + b_X + W_E·y_T[τ(t)]          τ(t) は捨てる牌（鳴いた後の打牌）、なければ主牌、なければ項なし
 * q  = W_Q·N_q(t0) + b_Q
 * [K_j V_j] = W_KV·y_j + b_KV                      j は行の局・席・牌の 39 個（行ごとに 1 回）
 * o  = Σ_j softmax_j((q·K_j + q·E_k[c_tj])/4)·(V_j + E_v[c_tj])   c_tj は鍵の符号、E_k・E_v は [109, A]・0 で始める・FP32
 * t1 = t0 + W_O·o + b_O
 * z  = t1 + W_d·silu(W_u·N_f(t1) + b_u) + b_d
 * </pre>
 *
 * <p>pack_t は遷移の欄（{@link FieldPack}、8 次元）・遷移の数値・前線（前線の腕だけ、{@link FrontierField} の列）・B の道
 * （DiscardExact を計算の型にしたもの）を並べる。前線の腕は W_X の入力の幅が {@link FrontierField#WIDTH} だけ広く、ほかは前線なしの網と
 * 同じ。交差注意は行の K・V と符号の表を写さずに引く演算（mappedGroupedIndexedScaledDotProductAttention）で、ロジットと値を FP32 で計算し、
 * 出力で 1 回丸める。
 */
final class CandidateLayer extends Module {

  /** 遷移の欄の埋め込みの次元。 */
  static final int DIMENSION = 8;

  private static final FieldPack PACK = FieldPack.TRANSITION;
  private static final int CALL_DISCARD = InputSchema.TRANSITION.column(TransitionField.CALL_DISCARD);
  private static final int MAIN_TILE = InputSchema.TRANSITION.column(TransitionField.MAIN_TILE);

  private final int attention;
  private final int heads;
  private final double epsilon;
  private final boolean frontier;
  private final Parameter table;
  private final Parameter project;
  private final Parameter projectBias;
  private final Parameter tileEntry;
  private final Parameter queryNorm;
  private final Parameter query;
  private final Parameter queryBias;
  private final Parameter keyValue;
  private final Parameter keyValueBias;
  private final Parameter codeKey;
  private final Parameter codeValue;
  private final Parameter output;
  private final Parameter outputBias;
  private final Parameter ffnNorm;
  private final Parameter up;
  private final Parameter upBias;
  private final Parameter down;
  private final Parameter downBias;

  /**
   * パラメーターを登録する。
   *
   * @param shape 網の形
   */
  CandidateLayer(PrismShape shape) {
    int hidden = shape.hidden();
    int ffn = shape.ffn();
    attention = shape.attention();
    heads = shape.heads();
    epsilon = shape.epsilon();
    frontier = shape.frontier();
    int scalars =
        TransitionScalar.values().length
            + (frontier ? FrontierField.WIDTH : 0)
            + DiscardExact.values().length;
    int inputs = PACK.inputWidth(DIMENSION, scalars);
    table = table("table", PACK.tableRows(), DIMENSION);
    project = weight("project", new Shape(hidden, inputs), Axis.of(Role.STREAM_OUT), Axis.of(Role.IN));
    projectBias = bias("projectBias", hidden, Axis.of(Role.STREAM_OUT));
    tileEntry =
        weight("tileEntry", new Shape(hidden, hidden), Axis.of(Role.STREAM_OUT), Axis.of(Role.STREAM_IN));
    queryNorm = gain("queryNorm", hidden);
    query = weight("query", new Shape(attention, hidden), Axis.of(Role.HEAD_OUT), Axis.of(Role.STREAM_IN));
    queryBias = bias("queryBias", attention, Axis.of(Role.HEAD_OUT));
    keyValue =
        weight(
            "keyValue",
            new Shape(2L * attention, hidden),
            Axis.of(Role.HEAD_OUT, 2),
            Axis.of(Role.STREAM_IN));
    keyValueBias = bias("keyValueBias", 2 * attention, Axis.of(Role.HEAD_OUT, 2));
    codeKey =
        zeros("codeKey", new Shape(KeyBlock.CODE_COUNT, attention), Axis.of(Role.FIXED), Axis.of(Role.HEAD_OUT));
    codeValue =
        zeros(
            "codeValue",
            new Shape(KeyBlock.CODE_COUNT, attention),
            Axis.of(Role.FIXED),
            Axis.of(Role.HEAD_OUT));
    output =
        weight("output", new Shape(hidden, attention), Axis.of(Role.STREAM_OUT), Axis.of(Role.HEAD_IN));
    outputBias = bias("outputBias", hidden, Axis.of(Role.STREAM_OUT));
    ffnNorm = gain("ffnNorm", hidden);
    up = weight("up", new Shape(ffn, hidden), Axis.of(Role.INNER_OUT), Axis.of(Role.STREAM_IN));
    upBias = bias("upBias", ffn, Axis.of(Role.INNER_OUT));
    down = weight("down", new Shape(hidden, ffn), Axis.of(Role.STREAM_OUT), Axis.of(Role.INNER_IN));
    downBias = bias("downBias", hidden, Axis.of(Role.STREAM_OUT));
  }

  /**
   * 遷移の表現 z を作る。
   *
   * @param context 文脈
   * @param inputs 入力
   * @param index 索引
   * @param states 背骨の最後の正規化の後の局・席・牌 [行, 39, H]（計算の型）
   * @return z [遷移, H]（計算の型）
   */
  NDArray forward(Context context, NetworkInputs inputs, BatchIndex index, NDArray states) {
    long transitions = inputs.transitions();
    NDArray category = inputs.get(Tensor.TRANSITION_CATEGORY);
    NDArray packed =
        PACK.pack(category, index.columns(PACK), context.value(table), scalars(context, inputs));
    NDArray start =
        context
            .linear(packed, project, projectBias)
            .add(context.linear(tileStates(context, category, index, states), tileEntry, null));

    NDArray queries =
        context
            .linear(context.norm(start, queryNorm, epsilon), query, queryBias)
            .reshape(transitions, heads, PrismShape.HEAD_WIDTH);
    NDArray memory = context.linear(states, keyValue, keyValueBias);
    NDArray codes = context.own(context.fp32(codeKey).concat(context.fp32(codeValue), 1));
    NDManager working = context.working();
    NDArray attended =
        NDArrays.mappedGroupedIndexedScaledDotProductAttention(
            context.toFloat(queries),
            context.toFloat(memory),
            index.transitionRows,
            codes,
            inputs.get(Tensor.TRANSITION_CODE).toType(DataType.INT32, false),
            working.zeros(new Shape(transitions, 1, 2L * attention), DataType.FLOAT32),
            working.zeros(new Shape(transitions, 1), DataType.INT32),
            EncoderLayer.SCALE);
    NDArray residual =
        start.add(
            context.linear(
                context.toCompute(attended.reshape(transitions, attention)), output, outputBias));
    NDArray hidden =
        Context.silu(context.linear(context.norm(residual, ffnNorm, epsilon), up, upBias));
    return residual.add(context.linear(hidden, down, downBias));
  }

  /** pack_t の数値の列（計算の型）: 遷移の数値、前線（前線の腕だけ）、B の道。 */
  private NDArray[] scalars(Context context, NetworkInputs inputs) {
    NDArray transition = context.toCompute(inputs.get(Tensor.TRANSITION_SCALAR));
    NDArray discard = context.toCompute(inputs.get(Tensor.DISCARD_EXACT));
    return frontier
        ? new NDArray[] {transition, context.toCompute(inputs.frontier()), discard}
        : new NDArray[] {transition, discard};
  }

  /** y_T[τ(t)]（τ がない遷移は 0）[遷移, H]。 */
  private static NDArray tileStates(
      Context context, NDArray category, BatchIndex index, NDArray states) {
    NDArray values = category.toType(DataType.INT32, false);
    NDArray callDiscard = values.get(new NDIndex(":, {}", CALL_DISCARD));
    NDArray mainTile = values.get(new NDIndex(":, {}", MAIN_TILE));
    NDArray tile = NDArrays.where(callDiscard.gt(0), callDiscard, mainTile);
    NDArray position =
        index
            .transitionRows
            .mul(TokenType.FIXED_TOKENS)
            .add(TokenType.TILE.first() - 1)
            .add(tile.maximum(1));
    long hidden = states.getShape().get(2);
    NDArray flat = states.reshape(-1, hidden);
    NDArray present = tile.gt(0).toType(context.compute(), false).expandDims(1);
    return NDArrays.gatherRows(flat, position).mul(present);
  }
}
