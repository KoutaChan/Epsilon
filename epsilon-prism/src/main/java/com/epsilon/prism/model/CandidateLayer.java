package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat.TransitionRole;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.FitProbe.Stage;

/**
 * 候補層（HIP の K6 に当たる）。遷移ごとに 1 段だけ通し、候補どうしは参照しない。
 *
 * <pre>
 * t0 = W_X·pack_t + b_X + W_E·y_T[τ(t)]          τ(t) は捨てる牌（鳴いた後の打牌、なければ主牌。どちらもなければ項なし）
 * q  = W_Q·N_q(t0) + b_Q
 * [K_j V_j] = W_KV·y_j + b_KV                      j は行の局・席・牌（四人打ち 39 個・三人打ち 31 個。行ごとに 1 回）
 * o  = Σ_j softmax_j((q·K_j + q·E_k[c_tj])/4)·(V_j + E_v[c_tj])   c_tj はキーのコード。E_k・E_v は [コードの数, A]、0 で初期化、FP32
 * t1 = t0 + W_O·o + b_O
 * z  = t1 + W_d·silu(W_u·N_f(t1) + b_u) + b_d
 * </pre>
 *
 * <p>pack_t は、遷移のフィールド（{@link FieldPack}、8 次元）、遷移の数値、B のヘッドが読む厳密なフィールド（計算の型にしたもの）を
 * 並べたもの。Cross-Attention には、行の K・V とコードの表をコピーせずに引く演算（mappedGroupedIndexedScaledDotProductAttention）を
 * 使う。ロジットと値は FP32 で計算し、出力で 1 回丸める。FFN の幅は
 * {@link PrismShape#candidateFfn()}。
 */
final class CandidateLayer extends Module {

  /** 遷移のフィールドの埋め込みの次元。 */
  static final int DIMENSION = 8;

  private final InputFormat format;
  private final FieldPack pack;
  private final int attention;
  private final int heads;
  private final double epsilon;
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
   * @param shape ネットワークの形
   * @param format 入力の形式
   */
  CandidateLayer(PrismShape shape, InputFormat format) {
    this.format = format;
    pack = FieldPack.of(format).transition();
    int hidden = shape.hidden();
    int ffn = shape.candidateFfn();
    attention = shape.attention();
    heads = shape.heads();
    epsilon = shape.epsilon();
    int scalars = format.width(Tensor.TRANSITION_SCALAR) + format.width(Tensor.DISCARD_EXACT);
    int inputs = pack.inputWidth(DIMENSION, scalars);
    table = table("table", pack.tableRows(), DIMENSION);
    project = weight("project", new Shape(hidden, inputs));
    projectBias = bias("projectBias", hidden);
    tileEntry = weight("tileEntry", new Shape(hidden, hidden));
    queryNorm = gain("queryNorm", hidden);
    query = weight("query", new Shape(attention, hidden));
    queryBias = bias("queryBias", attention);
    keyValue = weight("keyValue", new Shape(2L * attention, hidden));
    keyValueBias = bias("keyValueBias", 2 * attention);
    codeKey = zeros("codeKey", new Shape(format.keys().codes(), attention));
    codeValue = zeros("codeValue", new Shape(format.keys().codes(), attention));
    output = weight("output", new Shape(hidden, attention));
    outputBias = bias("outputBias", hidden);
    ffnNorm = gain("ffnNorm", hidden);
    up = weight("up", new Shape(ffn, hidden));
    upBias = bias("upBias", ffn);
    down = weight("down", new Shape(hidden, ffn));
    downBias = bias("downBias", hidden);
  }

  /**
   * 遷移の表現 z を作る。
   *
   * @param context 文脈
   * @param inputs 入力
   * @param index インデックス
   * @param states バックボーンの最後の正規化を通した局・席・牌 [行, 局・席・牌, H]（計算の型）
   * @return z [遷移, H]（計算の型）
   */
  NDArray forward(Context context, NetworkInputs inputs, BatchIndex index, NDArray states) {
    return context.scoped(
        scope -> represent(scope, inputs, index, states),
        states,
        inputs.get(Tensor.TRANSITION_CATEGORY),
        inputs.get(Tensor.TRANSITION_SCALAR),
        inputs.get(Tensor.TRANSITION_CODE),
        inputs.get(Tensor.DISCARD_EXACT));
  }

  private NDArray represent(
      Context context, NetworkInputs inputs, BatchIndex index, NDArray states) {
    long transitions = inputs.transitions();
    NDArray category = inputs.get(Tensor.TRANSITION_CATEGORY);
    NDArray packed =
        pack.pack(category, index.columns(pack), context.value(table), scalars(context, inputs));
    // 計測のときは t0 と、Cross-Attention の FP32 の入力と出力を葉にして、逆伝播を段に分けて測る（Context#probeCut）
    NDArray start =
        context.probeCut(
            context
                .linear(packed, project, projectBias)
                .add(context.linear(tileStates(context, format, category, index, states), tileEntry, null)),
            Stage.CANDIDATE_START_BACKWARD,
            0);

    NDArray queries =
        context
            .linear(context.norm(start, queryNorm, epsilon), query, queryBias)
            .reshape(transitions, heads, PrismShape.HEAD_WIDTH);
    NDArray memory = context.linear(states, keyValue, keyValueBias);
    NDArray codes = context.own(context.fp32(codeKey).concat(context.fp32(codeValue), 1));
    NDManager working = context.working();
    NDArray attended =
        context.probeCut(
            NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                context.probeCut(context.toFloat(queries), Stage.CROSS_PROJECTION_BACKWARD, 0),
                context.probeCut(context.toFloat(memory), Stage.CROSS_PROJECTION_BACKWARD, 0),
                index.transitionRows.array(),
                codes,
                inputs.get(Tensor.TRANSITION_CODE).toType(DataType.INT32, false),
                working.zeros(new Shape(transitions, 1, 2L * attention), DataType.FLOAT32),
                working.zeros(new Shape(transitions, 1), DataType.INT32),
                EncoderLayer.SCALE),
            Stage.CROSS_ATTENTION_BACKWARD,
            0);
    NDArray residual =
        start.add(
            context.linear(
                context.toCompute(attended.reshape(transitions, attention)), output, outputBias));
    NDArray hidden =
        Context.silu(context.linear(context.norm(residual, ffnNorm, epsilon), up, upBias));
    return residual.add(context.linear(hidden, down, downBias));
  }

  /** pack_t の数値の列（計算の型）。遷移の数値、B のヘッドが読む厳密なフィールドの順。 */
  private NDArray[] scalars(Context context, NetworkInputs inputs) {
    NDArray transition = context.toCompute(inputs.get(Tensor.TRANSITION_SCALAR));
    NDArray discard = context.toCompute(inputs.get(Tensor.DISCARD_EXACT));
    return new NDArray[] {transition, discard};
  }

  /**
   * 遷移ごとの τ を返す。τ は捨てる牌で、鳴いた後の打牌があればそれ、なければ主牌。
   *
   * @param format 入力の形式
   * @param category 遷移のフィールド [遷移, フィールド]
   * @return τ の牌のトークンの番号 + 1 [遷移]（INT32。牌がなければ 0）
   */
  static NDArray discardTiles(InputFormat format, NDArray category) {
    NDArray values = category.toType(DataType.INT32, false);
    NDArray callDiscard =
        values.get(new NDIndex(":, {}", format.transitionColumn(TransitionRole.CALL_DISCARD)));
    NDArray mainTile = values.get(new NDIndex(":, {}", format.transitionColumn(TransitionRole.MAIN_TILE)));
    return NDArrays.where(callDiscard.gt(0), callDiscard, mainTile);
  }

  /** y_T[τ(t)]（τ がない遷移は 0）[遷移, H]。 */
  private static NDArray tileStates(
      Context context, InputFormat format, NDArray category, BatchIndex index, NDArray states) {
    NDArray tile = discardTiles(format, category);
    NDArray position =
        index
            .transitionRows
            .array()
            .mul(format.fixedTokens())
            .add(TokenType.TILE.first(format) - 1)
            .add(tile.maximum(1));
    long hidden = states.getShape().get(2);
    NDArray flat = states.reshape(-1, hidden);
    NDArray present = tile.gt(0).toType(context.compute(), false).expandDims(1);
    return context
        .probeCut(NDArrays.gatherRows(flat, position), Stage.GATHER_BACKWARD, 0)
        .mul(present);
  }
}
