package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.agent.facts.MahjongTileRelation;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.InputFormat.RelationRole;

/**
 * 候補層の 2 段目（Attention だけ）。遷移ごとに、局・席・牌の最後の状態とイベントをもう一度読む。候補どうしは参照しない。
 *
 * <pre>
 * q  = W_Q·N_q(z1) + b_Q                               z1 は候補層の 1 段目の出力
 * [K_j V_j] = W_KV·y_j + b_KV                            j は局・席・牌（最後の正規化の後）とイベント（層 0 の出力を N_e で正規化）
 * o  = Σ_j softmax_j((q·K_j + q·E_k[c_tj])/4)·(V_j + E_v[c_tj])
 * z  = z1 + W_O·o + b_O                                  W_O・b_O は 0 で初期化する（初期化したままなら 1 段の候補層と同じ関数）
 * </pre>
 *
 * <p>キーのコード c_tj は、局・席・牌のキーでは 1 段目と同じ遷移のコード。イベントのキーでは「τ(t) とイベントの牌の関係」の行を使う。
 * この行は遷移のコードの後ろにあり、どちらかに牌がなければ 1 行、両方にあれば {@link MahjongTileRelation} の種類ごとに 1 行ある。
 * τ(t) は捨てる牌（{@link CandidateLayer#discardTiles}）。E_k・E_v は FP32 で、0 で初期化する。
 *
 * <p>キーとバリューは行ごとに 1 回作り、その行の全ての遷移で共有する（mappedGroupedIndexedScaledDotProductAttention）。行ごとに
 * イベントの数が違うので、キーの 16 次元に、定数 1 のクエリと組む 17 番目の次元を足す。パディングのイベントのキーにだけ、この次元に
 * −1e9 を置いて Attention から外す。ロジットと値は FP32 で計算し、出力で 1 回丸める。
 */
final class CandidateReread extends Module {

  /** パディングのキーで、足した次元に置く値。 */
  private static final float MASKED = -1e9f;

  private final InputFormat format;

  /** イベントのキーで、牌の関係がないときの行。番号は遷移のコードの数で、1 段目のコードの表の行の数と同じ。 */
  private final int noRelation;

  /**
   * 学習するコードの表の行の数（遷移のコード、牌の関係がない行、牌の関係の種類）。この数は、パディングのイベントに使うキーの行
   * （学習しない）の番号でもある。
   */
  private final int codeRows;

  private final int tileColumn;
  private final int attention;
  private final int heads;
  private final double epsilon;
  private final Parameter queryNorm;
  private final Parameter query;
  private final Parameter queryBias;
  private final Parameter eventNorm;
  private final Parameter keyValue;
  private final Parameter keyValueBias;
  private final Parameter codeKey;
  private final Parameter codeValue;
  private final Parameter output;
  private final Parameter outputBias;

  /**
   * パラメーターを登録する。
   *
   * @param shape ネットワークの形
   * @param format 入力の形式
   */
  CandidateReread(PrismShape shape, InputFormat format) {
    this.format = format;
    noRelation = format.keys().codes();
    codeRows = noRelation + 1 + MahjongTileRelation.values().length;
    tileColumn = format.relationColumn(RelationRole.TILE, 0);
    int hidden = shape.hidden();
    attention = shape.attention();
    heads = shape.heads();
    epsilon = shape.epsilon();
    queryNorm = gain("queryNorm", hidden);
    query = weight("query", new Shape(attention, hidden));
    queryBias = bias("queryBias", attention);
    eventNorm = gain("eventNorm", hidden);
    keyValue = weight("keyValue", new Shape(2L * attention, hidden));
    keyValueBias = bias("keyValueBias", 2 * attention);
    codeKey = zeros("codeKey", new Shape(codeRows, attention));
    codeValue = zeros("codeValue", new Shape(codeRows, attention));
    output = zeros("output", new Shape(hidden, attention));
    outputBias = bias("outputBias", hidden);
  }

  /**
   * 遷移の表現を読み直す。
   *
   * @param context 文脈
   * @param inputs 入力
   * @param index インデックス
   * @param start 1 段目の出力 z1 [遷移, H]（計算の型）
   * @param states バックボーンの最後の正規化を通した局・席・牌 [行, 局・席・牌, H]（計算の型）
   * @param events 層 0 が出力したイベント [行, 最大のイベント数, H]（計算の型。イベントがないバッチでは null）。Attention の記録が
   *     あるときは局・席・牌の最初の層の入力の葉から切り出したもので、2 段目の勾配はその葉に溜まる
   * @return z [遷移, H]（計算の型）
   */
  NDArray forward(
      Context context,
      NetworkInputs inputs,
      BatchIndex index,
      NDArray start,
      NDArray states,
      NDArray events) {
    return context.scoped(
        scope -> reread(scope, inputs, index, start, states, events),
        start,
        states,
        events,
        inputs.get(Tensor.TRANSITION_CODE),
        inputs.get(Tensor.RELATION),
        inputs.get(Tensor.TRANSITION_CATEGORY),
        index.tokenMask.array(),
        index.tileRelations.array());
  }

  private NDArray reread(
      Context context,
      NetworkInputs inputs,
      BatchIndex index,
      NDArray start,
      NDArray states,
      NDArray events) {
    long transitions = inputs.transitions();
    NDManager working = context.working();
    NDArray queries =
        context
            .toFloat(context.linear(context.norm(start, queryNorm, epsilon), query, queryBias))
            .reshape(transitions, heads, PrismShape.HEAD_WIDTH);
    NDArray keys =
        events == null
            ? states
            : NDArrays.concat(new NDList(states, context.norm(events, eventNorm, epsilon)), 1);
    NDArray memory = context.toFloat(context.linear(keys, keyValue, keyValueBias));
    // コードの表のパラメーターから作った配列はモデルのマネージャーに付くので、作るたびに作業用のマネージャーへ付け直す
    NDArray learned =
        context.own(
            NDArrays.concat(
                new NDList(withMask(context, context.fp32(codeKey), 0f), context.fp32(codeValue)), 1));
    NDArray table = context.own(NDArrays.concat(new NDList(learned, padding(context)), 0));
    NDArray attended =
        NDArrays.mappedGroupedIndexedScaledDotProductAttention(
            NDArrays.concat(
                new NDList(queries, working.ones(new Shape(transitions, heads, 1))), 2),
            NDArrays.concat(
                new NDList(
                    withMask(context, memory.get(new NDIndex(":, :, 0:{}", attention)), 0f),
                    memory.get(new NDIndex(":, :, {}:", attention))),
                2),
            index.transitionRows.array(),
            table,
            codes(inputs, index),
            working.zeros(new Shape(transitions, 1, packedWidth()), DataType.FLOAT32),
            working.zeros(new Shape(transitions, 1), DataType.INT32),
            EncoderLayer.SCALE);
    return start.add(
        context.linear(
            context.toCompute(attended.reshape(transitions, attention)), output, outputBias));
  }

  /** ヘッドごとに 17 番目の次元を足したキーの幅と、バリューの幅の和。 */
  private int packedWidth() {
    return heads * (PrismShape.HEAD_WIDTH + 1) + attention;
  }

  /** キー [..., A] の各ヘッドの後ろに値 {@code value} の次元を 1 つ足し、[..., ヘッド × 17] にする（作業用のマネージャーに付ける）。 */
  private NDArray withMask(Context context, NDArray keys, float value) {
    Shape shape = keys.getShape();
    long[] split = new long[shape.dimension() + 1];
    long[] mask = new long[shape.dimension() + 1];
    for (int axis = 0; axis < shape.dimension() - 1; axis++) {
      split[axis] = shape.get(axis);
      mask[axis] = shape.get(axis);
    }
    split[shape.dimension() - 1] = heads;
    split[shape.dimension()] = PrismShape.HEAD_WIDTH;
    mask[shape.dimension() - 1] = heads;
    mask[shape.dimension()] = 1;
    NDArray column = context.working().full(new Shape(mask), value, DataType.FLOAT32);
    long[] joined = shape.getShape().clone();
    joined[shape.dimension() - 1] = (long) heads * (PrismShape.HEAD_WIDTH + 1);
    NDArray perHead = context.own(keys.reshape(new Shape(split)));
    NDArray widened = context.own(NDArrays.concat(new NDList(perHead, column), shape.dimension()));
    return context.own(widened.reshape(new Shape(joined)));
  }

  /** パディングのイベントのキーとバリューの行 [1, 幅]。キーは足した次元だけ −1e9 でほかは 0、バリューは 0。 */
  private NDArray padding(Context context) {
    NDManager working = context.working();
    NDArray keys = withMask(context, working.zeros(new Shape(1, attention), DataType.FLOAT32), MASKED);
    return NDArrays.concat(
        new NDList(keys, working.zeros(new Shape(1, attention), DataType.FLOAT32)), 1);
  }

  /**
   * キーのコード [遷移, 局・席・牌 + 最大のイベント数]（INT32）。局・席・牌のキーには遷移のコード、イベントのキーには τ(t) と
   * イベントの牌の関係、パディングには Attention から外す行を置く。
   */
  private NDArray codes(NetworkInputs inputs, BatchIndex index) {
    NDArray fixed = inputs.get(Tensor.TRANSITION_CODE).toType(DataType.INT32, false);
    if (index.maxEvents == 0) {
      return fixed;
    }
    int rows = inputs.rows();
    NDArray transitionRows = index.transitionRows.array();
    NDArray tiles =
        NDArrays.scatterRows(
                inputs
                    .get(Tensor.RELATION)
                    .get(new NDIndex(":, {}:{}", tileColumn, tileColumn + 1))
                    .toType(DataType.INT32, false),
                index.tokenSlots.array(),
                (long) rows * index.tokens)
            .reshape(rows, index.tokens);
    NDIndex eventPositions = new NDIndex(":, {}:", format.fixedTokens());
    NDArray eventTiles = NDArrays.gatherRows(tiles, transitionRows).get(eventPositions);
    NDArray present =
        NDArrays.gatherRows(index.tokenMask.array(), transitionRows).get(eventPositions).gt(0);
    NDArray discard =
        CandidateLayer.discardTiles(format, inputs.get(Tensor.TRANSITION_CATEGORY)).expandDims(1);
    NDArray both = discard.gt(0).logicalAnd(eventTiles.gt(0));
    NDArray pair = discard.sub(1).maximum(0).mul(format.tiles()).add(eventTiles.sub(1).maximum(0));
    NDArray relation =
        NDArrays.gatherRows(index.tileRelations.array(), pair.reshape(-1)).reshape(pair.getShape());
    NDArray related = relation.add(noRelation + 1).mul(both.toType(DataType.INT32, false));
    NDArray unrelated = both.logicalNot().toType(DataType.INT32, false).mul(noRelation);
    NDArray code = related.add(unrelated);
    NDArray events = NDArrays.where(present, code, code.zerosLike().add(codeRows));
    return NDArrays.concat(new NDList(fixed, events), 1);
  }
}
