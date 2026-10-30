package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.model.FitProbe.Stage;
import com.epsilon.prism.model.WidthPlan.Axis;
import com.epsilon.prism.model.WidthPlan.Role;

/**
 * 背骨の 1 層（HIP の K2・K3、最後の層は K4・K5 に当たる）。pre-RMSNorm で、x += Attn(N(x))、x += FFN(N(x))。
 *
 * <p>注意は H/32 頭 × 16 で、ロジットは q·k/4 に関係の偏り Σ_s ρ[16s + c_s(i, j), 頭] を足す。q·k・偏り・softmax・p·v は全て FP32 で、
 * 出力で 1 回丸める（packedRelationScaledDotProductAttention）。ρ は層ごとの FP32 の表 [112, 頭] で、0 で始める。packed 演算の表の
 * 後ろの A 列（相対の値）は定数 0 にし、パラメーターにしない。FFN は SiLU。
 *
 * <p>qkv の行は q・k・v の塊を頭の順に並べる（[3A, H]）。問い合わせを先頭の n 個に限る層（最後の層は局・席・牌の 39 個）は、その n 個
 * だけを更新して返し、鍵と値は全てのトークンから作る。
 *
 * <p>段の時計の印（{@link FitProbe}）は層の番号で数える: 順伝播は正規化と q・kv の射影、注意、O 射影と FFN。逆伝播は注意の記録の葉で区切り、
 * 層の入力の葉は前の層の O 射影と FFN（最初の層は埋め込み）、正規化の出力の葉は注意の前の正規化と q・kv の射影として計る。
 */
public final class EncoderLayer extends Module {

  /** 注意のロジットの尺度（1/√16）。 */
  static final double SCALE = 1.0 / Math.sqrt(PrismShape.HEAD_WIDTH);

  /** 網の中の層の番号（段の時計の印だけが読む）。 */
  private final int layer;

  private final int attention;
  private final int heads;
  private final double epsilon;
  private final Parameter attentionNorm;
  private final Parameter qkv;
  private final Parameter qkvBias;
  private final Parameter relation;
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
   * @param layer 網の中の層の番号（0 から）
   */
  public EncoderLayer(PrismShape shape, int layer) {
    this.layer = layer;
    int hidden = shape.hidden();
    int ffn = shape.ffn();
    attention = shape.attention();
    heads = shape.heads();
    epsilon = shape.epsilon();
    attentionNorm = gain("attentionNorm", hidden);
    qkv =
        weight(
            "qkv", new Shape(3L * attention, hidden), Axis.of(Role.HEAD_OUT, 3), Axis.of(Role.STREAM_IN));
    qkvBias = bias("qkvBias", 3 * attention, Axis.of(Role.HEAD_OUT, 3));
    relation =
        zeros(
            "relation",
            new Shape(RelationClass.ROWS, heads),
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
   * 層を通す。
   *
   * @param context 文脈
   * @param tokens 流れ [行, S, H]（計算の型）
   * @param mask トークンの在否 [行, S]
   * @param codes 関係の符号 [行, S, S, 2]（INT16）
   * @param queries 問い合わせにする先頭のトークンの数
   * @return 更新した先頭 queries 個 [行, queries, H]（計算の型）
   */
  public NDArray forward(Context context, NDArray tokens, NDArray mask, NDArray codes, int queries) {
    AttentionTape tape = context.tape();
    FitProbe probe = context.probe();
    // 自前の注意の逆伝播をつなぐため、層の入力と、q・kv が共に読む正規化の出力を葉にする（AttentionTape）
    NDArray input =
        layer == 0
            ? cut(tape, tokens, Stage.EMBED_BACKWARD, 0)
            : cut(tape, tokens, Stage.O_FFN_BACKWARD, layer - 1);
    boolean all = queries == input.getShape().get(1);
    NDArray normalized =
        cut(tape, context.norm(input, attentionNorm, epsilon), Stage.NORM_QKV_BACKWARD, layer);
    NDArray weights = context.value(qkv);
    NDArray biases = context.value(qkvBias);
    NDArray keyValue =
        context.linear(
            normalized,
            context.slice(weights, attention, 3L * attention),
            context.slice(biases, attention, 3L * attention));
    NDArray query =
        context.linear(
            all ? normalized : prefix(normalized, queries),
            context.slice(weights, 0, attention),
            context.slice(biases, 0, attention));
    probe.mark(Stage.NORM_QKV, layer);
    NDArray attended =
        tape == null
            ? NDArrays.packedRelationScaledDotProductAttention(
                query,
                keyValue,
                mask,
                all ? codes : prefix(codes, queries),
                relationTable(context),
                heads,
                RelationClass.ENTRIES,
                SCALE)
            : tape.attend(
                query,
                keyValue,
                mask,
                codes,
                context.fp32(relation),
                heads,
                RelationClass.ENTRIES,
                SCALE,
                layer);
    probe.mark(Stage.ATTENTION, layer);
    NDArray residual =
        (all ? input : prefix(input, queries)).add(context.linear(attended, output, outputBias));
    NDArray hidden =
        Context.silu(context.linear(context.norm(residual, ffnNorm, epsilon), up, upBias));
    NDArray updated = residual.add(context.linear(hidden, down, downBias));
    probe.mark(Stage.O_FFN, layer);
    return updated;
  }

  /** packed 演算の表 [112, 頭 + A]（偏りの列が先、相対の値は 0。FP32）。 */
  private NDArray relationTable(Context context) {
    NDArray bias = context.fp32(relation);
    NDArray values =
        context.working().zeros(new Shape(RelationClass.ROWS, attention), DataType.FLOAT32);
    return context.own(bias.concat(values, 1));
  }

  /** 記録があれば値を葉にする（{@link AttentionTape#cut}。stage・layer は葉の逆伝播を計る段。なければそのまま）。 */
  private static NDArray cut(AttentionTape tape, NDArray source, Stage stage, int layer) {
    return tape == null ? source : tape.cut(source, stage, layer);
  }

  /** 第 2 軸の先頭 n 個。 */
  private static NDArray prefix(NDArray array, int count) {
    return array.get(new NDIndex(":, 0:{}", count));
  }
}
