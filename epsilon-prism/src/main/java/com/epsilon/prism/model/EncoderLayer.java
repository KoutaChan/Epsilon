package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.hip.HipFfn;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.FitProbe.Stage;

/**
 * バックボーンの 1 層。HIP の K2・K3 に当たり、局・席・牌の層は K4 と K3・K5 に当たる。pre-RMSNorm の形で、x += Attn(N(x))、
 * x += FFN(N(x)) と更新する。
 *
 * <p>Attention のヘッドは H/32 個で、ヘッドの幅は 16。ロジットは q·k/4 に関係のバイアス Σ_s ρ[16s + c_s(i, j), ヘッド] を足したもの。
 * q·k・バイアス・softmax・p·v は全て FP32 で計算し、出力で 1 回だけ丸める（{@code packedRelationScaledDotProductAttention}）。ρ は層ごとの
 * FP32 の表 [112, ヘッド] で、0 で初期化する。packed 演算の表の後ろの、相対のバリューの列は定数 0 にし、パラメーターにしない。FFN の
 * 活性化は SiLU。
 *
 * <p>qkv の行は、q・k・v のブロックをそれぞれヘッドの順に並べる（[3A, H]）。クエリを先頭の n 個に限る層（局・席・牌の層。四人打ちは
 * 39 個、三人打ちは 31 個）は、その n 個だけを更新して返す。キーとバリューは全てのトークンから作る。局・席・牌の数は入力の形式で決まる。
 *
 * <p>層の役と FFN の幅は {@link PrismShape#layers()} で決まる。型ごとの FFN を持つ層では、局・席・牌のトークンだけがクエリになる。
 * O 射影の後を局・席と牌に分け、それぞれの残差を別の FFN（それぞれ前に RMSNorm が付く）に通す。Attention とその前の正規化は、型によらず
 * 1 つだけ持つ。
 *
 * <p>Attention の記録があるときは、層の入力と残差を葉にする（どちらも正規化と残差の和の両方が読む値）。正規化の出力（q と kv の射影が
 * 両方とも読む）は、記録の正規化が葉にする（{@link AttentionTape}）。型ごとの FFN を持つ層では、残差は局・席と牌の 2 つの連続した配列に
 * なり、それぞれを葉にして記録の正規化に渡す。
 *
 * <p>計測点（{@link FitProbe}）では、段の時間を層ごとに分けて数える。順伝播は「正規化と q・kv の射影」「Attention」「O 射影と FFN」の
 * 3 段に分ける。逆伝播は Attention の記録の葉で区切る。層の入力の葉は前の層の「O 射影と FFN」（最初の層では埋め込み）に、Attention の
 * 前の正規化は「Attention の前の正規化と q・kv の射影」に、残差の葉と FFN の前の正規化は「O 射影と FFN」に数える。
 */
public final class EncoderLayer extends Module {

  /** Attention のロジットに掛ける倍率（1/√16）。 */
  static final double SCALE = 1.0 / Math.sqrt(PrismShape.HEAD_WIDTH);

  /** ネットワークの中の層の番号（計測点に段の切れ目を記録するときだけ使う）。 */
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

  /** 全てのトークンが通る FFN（型ごとの FFN を持つ層では null）。 */
  private final Ffn sharedFfn;

  /** 局・席のトークンの FFN（型ごとの FFN を持たない層では null）。 */
  private final Ffn contextFfn;

  /** 牌のトークンの FFN（型ごとの FFN を持たない層では null）。 */
  private final Ffn tileFfn;

  /** 局・席のトークンの数（型ごとの FFN で、ここから先を牌として分ける）。 */
  private final int contextTokens;

  /** 局・席・牌のトークンの数（局・席・牌の層でクエリになる数）。 */
  private final int fixedTokens;

  /** FFN の重み（前に付く RMSNorm のゲイン、上げと下げの Linear）。 */
  private record Ffn(Parameter norm, Parameter up, Parameter upBias, Parameter down, Parameter downBias) {}

  /** 局・席・牌の層の出力（最後の正規化の前）。 */
  sealed interface FixedTokens permits SharedTokens, TypedTokens {

    /**
     * 次の局・席・牌の層の入力を作る。
     *
     * @param events イベント [行, イベントの最大数, H]（なければ null）
     * @return 局・席・牌の後にイベントを並べた入力 [行, S, H]
     */
    NDArray withEvents(NDArray events);
  }

  /**
   * 共有の FFN を持つ層の出力。
   *
   * @param tokens 局・席・牌のトークン [行, 局・席・牌の数, H]（計算の型）
   */
  record SharedTokens(NDArray tokens) implements FixedTokens {

    @Override
    public NDArray withEvents(NDArray events) {
      return events == null ? tokens : NDArrays.concat(new NDList(tokens, events), 1);
    }
  }

  /**
   * 型ごとの FFN を持つ層の出力。局・席と牌を、それぞれ連続した配列で持つ（記録の正規化が連続した配列を読むため）。
   *
   * @param context 局・席のトークン [行, 1 + 席の数, H]（計算の型）
   * @param tiles 牌のトークン [行, 牌の数, H]（計算の型）
   */
  record TypedTokens(NDArray context, NDArray tiles) implements FixedTokens {

    @Override
    public NDArray withEvents(NDArray events) {
      NDList parts = new NDList(context, tiles);
      if (events != null) {
        parts.add(events);
      }
      return NDArrays.concat(parts, 1);
    }
  }

  /**
   * パラメーターを登録する。
   *
   * @param shape ネットワークの形
   * @param layer ネットワークの中の層の番号（0 から）
   * @param format 入力の形式（局・席・牌のトークンの数を決める）
   */
  public EncoderLayer(PrismShape shape, int layer, InputFormat format) {
    this.layer = layer;
    HipFfn layerFfn = shape.layers().get(layer).ffn();
    contextTokens = 1 + format.seats();
    fixedTokens = format.fixedTokens();
    int hidden = shape.hidden();
    attention = shape.attention();
    heads = shape.heads();
    epsilon = shape.epsilon();
    attentionNorm = gain("attentionNorm", hidden);
    qkv = weight("qkv", new Shape(3L * attention, hidden));
    qkvBias = bias("qkvBias", 3 * attention);
    relation = zeros("relation", new Shape(RelationClass.ROWS, heads));
    output = weight("output", new Shape(hidden, attention));
    outputBias = bias("outputBias", hidden);
    sharedFfn =
        layerFfn instanceof HipFfn.Shared shared ? ffn("ffnNorm", "up", "down", hidden, shared.width()) : null;
    contextFfn =
        layerFfn instanceof HipFfn.Typed typed
            ? ffn("contextNorm", "contextUp", "contextDown", hidden, typed.context())
            : null;
    tileFfn =
        layerFfn instanceof HipFfn.Typed typed ? ffn("tileNorm", "tileUp", "tileDown", hidden, typed.tile()) : null;
  }

  /** FFN の重みを登録する。順はゲイン、上げ、上げのバイアス、下げ、下げのバイアス。 */
  private Ffn ffn(String norm, String up, String down, int hidden, int width) {
    Parameter gain = gain(norm, hidden);
    Parameter upWeight = weight(up, new Shape(width, hidden));
    Parameter upBias = bias(up + "Bias", width);
    Parameter downWeight = weight(down, new Shape(hidden, width));
    Parameter downBias = bias(down + "Bias", hidden);
    return new Ffn(gain, upWeight, upBias, downWeight, downBias);
  }

  /**
   * 共有の FFN を持つ層を通す（バックボーンの、全てのトークンを扱う層）。記録があれば入力を葉にする。その葉の逆伝播の時間は、最初の層では
   * 埋め込みの段に、ほかの層では前の層の O 射影と FFN の段に数える。型ごとの FFN を持つ層は {@link #fixed} で通す。
   *
   * @param context 計算の文脈
   * @param tokens 各トークンの表現 [行, S, H]（計算の型）
   * @param mask トークンがあるか [行, S]
   * @param codes 関係のコード [行, S, S, 2]（INT16）
   * @param queries クエリにする先頭のトークンの数
   * @return 更新した先頭 {@code queries} 個のトークン [行, queries, H]（計算の型）
   * @throws IllegalStateException 型ごとの FFN を持つ層のとき
   */
  public NDArray forward(Context context, NDArray tokens, NDArray mask, NDArray codes, int queries) {
    if (sharedFfn == null) {
      throw new IllegalStateException("layer " + layer + " has typed FFNs; pass it through fixed()");
    }
    NDArray input =
        layer == 0
            ? context.cut(tokens, Stage.EMBED_BACKWARD, 0)
            : context.cut(tokens, Stage.O_FFN_BACKWARD, layer - 1);
    return shared(context, input, mask, codes, queries);
  }

  /**
   * 局・席・牌のトークンだけをクエリにする層を通す（バックボーンの局・席・牌の層）。
   *
   * @param context 計算の文脈
   * @param entry 層の入力 [行, S, H]（計算の型）。先頭に局・席・牌、続いてイベントが並ぶ。記録があるときは呼び出し元が葉にした値
   * @param mask トークンがあるか [行, S]
   * @param codes 関係のコード [行, S, S, 2]（INT16）
   * @return 局・席・牌の出力
   */
  FixedTokens fixed(Context context, NDArray entry, NDArray mask, NDArray codes) {
    return sharedFfn != null
        ? new SharedTokens(shared(context, entry, mask, codes, fixedTokens))
        : typed(context, entry, mask, codes);
  }

  /** 共有の FFN を持つ層の計算。x + Attn(N(x)) の残差を葉にしてから、x + FFN(N(x)) を求める（クエリは先頭 {@code queries} 個）。 */
  private NDArray shared(Context context, NDArray input, NDArray mask, NDArray codes, int queries) {
    NDArray residual = context.scoped(scope -> attend(scope, input, mask, codes, queries), input);
    NDArray updated = context.scoped(scope -> feedForward(scope, residual, sharedFfn), residual);
    context.probe().mark(Stage.O_FFN, layer);
    return updated;
  }

  /**
   * 型ごとの FFN を持つ層の計算。局・席・牌をクエリにして Attention を求めた後、局・席と牌のそれぞれで O 射影と残差の和を作って葉にし、
   * その型の FFN に通す。
   */
  private TypedTokens typed(Context context, NDArray entry, NDArray mask, NDArray codes) {
    NDArray attended = context.scoped(scope -> attention(scope, entry, mask, codes, fixedTokens), entry);
    NDArray seats =
        context.scoped(
            scope -> typedHalf(scope, entry, attended, 0, contextTokens, contextFfn), entry, attended);
    NDArray tileTokens =
        context.scoped(
            scope -> typedHalf(scope, entry, attended, contextTokens, fixedTokens, tileFfn), entry, attended);
    context.probe().mark(Stage.O_FFN, layer);
    return new TypedTokens(seats, tileTokens);
  }

  /** Attention の側の半分。x + Attn(N(x)) の先頭 {@code queries} 個を葉にして返す。 */
  private NDArray attend(Context context, NDArray input, NDArray mask, NDArray codes, int queries) {
    NDArray projected = context.linear(attention(context, input, mask, codes, queries), output, outputBias);
    NDArray kept = queries == input.getShape().get(1) ? input : prefix(input, queries);
    return context.cut(kept.add(projected), Stage.O_FFN_BACKWARD, layer);
  }

  /** 型ごとの半分。トークンの区間 [from, to) について x + O(Attn) を葉にし、その型の FFN に通す。 */
  private NDArray typedHalf(
      Context context, NDArray entry, NDArray attended, int from, int to, Ffn ffn) {
    NDIndex span = new NDIndex(":, {}:{}", from, to);
    NDArray projected = context.linear(attended.get(span), output, outputBias);
    NDArray residual = context.cut(entry.get(span).add(projected), Stage.O_FFN_BACKWARD, layer);
    return feedForward(context, residual, ffn);
  }

  /** O 射影の前の Attn(N(x)) [行, queries, A]。クエリは先頭 {@code queries} 個で、キーとバリューは全てのトークンから作る。 */
  private NDArray attention(Context context, NDArray input, NDArray mask, NDArray codes, int queries) {
    boolean all = queries == input.getShape().get(1);
    NDArray normalized =
        context.backboneNorm(input, attentionNorm, epsilon, Stage.NORM_QKV_BACKWARD, layer);
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
    context.probe().mark(Stage.NORM_QKV, layer);
    AttentionTape tape = context.tape();
    NDArray attended =
        tape == null
            ? eager(context, query, keyValue, mask, all ? codes : prefix(codes, queries))
            : tape.attend(
                context.working(),
                query,
                keyValue,
                mask,
                codes,
                context.fp32(relation),
                heads,
                RelationClass.ENTRIES,
                SCALE,
                layer);
    context.probe().mark(Stage.ATTENTION, layer);
    return attended;
  }

  /** x + FFN(N(x)) を求める。N はバックボーンの正規化。記録があるとき、x は葉になっている。 */
  private NDArray feedForward(Context context, NDArray residual, Ffn ffn) {
    NDArray normalized = context.backboneNorm(residual, ffn.norm(), epsilon, Stage.O_FFN_BACKWARD, layer);
    NDArray hidden = Context.silu(context.linear(normalized, ffn.up(), ffn.upBias()));
    return residual.add(context.linear(hidden, ffn.down(), ffn.downBias()));
  }

  /** 記録を使わずに Attention を計算する。GPU でも CPU でも djl-rocm の packed 演算を使う。 */
  private NDArray eager(Context context, NDArray query, NDArray keyValue, NDArray mask, NDArray codes) {
    return NDArrays.packedRelationScaledDotProductAttention(
        query, keyValue, mask, codes, relationTable(context), heads, RelationClass.ENTRIES, SCALE);
  }

  /** packed 演算に渡す表 [112, ヘッド + A]（FP32）。バイアスの列が先に並び、相対のバリューの列は 0。 */
  private NDArray relationTable(Context context) {
    NDArray bias = context.fp32(relation);
    NDArray values =
        context.working().zeros(new Shape(RelationClass.ROWS, attention), DataType.FLOAT32);
    return context.own(bias.concat(values, 1));
  }

  /** 第 2 軸の先頭 {@code count} 個を取り出す。 */
  private static NDArray prefix(NDArray array, int count) {
    return array.get(new NDIndex(":, 0:{}", count));
  }
}
