package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.FitProbe.Stage;
import java.util.EnumMap;
import java.util.Map;

/**
 * トークンの埋め込み（HIP の K1 に当たる）。型ごとに、フィールドの小さな埋め込み（局・席は 16 次元、牌・イベントは 8 次元）と、
 * 2 値のフィールドと、数値を {@link FieldPack} で並べ、型ごとの Linear で幅 H にする。Linear のバイアスが型の埋め込みも兼ねる。
 *
 * <p>パラメーターは型ごとに {@code table?}（[表の行, d]）・{@code project?}（[H, 入力]）・{@code project?Bias}（[H]）で、? は
 * {@link TokenType#suffix()}。
 */
final class TokenEmbedding extends Module {

  private final Map<TokenType, Parameter> tables = new EnumMap<>(TokenType.class);
  private final Map<TokenType, Parameter> projections = new EnumMap<>(TokenType.class);
  private final Map<TokenType, Parameter> biases = new EnumMap<>(TokenType.class);
  private final InputFormat format;

  /**
   * 型ごとのパラメーターを登録する。
   *
   * @param shape ネットワークの形
   * @param format 入力形式
   */
  TokenEmbedding(PrismShape shape, InputFormat format) {
    this.format = format;
    int hidden = shape.hidden();
    for (TokenType type : TokenType.values()) {
      String suffix = type.suffix();
      tables.put(type, table("table" + suffix, type.pack(format).tableRows(), type.dimension()));
      projections.put(type, weight("project" + suffix, new Shape(hidden, type.inputWidth(format))));
      biases.put(type, bias("project" + suffix + "Bias", hidden));
    }
  }

  /**
   * 1 つの型のトークンを埋め込む。
   *
   * @param context 文脈
   * @param inputs 入力
   * @param index インデックス（列の位置を引く）
   * @param type トークンの型
   * @return 埋め込み [その型のトークンの数, H]（計算に使う型）
   */
  NDArray embed(Context context, NetworkInputs inputs, BatchIndex index, TokenType type) {
    NDArray category = inputs.get(type.category());
    NDArray scalar = type.scalar() == null ? null : inputs.get(type.scalar());
    return context.scoped(scope -> embed(scope, index, type, category, scalar), category, scalar);
  }

  private NDArray embed(
      Context context, BatchIndex index, TokenType type, NDArray category, NDArray scalar) {
    NDArray table = context.value(tables.get(type));
    FieldPack pack = type.pack(format);
    FieldPack.Columns columns = index.columns(pack);
    NDArray packed =
        scalar == null
            ? pack.pack(category, columns, table)
            : pack.pack(category, columns, table, context.toCompute(scalar));
    // 計測のときは並べたフィールドを葉にして、表の勾配への足し込みを Linear の逆伝播と分けて計る（Context#probeCut）
    return context.linear(
        context.probeCut(packed, Stage.EMBED_PACK_BACKWARD, 0),
        projections.get(type),
        biases.get(type));
  }
}
