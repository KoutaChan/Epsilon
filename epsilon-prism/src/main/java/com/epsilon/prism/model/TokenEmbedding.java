package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.model.FitProbe.Stage;
import com.epsilon.prism.model.WidthPlan.Axis;
import com.epsilon.prism.model.WidthPlan.Role;
import java.util.EnumMap;
import java.util.Map;

/**
 * トークンの埋め込み（HIP の K1 に当たる）。型ごとに、欄の小さな埋め込み（局・席 16 次元、牌・出来事 8 次元）と 2 値の欄と
 * 数値を {@link FieldPack} で並べ、型ごとの Linear で幅 H にする。Linear の偏りが型の埋め込みを兼ねる。
 *
 * <p>パラメーターは型ごとに {@code table?}（[表の行, d]）・{@code project?}（[H, 入力]）・{@code project?Bias}（[H]）で、? は
 * {@link TokenType#suffix()}。
 */
final class TokenEmbedding extends Module {

  private final Map<TokenType, Parameter> tables = new EnumMap<>(TokenType.class);
  private final Map<TokenType, Parameter> projections = new EnumMap<>(TokenType.class);
  private final Map<TokenType, Parameter> biases = new EnumMap<>(TokenType.class);

  /**
   * 型ごとのパラメーターを登録する。
   *
   * @param shape 網の形
   */
  TokenEmbedding(PrismShape shape) {
    int hidden = shape.hidden();
    for (TokenType type : TokenType.values()) {
      String suffix = type.suffix();
      tables.put(type, table("table" + suffix, type.pack().tableRows(), type.dimension()));
      projections.put(
          type,
          weight(
              "project" + suffix,
              new Shape(hidden, type.inputWidth()),
              Axis.of(Role.STREAM_OUT),
              Axis.of(Role.IN)));
      biases.put(type, bias("project" + suffix + "Bias", hidden, Axis.of(Role.STREAM_OUT)));
    }
  }

  /**
   * 1 つの型のトークンを埋め込む。
   *
   * @param context 文脈
   * @param inputs 入力
   * @param index 索引（列の添字）
   * @param type 型
   * @return [その型のトークンの数, H]（計算の型）
   */
  NDArray embed(Context context, NetworkInputs inputs, BatchIndex index, TokenType type) {
    NDArray category = inputs.get(type.category());
    NDArray table = context.value(tables.get(type));
    FieldPack.Columns columns = index.columns(type.pack());
    NDArray packed =
        type.scalar() == null
            ? type.pack().pack(category, columns, table)
            : type.pack()
                .pack(category, columns, table, context.toCompute(inputs.get(type.scalar())));
    // 計測のときは欄の並べを葉にして、表の勾配への足し込みを Linear の逆伝播と分ける（Context#probeCut）
    return context.linear(
        context.probeCut(packed, Stage.EMBED_PACK_BACKWARD, 0),
        projections.get(type),
        biases.get(type));
  }
}
