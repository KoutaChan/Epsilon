package com.epsilon.prism.luck;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.input.FieldSpec;
import com.epsilon.prism.input.InputSchema.BoundedField;
import com.epsilon.prism.input.TokenLayout;
import com.epsilon.prism.luck.LuckInputSchema.RoundField;
import com.epsilon.prism.luck.LuckInputSchema.SeatField;
import com.epsilon.prism.luck.LuckInputSchema.TileField;
import com.epsilon.prism.luck.LuckInputSchema.Token;
import com.epsilon.prism.model.Context;
import com.epsilon.prism.model.Module;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.WidthPlan.Axis;
import com.epsilon.prism.model.WidthPlan.Role;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.Map;

/**
 * 運の網 Ω のトークンの埋め込み（打つ網の {@code TokenEmbedding} と同じ形で、欄は {@link LuckInputSchema}）。型ごとに、埋め込む欄の小さな埋め込み
 * （局・席 16 次元、牌 8 次元）と 2 値の欄（埋め込まず 0/1 の数値にする）と数値の欄を並べ、型ごとの Linear で幅 H にする。Linear の偏りが型の
 * 埋め込みを兼ねる。
 *
 * <p>パラメーターは型ごとに {@code table?}（[表の行, d]）・{@code project?}（[H, 入力]）・{@code project?Bias}（[H]）で、? は G・P・T。
 */
final class LuckEmbedding extends Module {

  private final Map<Token, Parameter> tables = new EnumMap<>(Token.class);
  private final Map<Token, Parameter> projections = new EnumMap<>(Token.class);
  private final Map<Token, Parameter> biases = new EnumMap<>(Token.class);
  private final Map<Token, Columns> columns = new EnumMap<>(Token.class);

  /**
   * 型ごとのパラメーターを登録する。
   *
   * @param shape 網の形
   */
  LuckEmbedding(PrismShape shape) {
    int hidden = shape.hidden();
    for (Token token : Token.values()) {
      Columns split = Columns.of(token);
      int dimension = dimension(token);
      int input = split.embedded().length * dimension + split.flags().length + token.scalars();
      String suffix = suffix(token);
      columns.put(token, split);
      tables.put(token, table("table" + suffix, token.layout().tableSize(), dimension));
      projections.put(
          token,
          weight(
              "project" + suffix,
              new Shape(hidden, input),
              Axis.of(Role.STREAM_OUT),
              Axis.of(Role.IN)));
      biases.put(token, bias("project" + suffix + "Bias", hidden, Axis.of(Role.STREAM_OUT)));
    }
  }

  /**
   * 1 つの型のトークンを埋め込む。
   *
   * @param context 文脈
   * @param inputs 入力
   * @param token 型
   * @return [その型のトークンの数, H]（計算の型）
   */
  NDArray embed(Context context, LuckInputs inputs, Token token) {
    NDManager manager = context.working();
    Columns split = columns.get(token);
    NDArray values = inputs.category(token).toType(DataType.INT32, false);
    NDArray table = context.value(tables.get(token));
    NDArray ids = values.get(new NDIndex(":, {}", manager.create(split.embedded())));
    NDArray offsets = manager.create(split.offsets(), new Shape(1, split.offsets().length));
    NDArray flags =
        values
            .get(new NDIndex(":, {}", manager.create(split.flags())))
            .toType(table.getDataType(), false);
    NDArray scalar = inputs.scalar(token);
    NDArray dense =
        scalar == null ? flags : NDArrays.concat(new NDList(flags, context.toCompute(scalar)), 1);
    return context.linear(
        NDArrays.embeddingFeaturePack(ids, offsets, table, dense),
        projections.get(token),
        biases.get(token));
  }

  /** 埋め込みの次元（打つ網と同じ: 局・席 16、牌 8）。 */
  private static int dimension(Token token) {
    return switch (token) {
      case ROUND, SEAT -> 16;
      case TILE -> 8;
    };
  }

  /** パラメーターの名前の接尾辞。 */
  private static String suffix(Token token) {
    return switch (token) {
      case ROUND -> "G";
      case SEAT -> "P";
      case TILE -> "T";
    };
  }

  /**
   * 埋め込む列（とその名前空間の先頭）と 2 値の列。
   *
   * @param embedded 埋め込む列
   * @param offsets 埋め込む列の名前空間の先頭
   * @param flags 2 値の列
   */
  private record Columns(long[] embedded, int[] offsets, long[] flags) {

    static Columns of(Token token) {
      return switch (token) {
        case ROUND -> of(LuckInputSchema.ROUND, RoundField.class);
        case SEAT -> of(LuckInputSchema.SEAT, SeatField.class);
        case TILE -> of(LuckInputSchema.TILE, TileField.class);
      };
    }

    private static <E extends Enum<E> & BoundedField> Columns of(
        TokenLayout<E> layout, Class<E> type) {
      int[] namespaces = layout.columnOffsets();
      long[] embedded = new long[layout.width()];
      long[] flags = new long[layout.width()];
      int embeddedCount = 0;
      int flagCount = 0;
      for (E field : type.getEnumConstants()) {
        FieldSpec spec = field.spec();
        for (int repeat = 0; repeat < spec.repeats(); repeat++) {
          int column = layout.column(field, repeat);
          if (spec.coding() == FieldSpec.Coding.FLAG) {
            flags[flagCount++] = column;
          } else {
            embedded[embeddedCount++] = column;
          }
        }
      }
      if (embeddedCount == 0 || flagCount == 0) {
        throw new IllegalStateException(type.getSimpleName() + " needs embedded and flag columns");
      }
      int[] offsets = new int[embeddedCount];
      for (int index = 0; index < embeddedCount; index++) {
        offsets[index] = namespaces[(int) embedded[index]];
      }
      return new Columns(
          Arrays.copyOf(embedded, embeddedCount), offsets, Arrays.copyOf(flags, flagCount));
    }
  }
}
