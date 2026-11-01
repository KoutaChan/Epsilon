package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;
import com.epsilon.prism.model.WidthPlan.Axis;
import com.epsilon.prism.model.WidthPlan.Role;
import java.util.EnumMap;
import java.util.Locale;
import java.util.Map;

/**
 * 神託の読みの頭（学習のときだけ計算し、推論の重みの束に入れない）。入力は相手 3 席（下家・対面・上家）の y_P と牌の y_T。
 *
 * <ul>
 *   <li>相手ごとの頭（TENPAI・SHANTEN）: y_P の Linear。
 *   <li>相手 × 牌の頭（DANGER・WAITS・DANGER_POINTS）: 64 次元の双線形 ⟨A·y_P, B·y_T⟩/8 + b。
 * </ul>
 *
 * <p>出力は {@link ReadingTarget} の順に区画を並べた [行, 3, {@link ReadingTarget#WIDTH}]（FP32）。
 */
final class ReadingHeads extends Module {

  /** 双線形の次元。 */
  static final int BILINEAR = 64;

  /** 双線形の尺度（1/√64）。 */
  private static final float SCALE = (float) (1.0 / Math.sqrt(BILINEAR));

  private static final int OPPONENT_FIRST = TokenType.SEAT.first() + 1;
  private static final int TILE_FIRST = TokenType.TILE.first();

  /** 頭の形。 */
  private enum Form {
    /** 相手ごとの Linear。 */
    LINEAR,
    /** 相手 × 牌の双線形。 */
    BILINEAR;

    static Form of(ReadingTarget target) {
      return switch (target) {
        case TENPAI, SHANTEN -> LINEAR;
        case DANGER, WAITS, DANGER_POINTS -> BILINEAR;
      };
    }
  }

  private final Map<ReadingTarget, Parameter> seats = new EnumMap<>(ReadingTarget.class);
  private final Map<ReadingTarget, Parameter> tiles = new EnumMap<>(ReadingTarget.class);
  private final Map<ReadingTarget, Parameter> biases = new EnumMap<>(ReadingTarget.class);

  /**
   * パラメーターを登録する。
   *
   * @param shape 網の形
   */
  ReadingHeads(PrismShape shape) {
    int hidden = shape.hidden();
    for (ReadingTarget target : ReadingTarget.values()) {
      String name = camel(target);
      switch (Form.of(target)) {
        case LINEAR -> {
          seats.put(
              target,
              weight(name, new Shape(target.width(), hidden), Axis.of(Role.FIXED), Axis.of(Role.STREAM_IN)));
          biases.put(target, bias(name + "Bias", target.width(), Axis.of(Role.FIXED)));
        }
        case BILINEAR -> {
          seats.put(
              target,
              weight(name + "Seat", new Shape(BILINEAR, hidden), Axis.of(Role.FIXED), Axis.of(Role.STREAM_IN)));
          tiles.put(
              target,
              weight(name + "Tile", new Shape(BILINEAR, hidden), Axis.of(Role.FIXED), Axis.of(Role.STREAM_IN)));
          biases.put(target, bias(name + "Bias", 1, Axis.of(Role.FIXED)));
        }
      }
    }
  }

  /**
   * 読みの頭の出力を計算する。
   *
   * @param context 文脈
   * @param states 読みの行の局・席・牌 [行, 39, H]（計算の型）
   * @return 出力 [行, 3, ReadingTarget.WIDTH]（FP32）
   */
  NDArray forward(Context context, NDArray states) {
    NDArray opponents =
        states.get(new NDIndex(":, {}:{}", OPPONENT_FIRST, OPPONENT_FIRST + AuxiliaryLabels.OPPONENTS));
    NDArray tileStates =
        states.get(new NDIndex(":, {}:{}", TILE_FIRST, TILE_FIRST + TokenType.TILE.perRow()));
    NDList parts = new NDList(ReadingTarget.values().length);
    for (ReadingTarget target : ReadingTarget.values()) {
      NDArray bias = context.fp32(biases.get(target));
      NDArray part =
          switch (Form.of(target)) {
            case LINEAR -> context.toFloat(context.linear(opponents, seats.get(target), null)).add(bias);
            case BILINEAR -> {
              NDArray seat = context.toFloat(context.linear(opponents, seats.get(target), null));
              NDArray tile = context.toFloat(context.linear(tileStates, tiles.get(target), null));
              yield seat.matMul(tile.transpose(0, 2, 1)).mul(SCALE).add(bias);
            }
          };
      parts.add(part);
    }
    return NDArrays.concat(parts, 2);
  }

  /** 頭の名前（DANGER_POINTS → dangerPoints）。 */
  private static String camel(ReadingTarget target) {
    StringBuilder name = new StringBuilder();
    for (String word : target.name().toLowerCase(Locale.ROOT).split("_")) {
      name.append(name.isEmpty() ? word : Character.toUpperCase(word.charAt(0)) + word.substring(1));
    }
    return name.toString();
  }
}
