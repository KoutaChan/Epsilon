package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;
import com.epsilon.prism.input.InputFormat;
import java.util.EnumMap;
import java.util.Locale;
import java.util.Map;

/**
 * 相手の手を読む Oracle の補助のヘッド。学習のときだけ計算し、推論で使う重みの束には入れない。入力は、相手の席（下家から上家までの
 * S − 1 席）の y_P と、牌の y_T。
 *
 * <ul>
 *   <li>相手ごとのヘッド（聴牌・向聴）: y_P の Linear。
 *   <li>相手 × 牌のヘッド（当たり牌か・待ち・ロンされたときの点）: 64 次元の双線形 ⟨A·y_P, B·y_T⟩/8 + b。
 * </ul>
 *
 * <p>出力は [行, S − 1, {@link TargetSchema#readingWidth}]（FP32）で、最後の軸には {@link ReadingTarget} の順に各ヘッドの列を並べる。
 */
final class ReadingHeads extends Module {

  /** 双線形の次元。 */
  static final int BILINEAR = 64;

  /** 双線形の値に掛ける係数（1/√64）。 */
  private static final float SCALE = (float) (1.0 / Math.sqrt(BILINEAR));

  /** ヘッドの形。 */
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
  private final int opponentFirst;
  private final int opponents;
  private final int tileFirst;
  private final int tileCount;

  /**
   * パラメーターを登録する。
   *
   * @param shape ネットワークの形
   * @param format 入力形式（相手の席と牌のトークンの位置を決める）
   */
  ReadingHeads(PrismShape shape, InputFormat format) {
    TargetSchema heads = TargetSchema.of(format.variant());
    opponentFirst = TokenType.SEAT.first(format) + 1;
    opponents = heads.opponents();
    tileFirst = TokenType.TILE.first(format);
    tileCount = format.tiles();
    int hidden = shape.hidden();
    for (ReadingTarget target : ReadingTarget.values()) {
      String name = camel(target);
      switch (Form.of(target)) {
        case LINEAR -> {
          seats.put(target, weight(name, new Shape(heads.width(target), hidden)));
          biases.put(target, bias(name + "Bias", heads.width(target)));
        }
        case BILINEAR -> {
          seats.put(target, weight(name + "Seat", new Shape(BILINEAR, hidden)));
          tiles.put(target, weight(name + "Tile", new Shape(BILINEAR, hidden)));
          biases.put(target, bias(name + "Bias", 1));
        }
      }
    }
  }

  /**
   * 読みのヘッドの出力を計算する。
   *
   * @param context 文脈
   * @param states 読みの行の、局・席・牌のトークンの状態 [行, 局・席・牌, H]（計算に使う型）
   * @return 出力 [行, S − 1, {@code TargetSchema.readingWidth}]（FP32）
   */
  NDArray forward(Context context, NDArray states) {
    NDArray opponentStates =
        states.get(new NDIndex(":, {}:{}", opponentFirst, opponentFirst + opponents));
    NDArray tileStates = states.get(new NDIndex(":, {}:{}", tileFirst, tileFirst + tileCount));
    NDList parts = new NDList(ReadingTarget.values().length);
    for (ReadingTarget target : ReadingTarget.values()) {
      NDArray bias = context.fp32(biases.get(target));
      NDArray part =
          switch (Form.of(target)) {
            case LINEAR -> context.toFloat(context.linear(opponentStates, seats.get(target), null)).add(bias);
            case BILINEAR -> {
              NDArray seat = context.toFloat(context.linear(opponentStates, seats.get(target), null));
              NDArray tile = context.toFloat(context.linear(tileStates, tiles.get(target), null));
              yield seat.matMul(tile.transpose(0, 2, 1)).mul(SCALE).add(bias);
            }
          };
      parts.add(part);
    }
    return NDArrays.concat(parts, 2);
  }

  /** ヘッドの名前（DANGER_POINTS → dangerPoints）。 */
  private static String camel(ReadingTarget target) {
    StringBuilder name = new StringBuilder();
    for (String word : target.name().toLowerCase(Locale.ROOT).split("_")) {
      name.append(name.isEmpty() ? word : Character.toUpperCase(word.charAt(0)) + word.substring(1));
    }
    return name.toString();
  }
}
