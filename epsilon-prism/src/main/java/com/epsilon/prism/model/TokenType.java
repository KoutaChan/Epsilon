package com.epsilon.prism.model;

import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;

/**
 * バックボーンのトークンの型。行の中のトークンは宣言の順に並ぶ。局が 1 個、席が S 個（相対席の順）、牌が T 個（正準の色の順）で、
 * その後にイベントが公開された順に続く。S と T は入力の形式が決める。局・席・牌のトークンは全ての行にあり、候補の層のキー
 * （{@link InputFormat.Keys} の順）と同じ順に並ぶ。
 */
enum TokenType {
  /** 局（行に 1 個）。埋め込みは 16 次元。 */
  ROUND(Tensor.ROUND_CATEGORY, Tensor.ROUND_SCALAR, 16),
  /** 席（行に S 個）。埋め込みは 16 次元。 */
  SEAT(Tensor.SEAT_CATEGORY, Tensor.SEAT_SCALAR, 16),
  /** 牌（行に T 個）。埋め込みは 8 次元。 */
  TILE(Tensor.TILE_CATEGORY, null, 8),
  /** イベント（行に 0 個以上）。埋め込みは 8 次元。 */
  EVENT(Tensor.EVENT_CATEGORY, Tensor.EVENT_SCALAR, 8);

  private static final TokenType[] VALUES = values();

  private final Tensor category;
  private final Tensor scalar;
  private final int dimension;

  TokenType(Tensor category, Tensor scalar, int dimension) {
    this.category = category;
    this.scalar = scalar;
    this.dimension = dimension;
  }

  Tensor category() {
    return category;
  }

  /** 数値の入力が入る領域（牌には無いので null）。 */
  Tensor scalar() {
    return scalar;
  }

  int dimension() {
    return dimension;
  }

  /** この型のフィールドの並べ方。 */
  FieldPack pack(InputFormat format) {
    return FieldPack.of(format).of(this);
  }

  /** 1 行あたりの決まった個数（イベントは数が決まらないので 0）。 */
  int perRow(InputFormat format) {
    return switch (this) {
      case ROUND -> 1;
      case SEAT -> format.seats();
      case TILE -> format.tiles();
      case EVENT -> 0;
    };
  }

  /** 行の中で、この型の最初のトークンがある位置。 */
  int first(InputFormat format) {
    return switch (this) {
      case ROUND -> 0;
      case SEAT -> 1;
      case TILE -> 1 + format.seats();
      case EVENT -> format.fixedTokens();
    };
  }

  /** 型ごとの Linear の入力の幅。 */
  int inputWidth(InputFormat format) {
    return pack(format).inputWidth(dimension, scalar == null ? 0 : format.width(scalar));
  }

  /** パラメーターの名前の接尾辞。 */
  String suffix() {
    return switch (this) {
      case ROUND -> "G";
      case SEAT -> "P";
      case TILE -> "T";
      case EVENT -> "E";
    };
  }

  /**
   * 行の中のある位置にあるトークンの型を返す。
   *
   * @param format 入力の形式
   * @param position 行の中の位置
   * @return その位置のトークンの型
   */
  static TokenType at(InputFormat format, int position) {
    for (int type = VALUES.length - 1; type > 0; type--) {
      if (position >= VALUES[type].first(format)) {
        return VALUES[type];
      }
    }
    return ROUND;
  }
}
