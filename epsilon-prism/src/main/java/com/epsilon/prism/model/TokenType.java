package com.epsilon.prism.model;

import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.prism.input.InputSchema.KeyBlock;
import com.epsilon.prism.input.InputSchema.Tensor;

/**
 * 背骨のトークンの型。行の中の並びは宣言の順で、局 1・席 4（相対席の順）・牌 34（正準の色の番号の順）の後に出来事が公開の順に続く。
 * 局・席・牌の 39 個は全ての行にあり、候補層の鍵（{@link KeyBlock} の順）と同じ並び。
 */
enum TokenType {
  /** 局（行に 1 個）。埋め込みは 16 次元。 */
  ROUND(FieldPack.ROUND, Tensor.ROUND_CATEGORY, Tensor.ROUND_SCALAR, 16, 1),
  /** 席（行に 4 個）。埋め込みは 16 次元。 */
  SEAT(FieldPack.SEAT, Tensor.SEAT_CATEGORY, Tensor.SEAT_SCALAR, 16, GameState.NUM_PLAYERS),
  /** 牌（行に 34 個）。埋め込みは 8 次元。 */
  TILE(FieldPack.TILE, Tensor.TILE_CATEGORY, null, 8, Tile.NUM_TILE_TYPES),
  /** 出来事（行に 0 個以上）。埋め込みは 8 次元。 */
  EVENT(FieldPack.EVENT, Tensor.EVENT_CATEGORY, Tensor.EVENT_SCALAR, 8, 0);

  /** 全ての行にあるトークン（局・席・牌）の数。 */
  static final int FIXED_TOKENS;

  private static final int[] FIRST;

  static {
    TokenType[] types = values();
    FIRST = new int[types.length];
    int position = 0;
    for (TokenType type : types) {
      FIRST[type.ordinal()] = position;
      position += type.perRow;
    }
    FIXED_TOKENS = position;
    if (FIXED_TOKENS != KeyBlock.KEY_COUNT) {
      throw new IllegalStateException("fixed tokens must match the candidate keys");
    }
  }

  private final FieldPack pack;
  private final Tensor category;
  private final Tensor scalar;
  private final int dimension;
  private final int perRow;

  TokenType(FieldPack pack, Tensor category, Tensor scalar, int dimension, int perRow) {
    this.pack = pack;
    this.category = category;
    this.scalar = scalar;
    this.dimension = dimension;
    this.perRow = perRow;
  }

  FieldPack pack() {
    return pack;
  }

  Tensor category() {
    return category;
  }

  /** 数値の区画（牌は null）。 */
  Tensor scalar() {
    return scalar;
  }

  int dimension() {
    return dimension;
  }

  /** 行あたりの個数（出来事は 0）。 */
  int perRow() {
    return perRow;
  }

  /** 行の中の最初の位置。 */
  int first() {
    return FIRST[ordinal()];
  }

  /** 型ごとの Linear の入力の幅。 */
  int inputWidth() {
    return pack.inputWidth(dimension, scalar == null ? 0 : scalar.width());
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
   * 行の中の位置の型を返す。
   *
   * @param position 位置
   * @return 型
   */
  static TokenType at(int position) {
    if (position < SEAT.first()) {
      return ROUND;
    }
    if (position < TILE.first()) {
      return SEAT;
    }
    return position < EVENT.first() ? TILE : EVENT;
  }
}
