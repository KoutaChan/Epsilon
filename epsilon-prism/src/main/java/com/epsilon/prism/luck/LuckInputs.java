package com.epsilon.prism.luck;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import com.epsilon.prism.luck.LuckInputSchema.Token;
import java.util.Objects;

/**
 * 運の網 Ω の入力の 1 塊（デバイスの上、同じ管理元）。行ごとに局 1・席 4・牌 34 のトークンが連続に並ぶ。
 *
 * @param round 局の欄 [行, 局の列]（UINT8）
 * @param roundScalar 局の数値 [行, 局の数値]（FP32、scale で割った値）
 * @param seat 席の欄 [行 × 4, 席の列]（UINT8）
 * @param seatScalar 席の数値 [行 × 4, 席の数値]（FP32）
 * @param tile 牌の欄 [行 × 34, 牌の列]（UINT8）
 * @param codes 関係の符号 [行, 39, 39, 2]（INT16、{@link LuckRelation}）
 * @param rows 行の数
 */
public record LuckInputs(
    NDArray round,
    NDArray roundScalar,
    NDArray seat,
    NDArray seatScalar,
    NDArray tile,
    NDArray codes,
    int rows) {

  /**
   * 形を確かめる。
   *
   * @throws IllegalArgumentException 形が行の数と合わないとき
   */
  public LuckInputs {
    require(round, new Shape(rows, LuckInputSchema.ROUND.width()));
    require(roundScalar, new Shape(rows, Token.ROUND.scalars()));
    require(seat, new Shape((long) rows * Token.SEAT.perRow(), LuckInputSchema.SEAT.width()));
    require(seatScalar, new Shape((long) rows * Token.SEAT.perRow(), Token.SEAT.scalars()));
    require(tile, new Shape((long) rows * Token.TILE.perRow(), LuckInputSchema.TILE.width()));
    require(
        codes, new Shape(rows, Token.ROW_TOKENS, Token.ROW_TOKENS, LuckRelation.WORDS));
  }

  /**
   * 型の欄を返す。
   *
   * @param token 型
   * @return 欄 [その型のトークンの数, 列]
   */
  public NDArray category(Token token) {
    return switch (token) {
      case ROUND -> round;
      case SEAT -> seat;
      case TILE -> tile;
    };
  }

  /**
   * 型の数値を返す。
   *
   * @param token 型
   * @return 数値 [その型のトークンの数, 数値]（数値のない型は null）
   */
  public NDArray scalar(Token token) {
    return switch (token) {
      case ROUND -> roundScalar;
      case SEAT -> seatScalar;
      case TILE -> null;
    };
  }

  /** 配列の管理元（作業の管理元にする）。 */
  NDManager manager() {
    return codes.getManager();
  }

  private static void require(NDArray array, Shape expected) {
    Objects.requireNonNull(array, "array");
    if (!array.getShape().equals(expected)) {
      throw new IllegalArgumentException(
          "luck input has shape " + array.getShape() + ", expected " + expected);
    }
  }
}
