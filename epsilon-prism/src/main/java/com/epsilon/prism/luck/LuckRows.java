package com.epsilon.prism.luck;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.core.GameState;
import com.epsilon.prism.luck.LuckInputSchema.Token;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

/**
 * 運の網 Ω の 1 塊のホストの配列: 入力（{@link LuckTokenizer} が書く）と、学習と採点に使う目標（結果・プール・Y・重み）。符号化のスレッドが行ごと
 * に書き（行が違えば並行に書いてよい）、学習のスレッドが {@link #inputs} と {@link #targets} でデバイスへ送る。使い回す。
 */
final class LuckRows {

  static final int OUTCOMES = LuckSchema.OUTCOMES;

  private static final int SEATS = GameState.NUM_PLAYERS;

  final int capacity;

  /** 局の欄 [行, 列]。 */
  final byte[] round;

  /** 局の数値 [行, 数値]。 */
  final float[] roundScalar;

  /** 席の欄 [行, 4, 列]。 */
  final byte[] seat;

  /** 席の数値 [行, 4, 数値]。 */
  final float[] seatScalar;

  /** 牌の欄 [行, 34, 列]。 */
  final byte[] tile;

  /** 関係の符号 [行, 39, 39, 2] の INT16（ネイティブのバイト順）。 */
  final ByteBuffer codes;

  /** 正準の結果の順のプールの枚数 [行, 37]（入力ではない。p(t) = 枚数 / N）。 */
  final float[] pool;

  /** 実際の結果の平らな番号 [行]（行 × 37 + 正準の結果）。 */
  final long[] chosen;

  /** 局末の Φ [行, 4]（行為者から見た相対席の順）。 */
  final float[] y;

  /** 重み [行]。 */
  final float[] weight;

  /** 出来事の種類の ordinal [行]（報告だけ）。 */
  final int[] event;

  /** 半荘 ID [行]（報告の束ねだけ）。 */
  final long[] game;

  private int rows;

  /**
   * 塊を作る。
   *
   * @param capacity 行の上限
   */
  LuckRows(int capacity) {
    if (capacity < 1) {
      throw new IllegalArgumentException("capacity must be positive: " + capacity);
    }
    this.capacity = capacity;
    round = new byte[capacity * LuckInputSchema.ROUND.width()];
    roundScalar = new float[capacity * Token.ROUND.scalars()];
    seat = new byte[capacity * SEATS * LuckInputSchema.SEAT.width()];
    seatScalar = new float[capacity * SEATS * Token.SEAT.scalars()];
    tile = new byte[capacity * Token.TILE.perRow() * LuckInputSchema.TILE.width()];
    codes =
        ByteBuffer.allocate(Math.multiplyExact(capacity, LuckRelation.ROW_BYTES))
            .order(ByteOrder.nativeOrder());
    pool = new float[capacity * OUTCOMES];
    chosen = new long[capacity];
    y = new float[capacity * SEATS];
    weight = new float[capacity];
    event = new int[capacity];
    game = new long[capacity];
  }

  int rows() {
    return rows;
  }

  /**
   * 行の数を決める（書く前に呼ぶ）。
   *
   * @param count 行の数
   */
  void resize(int count) {
    if (count < 1 || count > capacity) {
      throw new IllegalArgumentException("rows must be in 1.." + capacity + ": " + count);
    }
    rows = count;
  }

  /**
   * 標本を 1 行に書く。
   *
   * @param row 行
   * @param tokenizer 符号化（このスレッドのもの）
   * @param o 標本の O°（付け替えた読み手）
   * @param phi 局末の Φ を置いた配列（行為者から見た相対席の順に 4 席）
   * @param from Φ の先頭
   * @param sampleWeight 重み
   * @param sampleGame 半荘 ID
   */
  void put(
      int row,
      LuckTokenizer tokenizer,
      LuckObservation o,
      float[] phi,
      int from,
      float sampleWeight,
      long sampleGame) {
    tokenizer.write(o, this, row);
    chosen[row] = (long) row * OUTCOMES + tokenizer.canonicalOutcome(o.outcome());
    System.arraycopy(phi, from, y, row * SEATS, SEATS);
    weight[row] = sampleWeight;
    event[row] = o.event().ordinal();
    game[row] = sampleGame;
  }

  /** 行の入力を 0 にする（書く前に符号化が呼ぶ）。 */
  void clear(int row) {
    clearSlice(round, row, LuckInputSchema.ROUND.width());
    clearSlice(seat, row, SEATS * LuckInputSchema.SEAT.width());
    clearSlice(tile, row, Token.TILE.perRow() * LuckInputSchema.TILE.width());
  }

  /**
   * 入力をデバイスへ送る。
   *
   * @param manager 配列を作る管理元
   * @return 入力
   */
  LuckInputs inputs(NDManager manager) {
    int seats = rows * SEATS;
    int tiles = rows * Token.TILE.perRow();
    return new LuckInputs(
        bytes(manager, round, new Shape(rows, LuckInputSchema.ROUND.width())),
        floats(manager, roundScalar, new Shape(rows, Token.ROUND.scalars())),
        bytes(manager, seat, new Shape(seats, LuckInputSchema.SEAT.width())),
        floats(manager, seatScalar, new Shape(seats, Token.SEAT.scalars())),
        bytes(manager, tile, new Shape(tiles, LuckInputSchema.TILE.width())),
        manager.create(
            ByteBuffer.wrap(codes.array(), 0, rows * LuckRelation.ROW_BYTES),
            new Shape(rows, Token.ROW_TOKENS, Token.ROW_TOKENS, LuckRelation.WORDS),
            DataType.INT16),
        rows);
  }

  /**
   * 目標をデバイスへ送る。
   *
   * @param manager 配列を作る管理元
   * @return 目標
   */
  LuckTargets targets(NDManager manager) {
    return new LuckTargets(
        floats(manager, pool, new Shape(rows, OUTCOMES)),
        manager.create(Arrays.copyOf(chosen, rows)),
        floats(manager, y, new Shape(rows, SEATS)),
        floats(manager, weight, new Shape(rows)));
  }

  private void clearSlice(byte[] array, int row, int width) {
    Arrays.fill(array, row * width, (row + 1) * width, (byte) 0);
  }

  private static NDArray bytes(NDManager manager, byte[] array, Shape shape) {
    return manager.create(
        ByteBuffer.wrap(array, 0, Math.toIntExact(shape.size())), shape, DataType.UINT8);
  }

  private static NDArray floats(NDManager manager, float[] array, Shape shape) {
    return manager.create(
        FloatBuffer.wrap(array, 0, Math.toIntExact(shape.size())), shape, DataType.FLOAT32);
  }
}
