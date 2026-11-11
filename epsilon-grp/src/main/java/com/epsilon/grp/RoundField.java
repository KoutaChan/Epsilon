package com.epsilon.grp;

import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.function.ToIntFunction;

/**
 * 局が始まったときの卓を 1 行に書くときのフィールド。その行が RoundHistory の 1 行になり、GruSinkhorn の 1 ステップの入力にもなる。
 * 定数ごとに幅とスケールを持ち、位置は宣言の順に幅を足して一度だけ決める。行の値は整数で、GruSinkhorn には値をスケールで割って
 * 渡す（int を float にしてから float で割る）。
 */
enum RoundField {
  /** 局の番号（東 1 局が 0）。 */
  ROUND(variant -> 1, 8f),
  HONBA(variant -> 1, 8f),
  /** 供託の本数。 */
  KYOTAKU(variant -> 1, 4f),
  /** 起家を 0 とする絶対席ごとの点。 */
  SCORE(Variant::seats, 10_000f);

  private static final RoundField[] FIELDS = values();

  /** 行の中での開始位置。添字は [ルール][フィールド]（どちらも宣言の順）で、[ルール][フィールドの数] は行の幅。 */
  private static final int[][] STARTS = starts();

  private final ToIntFunction<Variant> width;
  private final float scale;

  RoundField(ToIntFunction<Variant> width, float scale) {
    this.width = width;
    this.scale = scale;
  }

  /** 卓の型ごとに点を読む関数。何もキャプチャしないメソッド参照で渡すので、呼び出しごとにメモリを確保しない。 */
  @FunctionalInterface
  private interface Scores<T> {
    int score(T table, int seat);
  }

  int offset(Variant variant) {
    return STARTS[variant.ordinal()][ordinal()];
  }

  /** 行の幅（四人打ちで 7、三人打ちで 6）。 */
  static int width(Variant variant) {
    return STARTS[variant.ordinal()][FIELDS.length];
  }

  /** 行の列ごとのスケール。 */
  static float[] scales(Variant variant) {
    float[] scales = new float[width(variant)];
    for (RoundField field : FIELDS) {
      int base = field.offset(variant);
      for (int column = 0; column < field.width.applyAsInt(variant); column++) {
        scales[base + column] = field.scale;
      }
    }
    return scales;
  }

  /** 局が始まったときの卓を、指定した位置から 1 行書く。 */
  static void write(TableState table, int[] out, int at) {
    write(table.variant(), table.round(), table.honba(), table.kyotaku(), table, TableState::score, out, at);
  }

  /** 局が始まったときの卓を、指定した位置から 1 行書く。 */
  static void write(GrpState state, int[] out, int at) {
    write(state.variant(), state.round(), state.honba(), state.sticks(), state, GrpState::score, out, at);
  }

  private static <T> void write(
      Variant variant, int round, int honba, int kyotaku, T table, Scores<T> scores, int[] out, int at) {
    for (RoundField field : FIELDS) {
      int base = at + field.offset(variant);
      switch (field) {
        case ROUND -> out[base] = round;
        case HONBA -> out[base] = honba;
        case KYOTAKU -> out[base] = kyotaku;
        case SCORE -> {
          for (int seat = 0; seat < variant.seats(); seat++) {
            out[base + seat] = scores.score(table, seat);
          }
        }
      }
    }
  }

  private static int[][] starts() {
    Variant[] variants = Variant.values();
    int[][] starts = new int[variants.length][FIELDS.length + 1];
    for (Variant variant : variants) {
      int[] row = starts[variant.ordinal()];
      for (RoundField field : FIELDS) {
        row[field.ordinal() + 1] = row[field.ordinal()] + field.width.applyAsInt(variant);
      }
    }
    return starts;
  }
}
