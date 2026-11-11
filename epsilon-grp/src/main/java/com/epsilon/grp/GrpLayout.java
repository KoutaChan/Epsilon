package com.epsilon.grp;

import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;

/**
 * GRP の入力でのフィールドの並び方。使うフィールドだけを宣言の順に並べ、各フィールドの開始位置は作るときに 1 回だけ求める。
 * ルールとフィールドの組ごとに作り、どのスレッドからも読める。入力を書くのは、スレッドごとに持つ GrpEncoder。
 */
public final class GrpLayout {

  /** 点のフィールドの単位。 */
  private static final float POINTS = 10_000f;

  private static final int ABSENT = -1;

  private final Variant variant;
  private final Set<GrpField> fields;

  /** 使うフィールドを宣言の順に並べたもの。イテレータを作らずに回すために持つ。 */
  private final GrpField[] present;

  /** フィールドごとの開始位置（宣言の順）。使わないフィールドは {@code ABSENT}。 */
  private final int[] offsets;

  private final int width;

  private GrpLayout(Variant variant, Set<GrpField> fields) {
    this.variant = variant;
    this.fields = fields;
    present = fields.toArray(GrpField[]::new);
    offsets = new int[GrpField.values().length];
    Arrays.fill(offsets, ABSENT);
    int next = 0;
    for (GrpField field : present) {
      offsets[field.ordinal()] = next;
      next += field.width(variant);
    }
    width = next;
  }

  /** フィールドが 1 つも無いときは IllegalArgumentException を投げる。 */
  public static GrpLayout of(Variant variant, Set<GrpField> fields) {
    Objects.requireNonNull(variant, "variant");
    EnumSet<GrpField> copy = EnumSet.noneOf(GrpField.class);
    copy.addAll(fields);
    if (copy.isEmpty()) {
      throw new IllegalArgumentException("A GRP layout needs at least one field");
    }
    return new GrpLayout(variant, Collections.unmodifiableSet(copy));
  }

  public Variant variant() {
    return variant;
  }

  /** 使うフィールドを返す。変更できない集合で、回す順は宣言の順。 */
  public Set<GrpField> fields() {
    return fields;
  }

  public boolean has(GrpField field) {
    return offsets[field.ordinal()] != ABSENT;
  }

  /** フィールドの開始位置を返す。使わないフィールドでは IllegalArgumentException を投げる。 */
  public int offset(GrpField field) {
    if (!has(field)) {
      throw new IllegalArgumentException(field + " is not in " + this);
    }
    return offsets[field.ordinal()];
  }

  public int width() {
    return width;
  }

  /**
   * 卓を入力に書く。呼ぶたびに作業領域を作るので、繰り返し書くなら GrpEncoder を使う。
   *
   * @param out 書き先。先頭から入力の幅の分だけ書く
   */
  public void encode(GrpState state, float[] out) {
    GrpEncoder.of(this).encode(state, out);
  }

  /**
   * 半荘が続く卓を入力に書く。
   *
   * @param out 書き先。先頭から入力の幅の分だけ書く
   */
  void write(TableState table, float[] out) {
    if (table.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: layout " + variant + ", table " + table.variant());
    }
    if (table.gameOver()) {
      throw new IllegalArgumentException("A finished game has no next hand: " + table);
    }
    if (out.length < width) {
      throw new IllegalArgumentException("Input buffer " + out.length + " is shorter than " + width);
    }
    Arrays.fill(out, 0, width, 0f);
    int seats = variant.seats();
    int dealer = table.dealer();
    for (GrpField field : present) {
      int base = offsets[field.ordinal()];
      switch (field) {
        case ROUND -> out[base + table.round()] = 1f;
        case HONBA -> out[base] = table.honba() * variant.ronHonba() / POINTS;
        case KYOTAKU -> out[base] = table.kyotaku() * Variant.DEPOSIT / POINTS;
        case SCORE -> {
          for (int seat = 0; seat < seats; seat++) {
            out[base + seat] = (table.score(Seats.absolute(dealer, seat, seats)) - variant.startPoints()) / POINTS;
          }
        }
        case RANK -> {
          for (int seat = 0; seat < seats; seat++) {
            out[base + seat * seats + table.rank(Seats.absolute(dealer, seat, seats))] = 1f;
          }
        }
        case GAP -> {
          int pair = 0;
          for (int above = 0; above < seats; above++) {
            for (int below = above + 1; below < seats; below++) {
              out[base + pair++] = (scoreOfRank(table, above) - scoreOfRank(table, below)) / POINTS;
            }
          }
        }
      }
    }
  }

  /** 指定した順位の席の点を返す。順位は席の順列なので、どの順位にもちょうど 1 席いる。 */
  private int scoreOfRank(TableState table, int rank) {
    int seat = 0;
    while (table.rank(seat) != rank) {
      seat++;
    }
    return table.score(seat);
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof GrpLayout layout && variant == layout.variant && fields.equals(layout.fields);
  }

  @Override
  public int hashCode() {
    return Objects.hash(variant, fields);
  }

  @Override
  public String toString() {
    return "GrpLayout[" + variant + " " + fields + " width=" + width + "]";
  }
}
