package com.epsilon.prism.input;

import com.epsilon.mahjong.rules.Vocabulary;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Objects;

/** 入力のフィールドに書く値（ID と個数）の求め方。入力を書くコードはここだけを使い、{@code + 1} をあちこちに書かない。 */
public final class Ids {

  public static final int NONE = 0;

  private Ids() {}

  /**
   * enum の値の ID（語彙の中の位置 + 1）を返す。0 は、フィールドが当てはまらないこと（打牌のイベントの副露のフィールドなど）を表す。
   * enum の側の NONE などの定数は、フィールドは当てはまるが値が無いこと（赤のない副露など）を表す。四人打ちでは位置が ordinal と同じなので、
   * ID は ordinal + 1 になる。
   *
   * @param values フィールドの語彙
   * @param value 値（null なら値が無い）
   * @return 位置 + 1、null なら {@link #NONE}
   * @throws IllegalArgumentException 値が、そのルールの語彙に無い定数のとき
   */
  @SuppressWarnings({"unchecked", "rawtypes"})
  public static int of(Vocabulary values, Enum<?> value) {
    return value == null ? NONE : values.index((Enum) value) + 1;
  }

  /** 牌の種類の ID を返す（種類 + 1、負なら {@link #NONE}）。 */
  public static int tile(int kind) {
    return kind < 0 ? NONE : Objects.checkIndex(kind, Tiles.KINDS) + 1;
  }

  /** 自分から見た相対の席の ID を返す（相対の席 + 1、負なら {@link #NONE}）。 */
  public static int seat(int relativeSeat, int seats) {
    return relativeSeat < 0 ? NONE : Objects.checkIndex(relativeSeat, seats) + 1;
  }

  /**
   * 0 始まりの番号（向聴・イベントの番号・何打目か）の ID を返す。
   *
   * @param value 番号（負なら値が無い）
   * @param kinds 番号が取りうる値の数（フィールドの {@link FieldSpec#ids} に渡した数）
   * @return 番号 + 1、負なら {@link #NONE}
   * @throws IndexOutOfBoundsException 番号が取りうる値の数以上のとき
   */
  public static int index(int value, int kinds) {
    return value < 0 ? NONE : Objects.checkIndex(value, kinds) + 1;
  }

  /**
   * 上限で頭打ちにした個数を返す。上限の意味は、そのフィールドの Javadoc に書く。
   *
   * @param value 個数
   * @param saturation 上限（これ以上の個数は全て上限にまとめる）
   * @return 上限以上なら上限、ほかはそのままの個数
   * @throws IllegalArgumentException 個数が負のとき
   */
  public static int saturated(int value, int saturation) {
    if (value < 0) {
      throw new IllegalArgumentException("negative count: " + value);
    }
    return Math.min(value, saturation);
  }
}
