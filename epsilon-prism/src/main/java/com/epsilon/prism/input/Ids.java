package com.epsilon.prism.input;

import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import java.util.Objects;

/** 入力の欄の値（ID と数え上げ）の書き方。書き手はここだけを使い、{@code + 1} を散らさない。 */
public final class Ids {

  public static final int NONE = 0;

  private Ids() {}

  /**
   * enum の値の ID を返す。外の enum に NONE などの定数があっても ordinal + 1 で書く。0 は欄が当てはまらないこと（打牌の出来事の副露の欄
   * など）、NONE は欄が当てはまって値がないこと（赤のない副露など）を表す。
   *
   * @param value 値（null はなし）
   * @return ordinal + 1、null なら {@link #NONE}
   */
  public static int of(Enum<?> value) {
    return value == null ? NONE : value.ordinal() + 1;
  }

  /**
   * 牌の種類の ID を返す。
   *
   * @param tileType 牌の種類（負はなし）
   * @return 牌の種類 + 1、負なら {@link #NONE}
   */
  public static int tile(int tileType) {
    return tileType < 0 ? NONE : Objects.checkIndex(tileType, Tile.NUM_TILE_TYPES) + 1;
  }

  /**
   * 観測席から見た相対席の ID を返す。
   *
   * @param relativeSeat 相対席（負はなし）
   * @return 相対席 + 1、負なら {@link #NONE}
   */
  public static int seat(int relativeSeat) {
    return relativeSeat < 0
        ? NONE
        : Objects.checkIndex(relativeSeat, GameState.NUM_PLAYERS) + 1;
  }

  /**
   * 0 始まりの番号（向聴・出来事の番号・何打目か）の ID を返す。
   *
   * @param value 番号（負はなし）
   * @param kinds 番号の種類の数（欄の {@link FieldSpec#ids} に渡した数）
   * @return 番号 + 1、負なら {@link #NONE}
   * @throws IndexOutOfBoundsException 番号が種類の数以上のとき
   */
  public static int index(int value, int kinds) {
    return value < 0 ? NONE : Objects.checkIndex(value, kinds) + 1;
  }

  /**
   * まとめる欄の数え上げを返す。まとめ方は欄の Javadoc に書く。
   *
   * @param value 数
   * @param saturation まとめ始める値
   * @return saturation 以上は saturation、ほかはそのまま
   * @throws IllegalArgumentException 数が負のとき
   */
  public static int saturated(int value, int saturation) {
    if (value < 0) {
      throw new IllegalArgumentException("negative count: " + value);
    }
    return Math.min(value, saturation);
  }
}
