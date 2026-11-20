package com.epsilon.prism.input;

import com.epsilon.mahjong.tile.Tiles;

/** 遷移後の牌の状態を、牌キーの通し番号に変換する。 */
public final class TileKeyCode {
  private final int first;
  private final CodeSpace<Digit> space;
  private final int[] counts;
  private final int[] waits;
  private final int ukeire;
  private final int discarded;

  public TileKeyCode(int first, CodeSpace<Digit> space) {
    this.first = first;
    this.space = space;
    counts = parts(Digit.AFTER_COUNT, first);
    waits = parts(Digit.WAIT_CLASS, 0);
    ukeire = space.weight(Digit.UKEIRE);
    discarded = space.weight(Digit.DISCARDED);
  }

  private int[] parts(Digit digit, int offset) {
    int[] parts = new int[space.radix(digit)];
    for (int i = 0; i < parts.length; i++) parts[i] = offset + space.part(digit, i);
    return parts;
  }

  public int first() {
    return first;
  }

  public CodeSpace<Digit> space() {
    return space;
  }

  /** 牌キーの桁を上位から並べた順。 */
  public enum Digit {
    AFTER_COUNT,
    UKEIRE,
    WAIT_CLASS,
    DISCARDED;

    private int radix(int waits) {
      return switch (this) {
        case AFTER_COUNT -> Tiles.COPIES + 1;
        case UKEIRE, DISCARDED -> 2;
        case WAIT_CLASS -> waits;
      };
    }
  }

  public static CodeSpace<Digit> space(int waitDigits) {
    return new CodeSpace<>(Digit.class, digit -> digit.radix(waitDigits));
  }

  public int waitDigits() {
    return space.radix(Digit.WAIT_CLASS);
  }

  public int encode(int afterCount, boolean ukeire, int wait, boolean discarded) {
    return counts[afterCount]
        + waits[wait]
        + (ukeire ? this.ukeire : 0)
        + (discarded ? this.discarded : 0);
  }
}
