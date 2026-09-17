package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBuffer;

/** 公開情報で確定する役・翻・符・基本点を所有する再利用出力。 */
public final class VisibleHandScoreBuffer {
  private long yakuBits;
  private int visibleHan;
  private int fu;
  private int basePoints;

  public VisibleHandScoreBuffer() {}

  public VisibleHandScoreBuffer(long yakuBits, int visibleHan, int fu, int basePoints) {
    set(yakuBits, visibleHan, fu, basePoints);
  }

  private void set(long yakuBits, int visibleHan, int fu, int basePoints) {
    this.yakuBits = yakuBits;
    this.visibleHan = visibleHan;
    this.fu = fu;
    this.basePoints = basePoints;
  }

  public long yakuBits() {
    return yakuBits;
  }

  public int visibleHan() {
    return visibleHan;
  }

  public int fu() {
    return fu;
  }

  public int basePoints() {
    return basePoints;
  }

  public boolean available() {
    return yakuBits != 0L;
  }

  static void copyInto(HandScoreBuffer source, VisibleHandScoreBuffer destination) {
    if (source == null) destination.set(0L, 0, 0, 0);
    else destination.set(source.yakuBits(), source.han(), source.fu(), source.basePoints());
  }
}
