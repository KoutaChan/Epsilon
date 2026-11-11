package com.epsilon.prism.input;

/**
 * 追い越すのに要る、最小の和了の格。各段は基本点で表し、本場・供託を含む点の移動は {@code Settlement} の式で求める。NOT_NEEDED と
 * UNREACHABLE は和了の点を持たない（{@link #scored()} が false で、{@link #basePoints()} は例外にする）。
 */
public enum PassGrade {
  /** 和了しなくても既に上にいる。 */
  NOT_NEEDED,
  HAN1_FU30(240),
  HAN1_FU40(320),
  HAN2_FU30(480),
  HAN2_FU40(640),
  HAN3_FU30(960),
  HAN3_FU40(1280),
  HAN4_FU30(1920),
  MANGAN(2000),
  HANEMAN(3000),
  BAIMAN(4000),
  SANBAIMAN(6000),
  YAKUMAN(8000),
  /** 役満でも届かない。 */
  UNREACHABLE;

  /** 和了の点を持たない段の基本点。 */
  private static final int UNSCORED = 0;

  private final int basePoints;

  PassGrade() {
    this(UNSCORED);
  }

  PassGrade(int basePoints) {
    this.basePoints = basePoints;
  }

  /**
   * 和了の点を持つ段（NOT_NEEDED と UNREACHABLE 以外）なら true を返す。
   *
   * @return 基本点を持つなら true
   */
  public boolean scored() {
    return basePoints != UNSCORED;
  }

  /**
   * 基本点を返す。
   *
   * @return 基本点
   * @throws IllegalStateException 和了の点を持たない段（NOT_NEEDED・UNREACHABLE）のとき
   */
  public int basePoints() {
    if (!scored()) {
      throw new IllegalStateException(name() + " has no base points");
    }
    return basePoints;
  }
}
