package com.epsilon.prism.hip;

import java.util.Arrays;

/**
 * 1 塊の入力の形: 数と、blob の中の区画の位置。HostBatch の計画（BatchPlan）から網の側が作る。
 *
 * @param rows 行の数
 * @param events 出来事のトークンの数
 * @param transitions 遷移の数
 * @param actions 合法手の数
 * @param bytes 使う blob のバイト数（H2D はこの大きさを 1 回）
 * @param sections 区画ごとの先頭のバイトの位置（{@link HipSection} の順）
 */
public record HipBatch(int rows, int events, int transitions, int actions, long bytes, long[] sections) {

  /** 区画の先頭のそろえ（native の 16 バイトの読み込みに要る下限）。 */
  public static final int SECTION_ALIGNMENT = 16;

  /**
   * 数と位置を確かめる。
   *
   * @throws IllegalArgumentException 数が負か、行が 0 か、区画の数・そろえが合わないとき
   */
  public HipBatch {
    if (rows <= 0 || events < 0 || transitions < rows || actions < rows || bytes <= 0) {
      throw new IllegalArgumentException(
          "invalid batch counts: rows="
              + rows
              + " events="
              + events
              + " transitions="
              + transitions
              + " actions="
              + actions
              + " bytes="
              + bytes);
    }
    if (sections.length != HipSection.values().length) {
      throw new IllegalArgumentException(
          "expected " + HipSection.values().length + " section offsets, got " + sections.length);
    }
    sections = sections.clone();
    for (HipSection section : HipSection.values()) {
      long offset = sections[section.ordinal()];
      if (offset < 0 || offset > bytes || offset % SECTION_ALIGNMENT != 0) {
        throw new IllegalArgumentException(
            section + " offset " + offset + " must be inside the blob and aligned to " + SECTION_ALIGNMENT);
      }
    }
  }

  public long offset(HipSection section) {
    return sections[section.ordinal()];
  }

  @Override
  public long[] sections() {
    return sections.clone();
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof HipBatch that
        && rows == that.rows
        && events == that.events
        && transitions == that.transitions
        && actions == that.actions
        && bytes == that.bytes
        && Arrays.equals(sections, that.sections);
  }

  @Override
  public int hashCode() {
    return 31 * (31 * (31 * (31 * (31 * rows + events) + transitions) + actions) + Long.hashCode(bytes))
        + Arrays.hashCode(sections);
  }

  @Override
  public String toString() {
    return "HipBatch[rows="
        + rows
        + ", events="
        + events
        + ", transitions="
        + transitions
        + ", actions="
        + actions
        + ", bytes="
        + bytes
        + ", sections="
        + Arrays.toString(sections)
        + "]";
  }
}
