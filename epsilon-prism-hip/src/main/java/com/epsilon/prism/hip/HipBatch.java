package com.epsilon.prism.hip;

import java.util.Arrays;

/**
 * 1 つのバッチの入力の形。行・イベント・遷移・合法手の数と、blob の中の各領域の位置を持つ。ネットワークの側が HostBatch の計画
 * （BatchPlan）から作る。
 *
 * @param rows 行の数
 * @param events イベントのトークンの数
 * @param transitions 遷移の数
 * @param actions 合法手の数
 * @param bytes 使う blob のバイト数（H2D のコピーはこの大きさで 1 回）
 * @param sections 領域ごとの先頭のバイト位置（{@link HipSection} の順）
 */
public record HipBatch(int rows, int events, int transitions, int actions, long bytes, long[] sections) {

  /** 領域の先頭のアラインメント（バイト）。native が 16 バイト単位で読むのに要る下限。 */
  public static final int SECTION_ALIGNMENT = 16;

  /**
   * 数と位置を確かめる。
   *
   * @throws IllegalArgumentException 行の数かバイト数が 0 以下、イベントの数が負、遷移か合法手が行より少ないとき。また、領域の数が
   *     {@link HipSection} と合わないか、領域の先頭が blob の外にあるかアラインメントに合わないとき
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
