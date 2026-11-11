package com.epsilon.prism.model;

import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;

/**
 * 補助のヘッドの教師を、1 バッチ分ホストの配列で持つ。ルールごとに行から教師を作るライターは、この形に合わせる。
 *
 * <p><b>候補の教師</b>（{@link CandidateTarget}）。教師のある（行, 手）だけを並べる。並べるのは、手の種類で
 * {@link TargetSchema#hasCandidateTargets} が真になる手だけ。
 *
 * <ul>
 *   <li>{@code candidateActions} [N_c]: バッチの中での手の通し番号。
 *   <li>{@code outcomes} [N_c]: 局の終わり方（{@link Outcome} の ordinal）。ヘッドごとの条件（{@link CandidateTarget#condition()}）はこれで
 *       決める。条件を満たさない（行, 手）は、損失にも分母にも入れない。
 *   <li>{@code candidateValues} [N_c × {@link TargetSchema#candidateWidth}]（行優先）: 列 {@code offset + k} が、そのヘッドの領域の列 k。
 *       値は生の値で、損失の側で {@link CandidateTarget#scale()} で割る。OUTCOME は 8 通りの one-hot、TENPAI と YAKU_TENPAI は 0/1。
 *       YAKU は、列 k が ScoringYaku の ordinal k の役の 0/1。WIN_POINTS・DEAL_IN_POINTS・POINT_DELTA は点。WINNER は、和了した他家の
 *       相対席（下家から上家まで）の one-hot。条件を満たさない領域の値は読まない（有限であればよい）。
 * </ul>
 *
 * <p><b>読みの教師</b>（{@link ReadingTarget}）。Oracle の値がある行だけを並べる。
 *
 * <ul>
 *   <li>{@code readingRows} [N_r]: バッチの中での行の番号。
 *   <li>{@code readingValues} [N_r × 相手の数 × {@link TargetSchema#readingWidth}]（行優先）: 相手（下家から上家まで）ごとに並べる。
 *       TENPAI は 0/1、SHANTEN は 4 通りの one-hot。DANGER と WAITS は、牌（正準の色の順に並べた牌のトークン）ごとの 0/1。DANGER_POINTS
 *       は牌ごとの点。WAITS は TENPAI が 1 の相手だけ、DANGER_POINTS は DANGER が 1 の牌だけを使う。
 * </ul>
 *
 * @param heads 補助のヘッドの形（ルールで決まる相手の数と、領域の幅）
 * @param candidateActions 候補の手
 * @param outcomes 候補ごとの局の終わり方
 * @param candidateValues 候補の教師
 * @param readingRows 読みの行
 * @param readingValues 読みの教師
 */
public record AuxiliaryLabels(
    TargetSchema heads,
    int[] candidateActions,
    int[] outcomes,
    float[] candidateValues,
    int[] readingRows,
    float[] readingValues) {

  /**
   * 形と値を確かめる。
   *
   * @throws IllegalArgumentException 長さが合わないか、局の終わり方が範囲の外か、値が有限でないとき
   */
  public AuxiliaryLabels {
    int candidates = candidateActions.length;
    if (outcomes.length != candidates
        || candidateValues.length != candidates * heads.candidateWidth()
        || readingValues.length != readingRows.length * heads.opponents() * heads.readingWidth()) {
      throw new IllegalArgumentException("auxiliary label lengths do not match");
    }
    for (int outcome : outcomes) {
      if (outcome < 0 || outcome >= Outcome.values().length) {
        throw new IllegalArgumentException("invalid outcome: " + outcome);
      }
    }
    requireFinite(candidateValues);
    requireFinite(readingValues);
  }

  /**
   * 教師のないバッチを返す。補助のヘッドは計算するが、損失は 0 になる。
   *
   * @param heads 補助のヘッドの形
   * @return 空の教師
   */
  public static AuxiliaryLabels empty(TargetSchema heads) {
    return new AuxiliaryLabels(heads, new int[0], new int[0], new float[0], new int[0], new float[0]);
  }

  private static void requireFinite(float[] values) {
    for (float value : values) {
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException("auxiliary labels must be finite");
      }
    }
  }
}
