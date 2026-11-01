package com.epsilon.prism.model;

import com.epsilon.core.GameState;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;

/**
 * 補助の頭の教師の 1 塊（ホストの配列）。行から教師を作る AuxiliaryTargets はこの形に合わせる。
 *
 * <p><b>候補の教師</b>（{@link CandidateTarget}、教師のある（行, 手）だけを並べる。手の役割は {@link TargetSchema#hasCandidateTargets}
 * が真のもの）
 *
 * <ul>
 *   <li>{@code candidateActions} [N_c]: 塊の手の通し番号。
 *   <li>{@code outcomes} [N_c]: 局の終わり方（{@link Outcome} の ordinal）。頭ごとの条件（{@link CandidateTarget#condition()}）はここから
 *       決め、満たさない（行, 手）は損失にも分母にも入れない。
 *   <li>{@code candidateValues} [N_c × {@link CandidateTarget#WIDTH}]（行優先）: 列 {@code offset() + k} が頭の区画の列 k。値は生の値で、
 *       損失が {@link CandidateTarget#scale()} で割る。OUTCOME は 8 類の one-hot、TENPAI・YAKU_TENPAI は 0/1、YAKU は列 k が
 *       ScoringYaku の ordinal k の役の 0/1、WIN_POINTS・DEAL_IN_POINTS・POINT_DELTA は点、WINNER は和了した他家の相対席（下家・
 *       対面・上家）の one-hot。条件を満たさない区画の値は読まない（有限であればよい）。
 * </ul>
 *
 * <p><b>読みの教師</b>（{@link ReadingTarget}、神託のある行だけを並べる）
 *
 * <ul>
 *   <li>{@code readingRows} [N_r]: 塊の行の番号。
 *   <li>{@code readingValues} [N_r × 3 × {@link ReadingTarget#WIDTH}]（行優先）: 相手（下家・対面・上家）ごとに、TENPAI は 0/1、SHANTEN は
 *       4 類の one-hot、DANGER・WAITS は牌（正準の色の番号）ごとの 0/1、DANGER_POINTS は牌ごとの点。WAITS は TENPAI が 1 の相手だけ、
 *       DANGER_POINTS は DANGER が 1 の牌だけを使う。
 * </ul>
 *
 * @param candidateActions 候補の手
 * @param outcomes 候補ごとの局の終わり方
 * @param candidateValues 候補の教師
 * @param readingRows 読みの行
 * @param readingValues 読みの教師
 */
public record AuxiliaryLabels(
    int[] candidateActions,
    int[] outcomes,
    float[] candidateValues,
    int[] readingRows,
    float[] readingValues) {

  /** 読みの相手の数。 */
  public static final int OPPONENTS = GameState.NUM_PLAYERS - 1;

  /**
   * 形と値を確かめる。
   *
   * @throws IllegalArgumentException 長さが合わないか、局の終わり方が範囲の外か、値が有限でないとき
   */
  public AuxiliaryLabels {
    int candidates = candidateActions.length;
    if (outcomes.length != candidates
        || candidateValues.length != candidates * CandidateTarget.WIDTH
        || readingValues.length != readingRows.length * OPPONENTS * ReadingTarget.WIDTH) {
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
   * 教師のない塊を返す（補助の頭を計算するが損失は 0）。
   *
   * @return 空の教師
   */
  public static AuxiliaryLabels empty() {
    return new AuxiliaryLabels(new int[0], new int[0], new float[0], new int[0], new float[0]);
  }

  private static void requireFinite(float[] values) {
    for (float value : values) {
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException("auxiliary labels must be finite");
      }
    }
  }
}
