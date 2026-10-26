package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import java.util.List;

/**
 * 当てはめの損失（{@link PrismLoss}・{@link Distill}）。学習器（{@link PrismLearner}）は 1 歩の損失を (1/B)·Σ_微小バッチ {@link #total} にする。B は
 * 1 歩の全レーンで {@link #contributes} が真の行の数（なければ 1）。
 */
public interface Objective<E extends Example> {
  /** 損失の分母に数える行か（数えない行は {@link #total} に何も足さない）。 */
  boolean contributes(E example);

  /**
   * 微小バッチの行の損失の和（スカラー、FP32。分母で割る前）。
   *
   * @param scope 微小バッチの管理元（作った配列はここに付ける）
   * @param scores 出力 s。[行, 容量]、FP32、勾配を持つ
   * @param legal 合法手のマスク。[行, 容量]（真偽値でなければ正の枠を合法とする）
   * @param part 微小バッチの行（scores の行と同じ順）
   * @param capacity 合法手の容量（scores の列の数）
   */
  NDArray total(NDManager scope, NDArray scores, NDArray legal, List<E> part, int capacity);
}
