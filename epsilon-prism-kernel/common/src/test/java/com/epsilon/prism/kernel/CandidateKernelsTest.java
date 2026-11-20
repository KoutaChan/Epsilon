package com.epsilon.prism.kernel;

import static org.testng.Assert.expectThrows;

import org.testng.annotations.Test;

/** 候補の演算が受け取る遷移ごとの行の番号の並び（{@link CandidateKernels#requireGroupedRows}）を確かめる。 */
public class CandidateKernelsTest {
  @Test
  public void acceptsRowsThatNeverDecrease() {
    CandidateKernels.requireGroupedRows(new int[0], 0);
    CandidateKernels.requireGroupedRows(new int[0], 3);
    CandidateKernels.requireGroupedRows(new int[] {0, 0, 1, 1, 1, 2}, 3);
    CandidateKernels.requireGroupedRows(new int[] {1, 1, 3}, 4);
  }

  @Test
  public void rejectsRowsOutOfOrderOrOutsideTheBatch() {
    for (int[] rows : new int[][] {{1, 0}, {0, 2, 1}, {0, 1, 1, 0}, {-1, 0}, {0, 3}}) {
      expectThrows(
          IllegalArgumentException.class, () -> CandidateKernels.requireGroupedRows(rows, 3));
    }
    expectThrows(
        IllegalArgumentException.class, () -> CandidateKernels.requireGroupedRows(new int[1], 0));
  }
}
