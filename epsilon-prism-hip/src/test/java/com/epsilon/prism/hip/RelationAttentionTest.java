package com.epsilon.prism.hip;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import org.testng.annotations.Test;

/** {@link RelationAttention.Dimensions} のテスト（GPU の演算が受け付ける形）。 */
public class RelationAttentionTest {

  /**
   * 本番の形（ヘッド 8・ワード 2・セグメント 7・セグメントあたりの値 16）と、学習のテストで使う形（ヘッド 2）は通る。本番の形では、
   * 幅と表の行の数も確かめる。ヘッドの数が奇数のときと、ワードの数が 2 でないときは例外で止める。
   */
  @Test
  public void dimensionsRejectOddHeadsAndOtherWords() {
    RelationAttention.Dimensions production =
        new RelationAttention.Dimensions(48, 39, 121, 121, 8, RelationCodes.WORDS, 7, 16, 0.25f);
    assertEquals(production.width(), 8 * RelationAttention.FEATURES);
    assertEquals(production.tableRows(), 7 * 16);
    new RelationAttention.Dimensions(12, 70, 70, 70, 2, RelationCodes.WORDS, 8, 12, 0.25f);

    for (int heads : new int[] {1, 3, 7}) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new RelationAttention.Dimensions(48, 70, 70, 70, heads, RelationCodes.WORDS, 7, 16, 0.25f));
    }
    for (int words : new int[] {1, 3, 4}) {
      assertThrows(
          IllegalArgumentException.class,
          () -> new RelationAttention.Dimensions(48, 70, 70, 70, 8, words, 7, 16, 0.25f));
    }
  }

  /** 範囲の外の値で止めるときの例外のメッセージは、渡した次元を示す。 */
  @Test
  public void rejectedDimensionsReportTheGivenValues() {
    String message =
        expectThrows(
                IllegalArgumentException.class,
                () -> new RelationAttention.Dimensions(48, 39, 121, 122, 3, RelationCodes.WORDS, 7, 16, 0.25f))
            .getMessage();
    String[] givens = {
      "rows=48", "queries=39", "keys=121", "codeQueries=122", "heads=3", "words=" + RelationCodes.WORDS, "segments=7",
      "entries=16", "scale=0.25"
    };
    for (String given : givens) {
      assertTrue(message.contains(given), message);
    }
  }
}
