package com.epsilon.prism.hip;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;

import org.testng.annotations.Test;

/** {@link RelationAttention.Dimensions} の約束（GPU の演算の形の決まり）。 */
public class RelationAttentionTest {

  /** 本番の形（8 頭・2 語・7 区画・16 値）と学習の試験の形（2 頭）は通り、奇数の頭と 2 でない語の数は英語の例外で止める。 */
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
}
