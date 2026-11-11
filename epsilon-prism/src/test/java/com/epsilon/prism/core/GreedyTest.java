package com.epsilon.prism.core;

import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 貪欲の比べ方（{@link Greedy}）: s の大きい手、同点なら p0 の大きい手、それも同じなら番号の小さい手が上。s が NaN の手は、どの位置に
 * あっても NaN でない手より下になり、NaN どうしは同点として p0 と番号で決まる。
 */
public class GreedyTest {
  private static final float NAN = Float.NaN;

  @Test
  public void tiesGoToThePriorThenTheIndex() {
    Assert.assertEquals(Greedy.argmax(new float[] {0.1f, 0.3f, 0.3f}, null, 3), 1);
    Assert.assertEquals(Greedy.argmax(new float[] {0.1f, 0.3f, 0.3f}, new float[] {0.5f, 0.2f, 0.3f}, 3), 2);
    Assert.assertEquals(Greedy.argmax(new float[] {0.1f, 0.3f, 0.3f}, new float[] {0.5f, 0.25f, 0.25f}, 3), 1);
    Assert.assertEquals(Greedy.argmax(new float[] {0.1f, 0.3f, 0.9f}, null, 2), 1);
  }

  /** NaN が先頭にあっても選ばない。 */
  @Test
  public void nanIsBelowEveryNumberAtAnyPosition() {
    Assert.assertEquals(Greedy.argmax(new float[] {NAN, 0.3f, 0.1f}, null, 3), 1);
    Assert.assertEquals(Greedy.argmax(new float[] {0.1f, NAN, 0.3f}, null, 3), 2);
    Assert.assertEquals(Greedy.argmax(new float[] {0.3f, 0.1f, NAN}, null, 3), 0);
    Assert.assertEquals(Greedy.argmax(new float[] {NAN, Float.NEGATIVE_INFINITY}, null, 2), 1);
    Assert.assertEquals(Greedy.argmax(new float[] {NAN, 0.3f, 0.3f}, new float[] {0.6f, 0.1f, 0.3f}, 3), 2);
    float[] scores = {NAN, 0.3f};
    Assert.assertTrue(Greedy.before(1, 0, scores, null));
    Assert.assertFalse(Greedy.before(0, 1, scores, null));
  }

  /** NaN どうしは同点。全ての手が NaN なら、p0 の大きい手、それも同じなら番号の小さい手。 */
  @Test
  public void nanTiesWithNan() {
    float[] scores = {NAN, NAN, NAN};
    Assert.assertEquals(Greedy.argmax(scores, null, 3), 0);
    Assert.assertEquals(Greedy.argmax(scores, new float[] {0.2f, 0.5f, 0.3f}, 3), 1);
    Assert.assertTrue(Greedy.before(0, 1, scores, null));
    Assert.assertFalse(Greedy.before(1, 0, scores, null));
  }

  /** 挿入ソートの順位は、NaN がどこにあっても同じ（NaN でない手の降順の後に NaN の手）。 */
  @Test
  public void insertionSortDoesNotDependOnWhereNanIs() {
    Assert.assertEquals(sort(new float[] {0.1f, NAN, 0.3f}), new int[] {2, 0, 1});
    Assert.assertEquals(sort(new float[] {NAN, 0.1f, 0.3f}), new int[] {2, 1, 0});
    Assert.assertEquals(sort(new float[] {0.3f, NAN, NAN, 0.5f}), new int[] {3, 0, 1, 2});
  }

  private static int[] sort(float[] scores) {
    int[] order = new int[scores.length];
    for (int i = 0; i < scores.length; i++) {
      int j = i - 1;
      while (j >= 0 && Greedy.before(i, order[j], scores, null)) {
        order[j + 1] = order[j];
        j--;
      }
      order[j + 1] = i;
    }
    return order;
  }
}
