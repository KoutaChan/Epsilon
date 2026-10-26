package com.epsilon.core;

import org.testng.Assert;
import org.testng.annotations.Test;

/** 河の記録はルール上の上限を持たない（鳴きで他家の手番が飛ぶと1席の打牌は 24 枚を超えうる）。 */
public class RiverTest {
  @Test
  public void growsPastTheInitialCapacity() {
    River river = new River();
    for (int i = 0; i < 30; i++) river.append(i % 34, i, i == 3, false, i % 2 == 0, i);
    river.markLastCalled();
    Assert.assertEquals(river.size(), 30);
    Assert.assertEquals(river.discard(29).tileType(), 29);
    Assert.assertTrue(river.discard(29).called());
    Assert.assertEquals(river.riichiDeclarationIndex(), 3);
    Assert.assertEquals(river.postRiichiDiscardCount(), 26);
  }

  @Test
  public void copyKeepsEveryDiscardAndIsIndependent() {
    River river = new River();
    for (int i = 0; i < 26; i++) river.append(i % 34, i, false, false, false, i);
    River copy = new River(river);
    Assert.assertEquals(copy.toString(), river.toString());
    copy.append(5, 26, false, false, false, 26);
    copy.markLastCalled();
    Assert.assertEquals(copy.size(), 27);
    Assert.assertEquals(river.size(), 26);
    Assert.assertFalse(river.discard(25).called());
    river.clear();
    Assert.assertEquals(copy.discard(10).tileType(), 10);
    river.append(7, 0, false, false, false, 0);
    Assert.assertEquals(river.discard(0).tileType(), 7);
    Assert.assertFalse(river.discard(0).called());
  }
}
