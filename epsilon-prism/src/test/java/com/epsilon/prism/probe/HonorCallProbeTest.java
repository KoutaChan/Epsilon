package com.epsilon.prism.probe;

import com.epsilon.core.Tile;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 字牌の種類は席の視点（自風・場風・ダブ風・三元牌・オタ風）で、数牌は null。 */
public class HonorCallProbeTest {
  @Test
  public void kindsFollowTheSeatAndRoundWinds() {
    // 南家（自風 南）、東場（場風 東）
    Assert.assertEquals(HonorCallProbe.kind(Tile.NAN, Tile.NAN, Tile.TON), HonorCallProbe.Kind.SEAT);
    Assert.assertEquals(HonorCallProbe.kind(Tile.TON, Tile.NAN, Tile.TON), HonorCallProbe.Kind.ROUND);
    Assert.assertEquals(HonorCallProbe.kind(Tile.PEI, Tile.NAN, Tile.TON), HonorCallProbe.Kind.GUEST);
    Assert.assertEquals(HonorCallProbe.kind(Tile.CHUN, Tile.NAN, Tile.TON), HonorCallProbe.Kind.DRAGON);
    // 親（自風 東）の東場は東がダブ風
    Assert.assertEquals(HonorCallProbe.kind(Tile.TON, Tile.TON, Tile.TON), HonorCallProbe.Kind.DOUBLE);
    Assert.assertNull(HonorCallProbe.kind(0, Tile.TON, Tile.TON));
  }
}
