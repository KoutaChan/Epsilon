package com.epsilon.mahjong.util;

import java.util.HashSet;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/** SplitMix64 の公開された参照値と一致すること、ソルト・番号・座標のどれかが違えば別の種になること。 */
public class SeedMixerTest {

  /** splitmix64.c（Vigna）を状態 1234567 から始めたときの、最初の 5 つの出力。 */
  private static final long[] REFERENCE_1234567 = {
    6457827717110365317L,
    3203168211198807973L,
    Long.parseUnsignedLong("9817491932198370423"),
    4593380528125082431L,
    Long.parseUnsignedLong("16408922859458223821"),
  };

  @Test
  public void indexedWithoutSaltFollowsTheSplitMix64Stream() {
    for (int index = 0; index < REFERENCE_1234567.length; index++) {
      Assert.assertEquals(SeedMixer.indexed(1234567L, 0, index + 1), REFERENCE_1234567[index]);
    }
  }

  @Test
  public void mixFixesZeroAndSeparatesNeighbours() {
    Assert.assertEquals(SeedMixer.mix64(0), 0L);
    Set<Long> seen = new HashSet<>();
    for (long value = -50_000; value < 50_000; value++) {
      Assert.assertTrue(seen.add(SeedMixer.mix64(value)), "collision at " + value);
    }
  }

  @Test
  public void saltsAndIndexesSeparateSeeds() {
    Set<Long> seeds = new HashSet<>();
    for (long salt : new long[] {0, 1, 0xC6BC279692B5CC83L}) {
      for (long index = 0; index < 1_000; index++) {
        Assert.assertTrue(seeds.add(SeedMixer.indexed(42, salt, index)), "salt " + salt + " index " + index);
      }
    }
  }

  @Test
  public void deriveDependsOnTheOrderOfCoordinates() {
    Assert.assertEquals(SeedMixer.derive(7, 9), SeedMixer.mix64(7 ^ 9));
    Assert.assertEquals(SeedMixer.derive(7, 9, 1, 2), SeedMixer.derive(7, 9, 1, 2));
    Assert.assertNotEquals(SeedMixer.derive(7, 9, 1, 2), SeedMixer.derive(7, 9, 2, 1));
    Assert.assertNotEquals(SeedMixer.derive(7, 9, 1), SeedMixer.derive(7, 9, 1, 0));
  }
}
