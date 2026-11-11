package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * {@code WallReplay} が、まだ読まれていない牌山の位置だけを並べ替えることを確かめる。牌譜の局を和了・槓・鳴きの多い打ち方で打ち、判断の時点で
 * 並べ替えて打ち直す。読まれた位置（配牌・山のツモ・表示牌・嶺上と補充）の牌はそのまま残る。ほかの位置は、牌の組み合わせを変えずに
 * 並べ替わる（裏ドラ表示牌の位置も入る）。打ち直したエンジンは、元のエンジンと同じビューと判断を持つ（{@code WallReplay} が確かめる）。
 */
public class WallReplayTest {

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test(dataProvider = "variants")
  public void aReplayKeepsTheReadPositionsAndShufflesTheRest(Variant variant) {
    Random random = new Random(29);
    int replays = 0;
    int afterReplacement = 0;
    int uraMoves = 0;
    for (int game = 0; game < 40; game++) {
      int[] wall = Scenes.first(variant).wall(random.nextLong());
      WallReplay replay = new WallReplay(RoundStart.first(variant), wall);
      while (replay.boundary() == Boundary.DECISIONS) {
        Engine engine = replay.engine();
        if (random.nextInt(3) == 0) {
          long seed = random.nextLong();
          int[] shuffled = replay.reshuffled(seed);
          Assert.assertEquals(sorted(shuffled), sorted(wall));
          boolean[] read = read(engine.full());
          for (int position = 0; position < wall.length; position++) {
            if (read[position]) {
              Assert.assertEquals(shuffled[position], wall[position], "read position " + position);
            }
          }
          for (int index = 0; index < engine.full().seat(0).indicatorCount(); index++) {
            int ura = Wall.uraPosition(variant, index);
            uraMoves += shuffled[ura] != wall[ura] ? 1 : 0;
          }
          Engine replayed = replay.replay(seed);
          Assert.assertEquals(replayed.decisions().size(), engine.decisions().size());
          afterReplacement += replacements(engine.full().chance()) > 0 ? 1 : 0;
          replays++;
        }
        Decisions decisions = engine.decisions();
        int[] choices = new int[decisions.size()];
        for (int slot = 0; slot < choices.length; slot++) {
          choices[slot] = random.nextBoolean() ? 0 : random.nextInt(decisions.legalCount(slot));
        }
        replay.commit(choices);
      }
    }
    Assert.assertTrue(replays > 100, "replays " + replays);
    Assert.assertTrue(afterReplacement > 0, "replays after a replacement draw " + afterReplacement);
    Assert.assertTrue(uraMoves > 0, "moved ura positions " + uraMoves);
  }

  /**
   * 牌山は全ての位置が分かっていて、ルールで決まる長さでなければならない。並べ替えと打ち直しは、元のエンジンが判断の
   * 時点にあるときだけできる。
   */
  @Test
  public void aReplayNeedsAKnownWallAndADecision() {
    Variant variant = Variant.TENHOU_4P;
    RoundStart start = RoundStart.first(variant);
    int[] wall = Scenes.first(variant).wall(1);
    int[] unknown = wall.clone();
    unknown[100] = Wall.UNKNOWN;
    Assert.assertThrows(IllegalArgumentException.class, () -> new WallReplay(start, unknown));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> new WallReplay(start, Arrays.copyOf(wall, wall.length - 1)));
    WallReplay replay = new WallReplay(start, wall);
    while (replay.boundary() == Boundary.DECISIONS) {
      replay.commit(new int[replay.engine().decisions().size()]);
    }
    Assert.assertThrows(IllegalStateException.class, () -> replay.reshuffled(3));
    Assert.assertThrows(IllegalStateException.class, () -> replay.replay(3));
    Assert.assertThrows(IllegalStateException.class, () -> replay.commit(0));
  }

  /**
   * 判断の時点までに読まれた位置を、偶然のイベントの数から求め直す。配牌と山のツモは前から順に読まれる。嶺上と補充は王牌の末尾から
   * 読まれ、5 回目からは王牌のすぐ手前、つまり山の末尾から読まれる。表示牌 i は、牌山の長さを L として位置 L − 6 − 2i から読まれる。
   */
  private static boolean[] read(FullView full) {
    Variant variant = full.variant();
    int size = variant.tiles().size();
    ChanceLog chance = full.chance();
    int draws = 0;
    for (int index = 0; index < chance.size(); index++) {
      draws += chance.kind(index) == ChanceKind.DRAW ? 1 : 0;
    }
    boolean[] read = new boolean[size];
    for (int position = 0; position < Variant.HAND * variant.seats() + draws; position++) {
      read[position] = true;
    }
    for (int k = 1; k <= replacements(chance); k++) {
      read[k <= Variant.MAX_KANS ? size - k : size - Variant.DEAD_WALL - (k - Variant.MAX_KANS)] = true;
    }
    for (int index = 0; index < full.seat(0).indicatorCount(); index++) {
      read[size - Variant.MAX_KANS - 2 - 2 * index] = true;
    }
    return read;
  }

  /** 嶺上・補充のツモの数。 */
  private static int replacements(ChanceLog chance) {
    int count = 0;
    for (int index = 0; index < chance.size(); index++) {
      count += chance.kind(index) == ChanceKind.RINSHAN ? 1 : 0;
    }
    return count;
  }

  private static int[] sorted(int[] wall) {
    int[] copy = wall.clone();
    Arrays.sort(copy);
    Assert.assertTrue(copy[0] >= 0 && copy[copy.length - 1] < Tiles.PHYSICAL_IDS);
    return copy;
  }
}
