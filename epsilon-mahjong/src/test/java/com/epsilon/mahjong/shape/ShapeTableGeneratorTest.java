package com.epsilon.mahjong.shape;

import com.epsilon.mahjong.shape.HandShape.Block;
import com.epsilon.mahjong.shape.ShapeTableGenerator.Distances;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 生成器が求めた d を総当たり（BruteShape）と照合する。resources にある表が生成器の出力と同じことも確かめる。表が違えば、
 * 作り直した表を target/shape/hand-shape.bin に書いて失敗する。
 */
public class ShapeTableGeneratorTest {

  private static final int RANDOM_SUIT_CODES = 3_000;

  /** 数牌の枚数がこれ以下のコードは、全て照合する。 */
  private static final int SMALL_SUIT = 5;

  @Test
  public void suitDistancesMatchEnumeration() {
    long[] distances = ShapeTableGenerator.blockDistances(Block.MAN);
    BruteShape brute = new BruteShape(TileSet.FULL);
    int[] digits = new int[Block.MAN.size()];
    for (int code = 0; code < distances.length; code++) {
      if (ShapeTableGenerator.digits(code, digits) <= SMALL_SUIT) {
        assertBlock(distances[code], brute.block(counts(Block.MAN, digits), Block.MAN), "suit code " + code);
      }
    }
    Random random = new Random(11);
    for (int checked = 0; checked < RANDOM_SUIT_CODES; ) {
      int code = random.nextInt(distances.length);
      if (ShapeTableGenerator.digits(code, digits) <= HandShape.MAX_SIZE) {
        assertBlock(distances[code], brute.block(counts(Block.MAN, digits), Block.MAN), "suit code " + code);
        checked++;
      } else {
        Assert.assertEquals(distances[code], -1L, "suit code " + code);
      }
    }
  }

  @Test
  public void honorDistancesMatchEnumeration() {
    long[] distances = ShapeTableGenerator.blockDistances(Block.HONOR);
    BruteShape brute = new BruteShape(TileSet.FULL);
    int[] digits = new int[Block.HONOR.size()];
    for (int code = 0; code < distances.length; code++) {
      if (ShapeTableGenerator.digits(code, digits) <= HandShape.MAX_SIZE) {
        assertBlock(distances[code], brute.block(counts(Block.HONOR, digits), Block.HONOR), "honor code " + code);
      }
    }
  }

  /** 三人打ちの萬子（1m と 9m だけ）の 25 通りの d は、2m〜8m を抜いた三人打ちの牌で総当たりした値と一致する。 */
  @Test
  public void terminalManDistancesMatchEnumerationWithoutTwoToEightMan() {
    BruteShape brute = new BruteShape(TileSet.SANMA);
    for (int one = 0; one <= Tiles.COPIES; one++) {
      for (int nine = 0; nine <= Tiles.COPIES; nine++) {
        int[] counts = new int[Tiles.KINDS];
        counts[Tiles.M1] = one;
        counts[Tiles.M9] = nine;
        long distances = ShapeTableGenerator.distances(new int[] {one, nine}, false);
        assertBlock(distances, brute.block(counts, Block.MAN), "1m x " + one + ", 9m x " + nine);
      }
    }
  }

  /** 1m 4 枚から面子 2 つを作るには、2m・3m を引ければ 2 枚で足りる（111m ＋ 123m）。三人打ちでは 1m が孤立するので 3 枚要る（111m ＋ 999m）。 */
  @Test
  public void fourOneManNeedThreeTilesForTwoMeldsInSanma() {
    long suit = ShapeTableGenerator.distances(new int[] {4, 0, 0, 0, 0, 0, 0, 0, 0}, true);
    long terminal = ShapeTableGenerator.distances(new int[] {4, 0}, false);
    Assert.assertEquals(Distances.get(suit, 2, 0), 2);
    Assert.assertEquals(Distances.get(terminal, 2, 0), 3);
    Assert.assertEquals(Distances.get(terminal, 1, 0), 0);
    Assert.assertEquals(Distances.get(terminal, 2, 1), ShapeTableGenerator.IMPOSSIBLE);
  }

  @Test
  public void resourceIsTheGeneratorOutput() throws IOException {
    Object[] sections = ShapeTableGenerator.generate();
    byte[] generated = ShapeTableGenerator.encode(sections);
    if (!Arrays.equals(resource(), generated)) {
      Path fresh = Path.of("target", "shape", "hand-shape.bin");
      ShapeTableGenerator.write(generated, fresh);
      Assert.fail(
          "src/main/resources" + ShapeSection.RESOURCE + " differs from the generator output; the generated table is at "
              + fresh.toAbsolutePath() + " (" + lengths(sections) + ")");
    }
  }

  private static void assertBlock(long distances, int[] expected, String context) {
    for (int melds = 0; melds < ShapeSection.MELD_TARGETS; melds++) {
      for (int pairs = 0; pairs <= 1; pairs++) {
        int value = Math.min(expected[melds * 2 + pairs], ShapeTableGenerator.IMPOSSIBLE);
        Assert.assertEquals(Distances.get(distances, melds, pairs), value, context + " melds " + melds + " pairs " + pairs);
      }
    }
  }

  private static int[] counts(Block block, int[] digits) {
    int[] counts = new int[Tiles.KINDS];
    System.arraycopy(digits, 0, counts, block.first(), digits.length);
    return counts;
  }

  private static byte[] resource() throws IOException {
    try (InputStream in = ShapeTableGeneratorTest.class.getResourceAsStream(ShapeSection.RESOURCE)) {
      if (in == null) {
        return null;
      }
      try (GZIPInputStream unzipped = new GZIPInputStream(in)) {
        return unzipped.readAllBytes();
      }
    }
  }

  private static String lengths(Object[] sections) {
    StringBuilder text = new StringBuilder();
    for (ShapeSection section : ShapeSection.values()) {
      text.append(section).append('=').append(Array.getLength(sections[section.ordinal()])).append(' ');
    }
    return text.toString().trim();
  }
}
