package com.epsilon.prism.luck;

import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.core.CrnSeeds;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 帳簿の確率的丸めに使う乱数の式を固定する。行ごとに 1 回求めるキー（{@link LedgerRounding#key}）を手と列で伸ばした乱数は、
 * どの入力でも unit(derive(コホートの種, LEDGER, 半荘, 判断, 手·17 + 列)) とビットまで同じになる。この derive は、座標ごとに mix64 で
 * 混ぜ直す定義を LEDGER のソルトで書き下した値とも同じ。行に書くときは、その手の 17 列だけを、列ごとの乱数で丸めて書く。
 */
public class LedgerRoundingTest {
  private static final int F = LedgerColumn.COUNT;

  /** LEDGER のソルト（"LEDG"）。値が変わると行の帳簿の丸めが変わるので、ここに書き写して固定する。 */
  private static final long LEDGER_SALT = 0x4c454447L;

  /** 座標ごとに mix64 で混ぜ直す定義を書き下した、5 座標の derive。 */
  private static long formerDerive(long base, long salt, long a, long b, long c) {
    long first = SeedMixer.mix64(SeedMixer.mix64(base ^ salt) ^ SeedMixer.mix64(a));
    long second = SeedMixer.mix64(first ^ SeedMixer.mix64(b));
    return SeedMixer.mix64(second ^ SeedMixer.mix64(c));
  }

  /** 書き下した derive で求めた、丸めに使う乱数。 */
  private static double formerCoin(long seed, long gameId, long decisionId, int action, int column) {
    return CrnSeeds.unit(formerDerive(seed, LEDGER_SALT, gameId, decisionId, (long) action * F + column));
  }

  /** 端の値（0・−1・最大・最小）と乱数を混ぜて、64 ビットの値を 1 つ返す。 */
  private static long anyLong(SplittableRandom random) {
    return switch (random.nextInt(8)) {
      case 0 -> 0L;
      case 1 -> -1L;
      case 2 -> Long.MAX_VALUE;
      case 3 -> Long.MIN_VALUE;
      case 4 -> random.nextInt(1 << 20);
      default -> random.nextLong();
    };
  }

  @Test
  public void theCoinIsTheLedgerDeriveOfTheSeedTheGameTheDecisionAndTheCell() {
    var random = new SplittableRandom(20261110);
    for (int i = 0; i < 200_000; i++) {
      long seed = anyLong(random), game = anyLong(random), decision = anyLong(random);
      int action = random.nextInt(Decisions.MAX_LEGAL), column = random.nextInt(F);
      double coin = LedgerRounding.coin(LedgerRounding.key(seed, game, decision), action, column);
      double derived =
          CrnSeeds.unit(CrnSeeds.derive(seed, CrnSeeds.Domain.LEDGER, game, decision, (long) action * F + column));
      double former = formerCoin(seed, game, decision, action, column);
      String where = "seed " + seed + " game " + game + " decision " + decision + " action " + action + " column " + column;
      Assert.assertEquals(Double.doubleToRawLongBits(coin), Double.doubleToRawLongBits(derived), where);
      Assert.assertEquals(Double.doubleToRawLongBits(coin), Double.doubleToRawLongBits(former), where);
    }
  }

  /** 座標の多い derive は、座標の少ない derive の値を extend で伸ばした値と同じで、書き下した定義とも同じ。 */
  @Test
  public void deriveExtendsTheShorterDerive() {
    var random = new SplittableRandom(17);
    for (int i = 0; i < 100_000; i++) {
      long base = anyLong(random), a = anyLong(random), b = anyLong(random), c = anyLong(random);
      for (CrnSeeds.Domain domain : CrnSeeds.Domain.values()) {
        long one = CrnSeeds.derive(base, domain, a);
        Assert.assertEquals(CrnSeeds.derive(base, domain, a, b), CrnSeeds.extend(one, b));
        Assert.assertEquals(
            CrnSeeds.derive(base, domain, a, b, c), CrnSeeds.extend(CrnSeeds.extend(one, b), c));
        Assert.assertEquals(CrnSeeds.derive(base, domain, a, b), SeedMixer.mix64(one ^ SeedMixer.mix64(b)));
      }
      Assert.assertEquals(
          CrnSeeds.derive(base, CrnSeeds.Domain.LEDGER, a, b, c), formerDerive(base, LEDGER_SALT, a, b, c));
    }
  }

  /**
   * 行に書くと、手 a の 17 列だけが書かれ、列 c には書き下した derive の乱数で丸めた値が入る。ほかの手の列には触らない。キーはコホートの
   * 種・半荘・判断で決まり、どれが違っても乱数が変わる。
   */
  @Test
  public void writeRoundsEachCellOfTheActionWithItsOwnCoin() {
    var random = new SplittableRandom(5);
    int actions = 6;
    for (int trial = 0; trial < 2_000; trial++) {
      long seed = anyLong(random), game = anyLong(random), decision = anyLong(random);
      int action = random.nextInt(actions);
      double[] terms = new double[F];
      for (int f = 0; f < F; f++)
        terms[f] = switch (random.nextInt(4)) {
          case 0 -> 0;
          case 1 -> random.nextGaussian() * 1e-6;
          default -> random.nextGaussian() * Math.scalb(1.0, random.nextInt(-6, 6));
        };
      float[] ledger = new float[actions * F];
      Arrays.fill(ledger, Float.NaN);
      LedgerRounding.write(LedgerRounding.key(seed, game, decision), action, terms, ledger);
      for (int a = 0; a < actions; a++)
        for (int f = 0; f < F; f++) {
          float got = ledger[a * F + f];
          if (a != action) {
            Assert.assertTrue(Float.isNaN(got), "touched action " + a);
            continue;
          }
          float expected = LedgerRounding.round(terms[f], formerCoin(seed, game, decision, action, f));
          Assert.assertEquals(Float.floatToRawIntBits(got), Float.floatToRawIntBits(expected), "column " + f);
        }
    }
    for (int f = 0; f < F; f++) {
      double coin = LedgerRounding.coin(LedgerRounding.key(1, 2, 3), 0, f);
      Assert.assertNotEquals(LedgerRounding.coin(LedgerRounding.key(1, 2, 4), 0, f), coin, "decision, column " + f);
      Assert.assertNotEquals(LedgerRounding.coin(LedgerRounding.key(1, 3, 3), 0, f), coin, "game, column " + f);
      Assert.assertNotEquals(LedgerRounding.coin(LedgerRounding.key(2, 2, 3), 0, f), coin, "seed, column " + f);
    }
  }
}
