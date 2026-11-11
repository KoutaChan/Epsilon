package com.epsilon.grp;

import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 順列の表を確かめる。順列の数・辞書順・番号の往復、終局の点で決まる順列（同点なら起家に近い席が上）、周辺確率。 */
public class RankPermutationsTest {

  @Test
  public void tablesListEveryPermutationInLexicographicOrder() {
    for (Variant variant : Variant.values()) {
      RankPermutations table = RankPermutations.of(variant);
      int seats = variant.seats();
      Assert.assertEquals(table.count(), seats == 4 ? 24 : 6);
      int[] previous = null;
      for (int permutation = 0; permutation < table.count(); permutation++) {
        int[] ranks = ranks(table, permutation, seats);
        Assert.assertEquals(table.index(ranks), permutation, Arrays.toString(ranks));
        if (previous != null) {
          Assert.assertTrue(Arrays.compare(previous, ranks) < 0, Arrays.toString(previous) + Arrays.toString(ranks));
        }
        previous = ranks;
      }
      int[] identity = new int[seats];
      int[] reversed = new int[seats];
      for (int seat = 0; seat < seats; seat++) {
        identity[seat] = seat;
        reversed[seat] = seats - 1 - seat;
      }
      Assert.assertEquals(ranks(table, 0, seats), identity);
      Assert.assertEquals(ranks(table, table.count() - 1, seats), reversed);
    }
  }

  /** {@code rotate(π, d)} の相対席 r の順位は、π の相対席 r − d の順位。d ずらしてから S − d ずらすと元に戻る。 */
  @Test
  public void rotationRenumbersTheSeats() {
    for (Variant variant : Variant.values()) {
      RankPermutations table = RankPermutations.of(variant);
      int seats = variant.seats();
      for (int permutation = 0; permutation < table.count(); permutation++) {
        Assert.assertEquals(table.rotate(permutation, 0), permutation);
        for (int shift = 1; shift < seats; shift++) {
          int rotated = table.rotate(permutation, shift);
          for (int seat = 0; seat < seats; seat++) {
            Assert.assertEquals(table.rank(rotated, seat), table.rank(permutation, Math.floorMod(seat - shift, seats)));
          }
          Assert.assertEquals(table.rotate(rotated, seats - shift), permutation);
        }
      }
    }
  }

  @Test
  public void indexRejectsNonPermutations() {
    RankPermutations table = RankPermutations.of(Variant.TENHOU_4P);
    Assert.assertThrows(IllegalArgumentException.class, () -> table.index(new int[] {0, 1, 1, 3}));
    Assert.assertThrows(IllegalArgumentException.class, () -> table.index(new int[] {0, 1, 2, 4}));
    Assert.assertThrows(IllegalArgumentException.class, () -> table.index(new int[] {0, 1, 2}));
  }

  @Test
  public void tiesGoToTheSeatNearestEastOne() {
    RankPermutations four = RankPermutations.of(Variant.TENHOU_4P);
    // 全員が同点なら絶対席の順。親が絶対席 2 なら相対席は絶対席 2・3・0・1。
    Assert.assertEquals(four.realized(new int[] {25_000, 25_000, 25_000, 25_000}, 2), four.index(new int[] {2, 3, 0, 1}));
    // 絶対席 1 と 3 が 30,000 点、0 と 2 が 20,000 点。順位は 1・3・0・2 の順で、親が絶対席 3 なら相対席は 3・0・1・2。
    Assert.assertEquals(four.realized(new int[] {20_000, 30_000, 20_000, 30_000}, 3), four.index(new int[] {1, 2, 0, 3}));
    RankPermutations three = RankPermutations.of(Variant.TENHOU_3P);
    // 絶対席 0 と 2 が同点のトップ。親が絶対席 1 なら相対席は 1・2・0。
    Assert.assertEquals(three.realized(new int[] {40_000, 25_000, 40_000}, 1), three.index(new int[] {2, 1, 0}));
  }

  @Test
  public void realizedMatchesPlacementInRelativeSeats() {
    for (Variant variant : Variant.values()) {
      RankPermutations table = RankPermutations.of(variant);
      int seats = variant.seats();
      Random random = new Random(7 + variant.ordinal());
      for (int trial = 0; trial < 5_000; trial++) {
        int[] scores = new int[seats];
        for (int seat = 0; seat < seats; seat++) {
          scores[seat] = (random.nextInt(9) - 2) * 10_000;
        }
        int dealer = random.nextInt(seats);
        int[] ranks = new int[seats];
        for (int seat = 0; seat < seats; seat++) {
          ranks[seat] = Placement.rank(scores, (dealer + seat) % seats);
        }
        Assert.assertEquals(table.realized(scores, dealer), table.index(ranks), Arrays.toString(scores) + " " + dealer);
      }
    }
  }

  @Test
  public void marginalsOfAPermutationAreItsMatrix() {
    RankPermutations table = RankPermutations.of(Variant.TENHOU_4P);
    double[] probabilities = new double[table.count()];
    int permutation = table.index(new int[] {3, 0, 2, 1});
    probabilities[permutation] = 1;
    double[] marginals = new double[16];
    table.marginals(probabilities, marginals);
    for (int seat = 0; seat < 4; seat++) {
      for (int rank = 0; rank < 4; rank++) {
        Assert.assertEquals(marginals[seat * 4 + rank], table.rank(permutation, seat) == rank ? 1.0 : 0.0, 0.0);
      }
    }
  }

  /** 絶対席の周辺確率は、相対席の周辺確率の行を親の席の分だけ回したもの。 */
  @Test
  public void absoluteMarginalsRotateTheRelativeRows() {
    for (Variant variant : Variant.values()) {
      RankPermutations table = RankPermutations.of(variant);
      int seats = variant.seats();
      Random random = new Random(seats);
      double[] probabilities = new double[table.count()];
      double total = 0;
      for (int permutation = 0; permutation < probabilities.length; permutation++) {
        probabilities[permutation] = random.nextDouble();
        total += probabilities[permutation];
      }
      for (int permutation = 0; permutation < probabilities.length; permutation++) {
        probabilities[permutation] /= total;
      }
      double[] relative = new double[seats * seats];
      table.marginals(probabilities, relative);
      double[] absolute = new double[seats * seats];
      for (int dealer = 0; dealer < seats; dealer++) {
        table.marginals(probabilities, dealer, absolute);
        for (int seat = 0; seat < seats; seat++) {
          for (int rank = 0; rank < seats; rank++) {
            Assert.assertEquals(
                absolute[Seats.absolute(dealer, seat, seats) * seats + rank], relative[seat * seats + rank], 0.0,
                variant + " dealer " + dealer + " seat " + seat);
          }
        }
      }
    }
  }

  private static int[] ranks(RankPermutations table, int permutation, int seats) {
    int[] ranks = new int[seats];
    for (int seat = 0; seat < seats; seat++) {
      ranks[seat] = table.rank(permutation, seat);
    }
    return ranks;
  }
}
