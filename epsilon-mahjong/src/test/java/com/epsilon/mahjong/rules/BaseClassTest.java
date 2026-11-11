package com.epsilon.mahjong.rules;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 基本点の表を確かめる。値は昇順に並び、値から表を引くと同じ項目が返り、満貫未満の値を全て含む。 */
public class BaseClassTest {

  @Test
  public void basesAscendAndRoundTrip() {
    int previous = 0;
    for (BaseClass base : BaseClass.values()) {
      Assert.assertTrue(base.base() > previous, base.name());
      Assert.assertSame(BaseClass.of(base.base()), base);
      previous = base.base();
    }
  }

  /** 満貫未満の値は、1〜4 翻での 符 × 2^(2 ＋ 翻) のうち 2000 未満のもの全て。20 符（平和ツモ）と 25 符（七対子）は 2 翻からしか無い。 */
  @Test
  public void belowManganHoldsEveryHanFuProduct() {
    TreeSet<Integer> expected = new TreeSet<>();
    int[] fus = {20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110};
    for (int han = 1; han <= 4; han++) {
      for (int fu : fus) {
        int base = fu << (2 + han);
        if (base < BaseClass.MANGAN.base() && (fu > 25 || han >= 2)) {
          expected.add(base);
        }
      }
    }
    List<Integer> actual = new ArrayList<>();
    for (BaseClass base : BaseClass.values()) {
      if (base.base() < BaseClass.MANGAN.base()) {
        actual.add(base.base());
      }
    }
    Assert.assertEquals(actual, new ArrayList<>(expected));
  }

  @Test
  public void yakumanMultiplesUpToSix() {
    BaseClass[] yakuman = {
      BaseClass.YAKUMAN, BaseClass.YAKUMAN_2, BaseClass.YAKUMAN_3, BaseClass.YAKUMAN_4, BaseClass.YAKUMAN_5,
      BaseClass.YAKUMAN_6
    };
    for (int multiple = 1; multiple <= yakuman.length; multiple++) {
      Assert.assertSame(BaseClass.of(8_000 * multiple), yakuman[multiple - 1]);
    }
  }

  @Test
  public void valuesOutsideTheTableAreRejected() {
    for (int base : new int[] {0, 160, 200, 1_000, 2_001, 2_240, 56_000}) {
      Assert.assertThrows(IllegalArgumentException.class, () -> BaseClass.of(base));
    }
  }
}
