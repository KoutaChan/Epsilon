package com.epsilon.prism.luck;

import com.epsilon.mahjong.rules.Variant;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** {@link LedgerLawTest} の検定を大きな件数で回す。 */
public class LedgerLawBench {

  @DataProvider
  public Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test(dataProvider = "variants", timeOut = 3_600_000)
  public void drawsAreUniformOverThePool(Variant variant) {
    LedgerLawTest.drawsAreUniform(variant, 60_000);
  }

  @Test(dataProvider = "variants", timeOut = 3_600_000)
  public void poolMatchesTheUnseenSet(Variant variant) {
    LedgerLawTest.poolMatches(variant, 9, 2, 4000);
  }

  @Test(dataProvider = "variants", timeOut = 3_600_000)
  public void conditionalMeanIsZeroUnderReshuffle(Variant variant) {
    LedgerLawTest.conditionalMean(variant, new int[] {8, 8, 6, 6}, 8000);
  }

  @Test(dataProvider = "variants", timeOut = 3_600_000)
  public void uraTermIsCentredUnderReshuffle(Variant variant) {
    LedgerLawTest.uraCentred(variant, 8, 2000);
  }
}
