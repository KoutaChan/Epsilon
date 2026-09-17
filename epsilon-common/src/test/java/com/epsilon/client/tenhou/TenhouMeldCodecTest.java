package com.epsilon.client.tenhou;

import com.epsilon.core.Meld;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 既知のm値を直接使い、デコーダー自身のビット式で期待入力を作り直さない。 */
public class TenhouMeldCodecTest {
  @DataProvider
  public Object[][] melds() {
    return new Object[][] {
      // 赤5mを下家から鳴いたポン。
      {
        6249,
        Meld.Type.PON,
        Meld.RelativeSource.SHIMOCHA,
        Meld.AkaSource.CALLED_TILE,
        16,
        new int[] {17, 18}
      },
      // 対面の通常5mを鳴き、自手牌の赤5mを消費したポン。
      {
        6762,
        Meld.Type.PON,
        Meld.RelativeSource.TOIMEN,
        Meld.AkaSource.CONSUMED_HAND_TILE,
        17,
        new int[] {16, 18}
      },
      // 元のポンへ赤5mを追加する加槓。
      {
        6162,
        Meld.Type.KAKAN,
        Meld.RelativeSource.TOIMEN,
        Meld.AkaSource.ADDED_KAN_TILE,
        17,
        new int[] {16}
      },
      // 4p・赤5pを消費し、6pの実牌59を上家からチー。
      {
        33167,
        Meld.Type.CHI,
        Meld.RelativeSource.KAMICHA,
        Meld.AkaSource.CONSUMED_HAND_TILE,
        59,
        new int[] {49, 52}
      },
      {
        10240,
        Meld.Type.ANKAN,
        Meld.RelativeSource.NONE,
        Meld.AkaSource.NONE,
        -1,
        new int[] {40, 41, 42, 43}
      },
      {
        32257,
        Meld.Type.DAIMINKAN,
        Meld.RelativeSource.SHIMOCHA,
        Meld.AkaSource.NONE,
        126,
        new int[] {124, 125, 127}
      }
    };
  }

  @Test(dataProvider = "melds")
  public void calledTileAndConsumedTilesRetainTheirSeparateRedProvenance(
      int code,
      Meld.Type kind,
      Meld.RelativeSource source,
      Meld.AkaSource redSource,
      int called,
      int[] consumed) {
    var decoded = TenhouMeldDecoder.decode(code);
    Assert.assertEquals(decoded.meld().type(), kind);
    Assert.assertEquals(decoded.meld().relativeSource(), source);
    Assert.assertEquals(decoded.meld().akaSource(), redSource);
    Assert.assertEquals(decoded.calledPhysicalTileId(), called);
    Assert.assertEquals(decoded.consumedPhysicalTileIds(), consumed);
  }

  @Test
  public void chiFromAnImpossibleSeatIsRejectedAtTheCodecBoundary() {
    Assert.expectThrows(IllegalArgumentException.class, () -> TenhouMeldDecoder.decode(4));
  }
}
