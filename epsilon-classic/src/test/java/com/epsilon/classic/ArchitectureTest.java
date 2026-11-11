package com.epsilon.classic;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.policy.Series;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import com.google.gson.Gson;
import java.util.EnumSet;
import java.util.Set;
import org.testng.annotations.Test;

/** Nano と Major の定数と、チェックポイントのヘッダーの値を JSON に書いて読み戻せることを確かめる。 */
public class ArchitectureTest {

  @Test
  public void architecturesKeepTheirSeriesAndHiddenWidths() {
    assertEquals(Architecture.NANO.series(), Series.NANO);
    assertEquals(Architecture.NANO.hidden(), 256);
    assertFalse(Architecture.NANO.publicHistory());
    assertEquals(Architecture.MAJOR.series(), Series.MAJOR);
    assertEquals(Architecture.MAJOR.hidden(), 384);
    assertTrue(Architecture.MAJOR.publicHistory());
  }

  @Test
  public void scopesNameTheArchitecturesThatHoldAConstant() {
    Set<Architecture> all = ArchitectureScoped.scope();
    assertEquals(all, EnumSet.allOf(Architecture.class));
    Set<Architecture> nano = ArchitectureScoped.scope(Architecture.NANO);
    assertEquals(nano, EnumSet.of(Architecture.NANO));
    assertThrows(UnsupportedOperationException.class, () -> nano.add(Architecture.MAJOR));
  }

  @Test
  public void headersRoundTripThroughJson() {
    ClassicHeader header =
        new ClassicHeader(Architecture.MAJOR, Variant.TENHOU_4P, RankUtility.AVERAGE_RANK, "0123456789abcdef", 42);
    Gson gson = new Gson();
    assertEquals(gson.fromJson(gson.toJson(header), ClassicHeader.class), header);
    assertThrows(
        IllegalArgumentException.class,
        () -> new ClassicHeader(Architecture.NANO, Variant.TENHOU_4P, RankUtility.TENHOU, "x", -1));
    assertThrows(
        NullPointerException.class,
        () -> new ClassicHeader(Architecture.NANO, Variant.TENHOU_4P, null, "x", 0));
  }
}
