package com.epsilon.prism.core;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.SeatView;
import java.lang.reflect.Proxy;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 判断する席の手の形は、その席の面子だけで決まる。暗槓だけの手は門前。 */
public class OpennessTest {

  /** 自分の席と各席の面子だけを返す卓（手の形を求めるときに読む 3 つのメソッドのほかは呼ばない）。 */
  private static SeatView view(int seat, List<List<Meld>> melds) {
    return (SeatView)
        Proxy.newProxyInstance(
            SeatView.class.getClassLoader(),
            new Class<?>[] {SeatView.class},
            (proxy, method, args) ->
                switch (method.getName()) {
                  case "seat" -> seat;
                  case "meldCount" -> melds.get((int) args[0]).size();
                  case "meld" -> melds.get((int) args[0]).get((int) args[1]).id();
                  default -> throw new UnsupportedOperationException(method.getName());
                });
  }

  private static Openness of(Meld... melds) {
    // 自分は席 1、ほかの席はポンを持つ（他家の面子は手の形に入らない）
    Meld pon = Meld.pon(31, Meld.RelativeSource.TOIMEN, Meld.AkaSource.NONE);
    return Openness.of(view(1, List.of(List.of(pon), List.of(melds), List.of(pon, pon), List.of())));
  }

  @Test
  public void onlyCallsOpenTheHand() {
    Meld pon = Meld.pon(5, Meld.RelativeSource.KAMICHA, Meld.AkaSource.NONE);
    Assert.assertEquals(of(), Openness.CLOSED);
    Assert.assertEquals(of(Meld.ankan(3, false)), Openness.CLOSED);
    Assert.assertEquals(of(Meld.ankan(3, false), Meld.ankan(27, false)), Openness.CLOSED);
    Assert.assertEquals(of(Meld.chi(0, 1, Meld.AkaSource.NONE)), Openness.OPEN);
    Assert.assertEquals(of(pon), Openness.OPEN);
    Assert.assertEquals(of(Meld.daiminkan(9, Meld.RelativeSource.SHIMOCHA, Meld.AkaSource.NONE)), Openness.OPEN);
    Assert.assertEquals(of(Meld.kakan(pon, false)), Openness.OPEN);
    Assert.assertEquals(of(Meld.ankan(3, false), Meld.chi(10, 12, Meld.AkaSource.NONE)), Openness.OPEN);
    Assert.assertEquals(Openness.COUNT, 2);
    for (Openness hand : Openness.values()) Assert.assertSame(Openness.of(hand.ordinal()), hand);
  }
}
