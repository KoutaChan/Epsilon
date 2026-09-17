package com.epsilon.engine;

import java.util.Objects;

/** 一人の和了に対する支払い方法。 */
public sealed interface WinPayment permits WinPayment.Ron, WinPayment.Tsumo {

  /**
   * 全支払者から受け取る点数の合計を返す。
   *
   * @return 本場・供託を含まない和了点
   */
  int total();

  /**
   * 放銃者一人が支払うロン和了。
   *
   * <dl>
   *   <dt>{@code points}
   *   <dd>放銃者が支払う点数
   * </dl>
   */
  final class Ron implements WinPayment {
    private final int points;

    public Ron(int points) {
      this.points = points;
    }

    public int points() {
      return points;
    }

    @Override
    public int total() {
      return points;
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Ron ron && points == ron.points;
    }

    @Override
    public int hashCode() {
      return Integer.hashCode(points);
    }

    @Override
    public String toString() {
      return "Ron[points=" + points + ']';
    }
  }

  /** 複数の他家が支払うツモ和了。 */
  sealed interface Tsumo extends WinPayment permits DealerTsumo, ChildTsumo {}

  /**
   * 親ツモ。三人の子がそれぞれ同額を支払う。
   *
   * <dl>
   *   <dt>{@code each}
   *   <dd>各子が支払う点数
   * </dl>
   */
  final class DealerTsumo implements Tsumo {
    private final int each;

    public DealerTsumo(int each) {
      this.each = each;
    }

    public int each() {
      return each;
    }

    @Override
    public int total() {
      return each * 3;
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof DealerTsumo tsumo && each == tsumo.each;
    }

    @Override
    public int hashCode() {
      return Integer.hashCode(each);
    }

    @Override
    public String toString() {
      return "DealerTsumo[each=" + each + ']';
    }
  }

  /**
   * 子ツモ。親一人と子二人がそれぞれの額を支払う。
   *
   * <dl>
   *   <dt>{@code fromDealer}
   *   <dd>親が支払う点数
   *   <dt>{@code fromChild}
   *   <dd>各子が支払う点数
   * </dl>
   */
  final class ChildTsumo implements Tsumo {
    private final int fromDealer;
    private final int fromChild;

    public ChildTsumo(int fromDealer, int fromChild) {
      this.fromDealer = fromDealer;
      this.fromChild = fromChild;
    }

    public int fromDealer() {
      return fromDealer;
    }

    public int fromChild() {
      return fromChild;
    }

    @Override
    public int total() {
      return fromDealer + fromChild * 2;
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof ChildTsumo tsumo
          && fromDealer == tsumo.fromDealer
          && fromChild == tsumo.fromChild;
    }

    @Override
    public int hashCode() {
      return Objects.hash(fromDealer, fromChild);
    }

    @Override
    public String toString() {
      return "ChildTsumo[fromDealer=" + fromDealer + ", fromChild=" + fromChild + ']';
    }
  }
}
