package com.epsilon.engine;

import java.util.Objects;

/** 1局精算後の半荘進行。 */
public sealed interface RoundTransition
    permits RoundTransition.NextRound, RoundTransition.HanchanFinished {

  /**
   * 次局へ進む。
   *
   * <dl>
   *   <dt>{@code kyokuIndex}
   *   <dd>次局の通算局インデックス
   *   <dt>{@code oya}
   *   <dd>次局の親席番号（0-3）
   *   <dt>{@code honba}
   *   <dd>次局開始時の本場数
   *   <dt>{@code kyotakuCount}
   *   <dd>次局へ持ち越す供託本数
   * </dl>
   */
  final class NextRound implements RoundTransition {
    private final int kyokuIndex;
    private final int oya;
    private final int honba;
    private final int kyotakuCount;

    public NextRound(int kyokuIndex, int oya, int honba, int kyotakuCount) {
      this.kyokuIndex = kyokuIndex;
      this.oya = oya;
      this.honba = honba;
      this.kyotakuCount = kyotakuCount;
    }

    public int kyokuIndex() {
      return kyokuIndex;
    }

    public int oya() {
      return oya;
    }

    public int honba() {
      return honba;
    }

    public int kyotakuCount() {
      return kyotakuCount;
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof NextRound next
          && kyokuIndex == next.kyokuIndex
          && oya == next.oya
          && honba == next.honba
          && kyotakuCount == next.kyotakuCount;
    }

    @Override
    public int hashCode() {
      return Objects.hash(kyokuIndex, oya, honba, kyotakuCount);
    }

    @Override
    public String toString() {
      return "NextRound[kyokuIndex="
          + kyokuIndex
          + ", oya="
          + oya
          + ", honba="
          + honba
          + ", kyotakuCount="
          + kyotakuCount
          + ']';
    }
  }

  /** 半荘が終了した。 */
  enum HanchanFinished implements RoundTransition {
    /** 半荘終了を表す唯一の値。 */
    INSTANCE
  }
}
