package com.epsilon.prism.input;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.input.EventTable.PublicEvent;
import java.util.Arrays;

/**
 * 牌の安全度のうち、振聴のルールだけで厳密に決まる 3 種類（{@link Safety}）を求める。席ごとに牌のマスクを 3 つ持つ。判断ごとに 1 回
 * 作り、メモリを割り当てない。
 *
 * <ul>
 *   <li>現物: その席の河にある牌。捨て牌で振聴なので、その席はその牌でロンできない。鳴かれた打牌も河の牌に数える。
 *   <li>立直後に通った牌: その席が立直を宣言した後に、ほかの席が捨てた牌。立直後の見逃しは局の終わりまで振聴。
 *   <li>同巡に通った牌: その席の最後の打牌より後に、ほかの席が捨てた牌。同巡の見逃しは次の自分の打牌まで振聴。
 * </ul>
 *
 * <p>どれも、捨てた牌が鳴かれた場合を含めて、その席がロンを見送った牌の集まり。通った牌には打牌だけを数え、加槓の牌（槍槓の見送り）は
 * 数えない。現物と立直後に通った牌は {@link SeatFacts} の河の種類と立直後に通った種類から取り、同巡に通った牌だけをイベントから数える。
 */
public final class TileSafety {

  /** その席（他家）に対する牌の安全度。振聴のルールで厳密に決まるものだけを持つ。複数に当てはまる牌は、ここに並べた順で先の種類にする。 */
  public enum Safety {
    /** その席の河にある。 */
    GENBUTSU,
    /** その席の立直より後に通った。 */
    PASSED_AFTER_RIICHI,
    /** その席の最後の打牌より後に通った。 */
    PASSED_THIS_CYCLE
  }

  private final long[] river;
  private final long[] afterRiichi;
  private final long[] thisCycle;

  public TileSafety(int seats) {
    river = new long[seats];
    afterRiichi = new long[seats];
    thisCycle = new long[seats];
  }

  /**
   * 3 種類のマスクを作る。
   *
   * @param events 判断の記録の公開イベント
   * @param seats 席ごとの公開情報（同じ判断の記録から作ったもの）
   */
  public void load(EventTable events, SeatFacts seats) {
    for (int seat = 0; seat < river.length; seat++) {
      river[seat] = seats.riverKinds(seat);
      afterRiichi[seat] = seats.riichiSafe(seat);
    }
    Arrays.fill(thisCycle, 0L);
    for (int e = 0, count = events.events(); e < count; e++) {
      if (events.kind(e) != PublicEvent.DISCARD) {
        continue;
      }
      int author = events.author(e);
      long bit = Tiles.bit(events.tile(e));
      for (int seat = 0; seat < thisCycle.length; seat++) {
        thisCycle[seat] |= bit;
      }
      thisCycle[author] = 0L;
    }
  }

  /**
   * その席に対して、指定の種類に当たる牌のマスクを返す。複数に当てはまる牌は {@link Safety} に並べた順で先の種類にだけ入れるので、
   * 3 種類のマスクは重ならない。
   *
   * @param seat 相対席
   * @param safety 安全度の種類
   * @return 牌の種類のマスク
   */
  public long mask(int seat, Safety safety) {
    return switch (safety) {
      case GENBUTSU -> river[seat];
      case PASSED_AFTER_RIICHI -> afterRiichi[seat] & ~river[seat];
      case PASSED_THIS_CYCLE -> thisCycle[seat] & ~(river[seat] | afterRiichi[seat]);
    };
  }

  /** その席の河にある牌の種類のマスク。 */
  public long river(int seat) {
    return river[seat];
  }
}
