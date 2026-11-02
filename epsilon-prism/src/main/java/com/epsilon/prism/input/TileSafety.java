package com.epsilon.prism.input;

import com.epsilon.core.GameState;
import com.epsilon.prism.input.InputSchema.PublicEvent;
import com.epsilon.prism.input.InputSchema.Safety;
import java.util.Arrays;

/**
 * 牌の安全のうち、振聴の規則だけで厳密に決まる 3 類（{@link Safety}）。席ごとに牌のマスクを 3 本持つ。判断ごとに 1 回作り、割り当てをしない。
 *
 * <ul>
 *   <li>現物: その席の河にある（捨て牌の振聴で、その席はその牌でロンできない）。鳴かれた打牌も河の牌に数える。
 *   <li>立直後の通過: その席の立直宣言より後に、他の席が捨てた（立直後の見逃しは局の終わりまで振聴）。
 *   <li>同巡の通過: その席の最後の打牌より後に、他の席が捨てた（同巡の見逃しは次の自分の打牌まで振聴）。
 * </ul>
 *
 * <p>どれも、捨てた牌を鳴かれた場合を含めて「その席がロンを見送った牌」の集まり。通過は打牌だけを数え、加槓の牌（槍槓の見送り）は数えない。
 */
final class TileSafety {

  private static final int SEATS = GameState.NUM_PLAYERS;

  private final long[] river = new long[SEATS];
  private final long[] afterRiichi = new long[SEATS];
  private final long[] thisCycle = new long[SEATS];

  /**
   * 出来事から 3 類のマスクを作る。
   *
   * @param events O の出来事
   */
  void load(EventTable events) {
    Arrays.fill(river, 0L);
    Arrays.fill(afterRiichi, 0L);
    Arrays.fill(thisCycle, 0L);
    for (int e = 0, count = events.events(); e < count; e++) {
      if (events.kind(e) != PublicEvent.DISCARD) {
        continue;
      }
      int author = events.author(e);
      long bit = 1L << events.tile(e);
      for (int seat = 0; seat < SEATS; seat++) {
        if (seat == author) {
          continue;
        }
        thisCycle[seat] |= bit;
        int declaration = events.riichiDeclaration(seat);
        if (declaration >= 0 && declaration < e) {
          afterRiichi[seat] |= bit;
        }
      }
      river[author] |= bit;
      thisCycle[author] = 0L;
    }
  }

  /**
   * 席から見て、その類に当たる牌のマスクを返す。複数に当たる牌は {@link Safety} の宣言の順で先の類にだけ入れるので、3 つの類のマスクは
   * 重ならない。
   *
   * @param seat 相対席
   * @param safety 安全の類
   * @return 牌のマスク（物理の番号）
   */
  long mask(int seat, Safety safety) {
    return switch (safety) {
      case GENBUTSU -> river[seat];
      case PASSED_AFTER_RIICHI -> afterRiichi[seat] & ~river[seat];
      case PASSED_THIS_CYCLE -> thisCycle[seat] & ~(river[seat] | afterRiichi[seat]);
    };
  }

  /** 席の河の牌のマスク。 */
  long river(int seat) {
    return river[seat];
  }
}
