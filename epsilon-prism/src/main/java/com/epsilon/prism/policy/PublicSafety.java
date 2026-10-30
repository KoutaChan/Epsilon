package com.epsilon.prism.policy;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.RoundPublicStateIndex;
import com.epsilon.core.Tile;
import com.epsilon.prism.core.Threats;
import com.epsilon.prism.core.Twins;
import java.util.List;

/**
 * 公開情報だけで測った打牌の安全度（体制の代表の「安全度の最善」）。判断の席から見える情報（河・立直の現物・見えている枚数）だけを読み、ほかの席の
 * 伏せた牌を見ない。
 *
 * <p>脅威の相手 j（{@link Threats#isThreat}）に対する現物の集合 G_j: j が立直なら {@code
 * riichiGenbutsuTileTypeMask(j)}（立直前の j の河と、立直後に誰かが通した牌）、立直でなければ j の河（{@code riverTileTypeMask(j)}）。筋も G_j
 * で数える（立直者の待ちは変わらないので、立直後に通った牌の筋も河の筋と同じ理由で安全になる）。
 *
 * <p>牌種 t の危険の類（小さいほど安全）:
 *
 * <table>
 *   <caption>危険の類</caption>
 *   <tr><td>0</td><td>現物: t が G_j にある</td></tr>
 *   <tr><td>1</td><td>字牌で見えている枚数（{@code visibleTileCount}。自分の手牌を含む）が 3 以上、または 1・9 の筋（1 は 4、9 は 6 が G_j に
 *       ある）</td></tr>
 *   <tr><td>2</td><td>字牌で見えている枚数が 2、または 2・3・7・8 の筋（2 は 5、3 は 6、7 は 4、8 は 5 が G_j にある）、または筋でない 1・9</td></tr>
 *   <tr><td>3</td><td>4・5・6 の両筋（4 は 1 と 7、5 は 2 と 8、6 は 3 と 9 の両方が G_j にある）、または字牌で見えている枚数が 1</td></tr>
 *   <tr><td>4</td><td>字牌で見えている枚数が 0、または筋でない 2・3・7・8、または片筋の 4・5・6</td></tr>
 *   <tr><td>5</td><td>筋のない 4・5・6</td></tr>
 * </table>
 *
 * <p>安全度の順（{@link #safest}）: 打牌ごとに (立直している相手のうち現物でない相手の数, 脅威の相手の危険の類の最大, −p0, 合法手の位置) を辞書式に
 * 比べ、最小の打牌を選ぶ（立直者全員の現物が先、次に筋と字牌・端牌の順）。
 */
public final class PublicSafety {
  private PublicSafety() {}

  /**
   * 席 seat から見た、脅威の相手 opponent に対する牌種 tileType の危険の類（0〜5）。
   *
   * @param state 判断の時点の state
   */
  public static int dangerClass(GameState state, int seat, int opponent, int tileType) {
    RoundPublicStateIndex open = state.publicState();
    long safe = safeMask(state, open, opponent);
    if (has(safe, tileType)) return 0;
    if (tileType >= Tile.TILE_TYPES_PER_SUIT * 3) {
      int visible = open.visibleTileCount(tileType) + state.hand(seat).count(tileType);
      return visible >= 3 ? 1 : visible == 2 ? 2 : visible == 1 ? 3 : 4;
    }
    int rank = tileType % Tile.TILE_TYPES_PER_SUIT; // 0 が 1、8 が 9
    int base = tileType - rank;
    return switch (rank) {
      case 0 -> has(safe, base + 3) ? 1 : 2;
      case 8 -> has(safe, base + 5) ? 1 : 2;
      case 1, 2 -> has(safe, tileType + 3) ? 2 : 4;
      case 6, 7 -> has(safe, tileType - 3) ? 2 : 4;
      default -> {
        // 4・5・6: 両側の筋（t−3 と t+3）
        boolean low = has(safe, tileType - 3), high = has(safe, tileType + 3);
        yield low && high ? 3 : low || high ? 4 : 5;
      }
    };
  }

  /**
   * 安全度の最善の打牌の位置（打牌がなければ -1。脅威の相手がいなければ p0 の最大の打牌）。
   *
   * @param prior 合法手ごとの p0
   */
  public static int safest(GameState state, int seat, List<Action> legal, float[] prior) {
    int n = legal.size();
    boolean anyThreat = false;
    for (int other = 0; other < GameState.NUM_PLAYERS; other++)
      if (Threats.isThreat(state, seat, other)) anyThreat = true;
    RoundPublicStateIndex open = state.publicState();
    int best = -1, bestUnsafeRiichi = 0, bestDanger = 0;
    for (int a = 0; a < n; a++) {
      Action action = legal.get(a);
      if (!Twins.discard(action)) continue;
      int unsafeRiichi = 0, danger = 0;
      if (anyThreat) {
        int tileType = action.tileType();
        for (int other = 0; other < GameState.NUM_PLAYERS; other++) {
          if (!Threats.isThreat(state, seat, other)) continue;
          if (state.isRiichi(other) && !has(open.riichiGenbutsuTileTypeMask(other), tileType))
            unsafeRiichi++;
          danger = Math.max(danger, dangerClass(state, seat, other, tileType));
        }
      }
      // 辞書式: 立直者の現物でない数 → 危険の類の最大 → p0 の大きい手 → 番号の小さい手（同点なら先の手を残す）
      if (best < 0
          || unsafeRiichi < bestUnsafeRiichi
          || (unsafeRiichi == bestUnsafeRiichi
              && (danger < bestDanger || (danger == bestDanger && prior[a] > prior[best])))) {
        best = a;
        bestUnsafeRiichi = unsafeRiichi;
        bestDanger = danger;
      }
    }
    return best;
  }

  /** 相手 opponent に対する現物の牌種の集合 G_j（立直なら立直の現物、立直でなければ河）。 */
  private static long safeMask(GameState state, RoundPublicStateIndex open, int opponent) {
    return state.isRiichi(opponent)
        ? open.riichiGenbutsuTileTypeMask(opponent)
        : open.riverTileTypeMask(opponent);
  }

  private static boolean has(long mask, int tileType) {
    return ((mask >>> tileType) & 1L) != 0;
  }
}
