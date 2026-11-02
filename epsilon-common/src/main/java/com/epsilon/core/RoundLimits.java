package com.epsilon.core;

/**
 * 1 局の規則から導く上限。公開の出来事の記録（{@link PublicEventLog}）と河の容量、それを読む入力の欄の範囲はここから作り、上限を 2 か所に
 * 持たない。
 *
 * <p>導出の前提:
 *
 * <ul>
 *   <li>局の初めの山（王牌を除く）は 136 − 14 − 配牌 13 × 4 = 70 枚。嶺上牌を引くと山の最後の 1 枚が王牌へ回るので、嶺上を含むツモの
 *       総数も 70 以下。
 *   <li>打牌はツモの直後か、チー・ポンの直後にだけ起きる（大明槓・暗槓・加槓の後は嶺上のツモ）。1 席の副露（暗槓を含む）は 4 つまで。
 *   <li>加槓は元のポンを置き換える別の出来事で、槓は局で 4 つまで。
 * </ul>
 */
public final class RoundLimits {

  /** 1 席の配牌の枚数。 */
  private static final int DEALT_TILES_PER_SEAT = 13;

  /** 王牌を除く山の最大（122 − 配牌 52 = 70）。嶺上を引くと縮むので、嶺上を含むツモの総数もこれ以下。 */
  public static final int MAX_WALL =
      GameState.LIVE_WALL_SIZE - GameState.NUM_PLAYERS * DEALT_TILES_PER_SEAT;

  /** 1 席の副露（暗槓を含む）の最大。 */
  public static final int MAX_MELDS = 4;

  /** 1 局の槓の最大。 */
  public static final int MAX_KANS = 4;

  /** 1 局の打牌の最大（ツモ 70 ＋ チー・ポンの後の打牌 4 × 4 = 86）。 */
  public static final int MAX_DISCARDS = MAX_WALL + GameState.NUM_PLAYERS * MAX_MELDS;

  /**
   * 1 席の打牌の最大（ツモ 70 ＋ その席のチー・ポンの後の打牌 4 = 74）。鳴きで他家の手番が飛ぶと 24 枚を超えうるので、巡目からは決めない。
   */
  public static final int MAX_SEAT_DISCARDS = MAX_WALL + MAX_MELDS;

  /**
   * 1 局の公開の出来事の最大（打牌 86 ＋ 副露 16 ＋ 加槓 4 = 106）。槓ドラの表示は出来事に数えない。
   *
   * <p>106 &lt; 128 なので、出来事の番号は byte に入る。打牌の巡目（86 / 4 以下）と通し番号（86 未満）も byte に収まるが、{@link
   * PublicEventLog} は余裕を残してどちらも short で持つ。
   */
  public static final int MAX_PUBLIC_EVENTS =
      MAX_DISCARDS + GameState.NUM_PLAYERS * MAX_MELDS + MAX_KANS;

  private RoundLimits() {}
}
