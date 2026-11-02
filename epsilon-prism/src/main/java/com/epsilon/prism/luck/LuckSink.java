package com.epsilon.prism.luck;

import com.epsilon.core.GameState;
import com.epsilon.core.TurnEvent;

/**
 * 判断の境界で復元した偶然の出来事の受け口。
 *
 * <p>呼ぶ場所: {@code EventLedger.Tracker.observe} が、前の境界からの出来事を線形の欄と同じ順に復元する
 * ところ。新しい表示 {@code k}（{@code count − fresh} から {@code count − 1} まで、古い順）ごとに {@link #reveal} を、境界の
 * 直前のツモがあればその後に {@link #draw} を呼ぶ。帳簿の枠の有無（{@code seats == 0} の早い戻り）より前に呼び、運の網の出来事を
 * 枠の有無で選ばない。state は境界の状態で、受け取った側は書き換えない。
 */
public interface LuckSink {

  /**
   * 槓ドラの表示を受け取る。
   *
   * @param state 境界の状態（表示の後）
   * @param indicator 開いた表示の番号（表示の順、1 始まり）
   * @param laterDraw 表示の後、境界の前のツモ（なければ null）
   */
  void reveal(GameState state, int indicator, TurnEvent.Draw laterDraw);

  /**
   * 境界の直前のツモ（生きた山・嶺上）を受け取る。
   *
   * @param state 境界の状態（ツモの後）
   * @param draw そのツモ
   */
  void draw(GameState state, TurnEvent.Draw draw);
}
