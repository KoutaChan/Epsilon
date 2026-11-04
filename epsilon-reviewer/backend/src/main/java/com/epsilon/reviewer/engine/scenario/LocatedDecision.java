package com.epsilon.reviewer.engine.scenario;

import com.epsilon.core.Action;
import com.epsilon.reviewer.engine.view.TableState;
import java.util.List;

/**
 * 解析の 1 判断の場所と、その時点の局面。
 *
 * @param start 局の始め（StartKyoku）の出来事の位置
 * @param end 局の終わりの次の出来事の位置（次の局の始めか牌譜の終わり）
 * @param eventIndex 判断が解く出来事の位置
 * @param table 判断の時点の表示の局面（他家の手も含む）
 * @param recorded 牌譜の手。未観測の応答は null
 * @param concealed 判断の席の伏せた手の物理牌（手に入った順、配牌は配牌の順）。表示のツモ牌は含まない
 * @param drawn 表示のツモ牌の物理牌。ツモ牌が見えなければ -1
 */
public record LocatedDecision(
    int start,
    int end,
    int eventIndex,
    int seat,
    Kind kind,
    TableState table,
    Action recorded,
    List<Integer> concealed,
    int drawn) {
  public enum Kind {
    /** 手番の判断（ツモの後・鳴いた後の打牌）。 */
    TURN,
    /** 他家の打牌・加槓への応答（窓の判断）。 */
    RESPONSE
  }

  /** 判断の局の中の位置。 */
  public int target() {
    return eventIndex - start;
  }

  /** 判断の席がリーチしているか。 */
  public boolean riichi() {
    return table.players().get(seat).riichi();
  }
}
