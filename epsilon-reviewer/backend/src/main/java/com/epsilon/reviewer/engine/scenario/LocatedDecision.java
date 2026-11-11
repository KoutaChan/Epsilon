package com.epsilon.reviewer.engine.scenario;

import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.reviewer.engine.view.TableState;
import java.util.List;

/**
 * 解析した 1 つの判断の位置と、その時点の局面。
 *
 * @param round 牌譜の何局目か（0 から）
 * @param position 判断の時点（判断を決める牌譜のイベントの位置）
 * @param table 判断の時点の表示用の局面（他家の手牌も含む）
 * @param recorded 牌譜の手の行動の番号。牌譜から分からない応答は -1
 * @param legal 合法手の行動の番号（エンジンが並べた順）
 * @param concealed 判断する席の伏せた手牌の物理牌（手に入った順。配牌の分は配られた順）。表示するツモ牌は含まない
 * @param drawn 表示するツモ牌の物理牌。表示するツモ牌が無ければ -1
 */
public record LocatedDecision(
    int round,
    int position,
    int seat,
    Kind kind,
    TableState table,
    int recorded,
    int[] legal,
    List<Integer> concealed,
    int drawn) {
  public enum Kind {
    /** 自分の手番の判断（ツモの後、または鳴いた後の打牌）。 */
    TURN,
    /** 他家の打牌・加槓・北抜きへの応答。 */
    RESPONSE;

    public static Kind of(EngineDecisionKind kind) {
      return switch (kind) {
        case TURN, POST_CALL_DAHAI -> TURN;
        case RESPONSE, CHANKAN, NUKI_RESPONSE -> RESPONSE;
      };
    }
  }

  /** 判断する席が立直しているか。 */
  public boolean riichi() {
    return table.players().get(seat).riichi();
  }
}
