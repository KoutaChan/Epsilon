package com.epsilon.replay;

import com.epsilon.engine.EngineDecisionKind;

/**
 * 再生の 1 判断。位置は {@link ReplayRecord#events()} の添字。
 *
 * @param kind 窓の外は TURN か POST_CALL_DAHAI（チー・ポンの直後の打牌）、打牌への応答は RESPONSE、加槓への応答は CHANKAN
 * @param chosenSlot 合法手の中の、牌譜の手か見送り（PASS）の位置。UNOBSERVED は -1
 * @param causeIndex 判断を起こした出来事の位置。手番はツモか鳴き、応答は打牌か加槓
 * @param resolveIndex 判断を解いた出来事の位置。牌譜の手はその出来事。見送りと未観測は窓を解いた最初の出来事（応答の並びの始めか、窓を閉じた出来事）で、明示の見送りでも並びの
 *     始め。窓が開いたまま牌譜が終われば出来事の数
 * @param revision 判断が見る局面の改訂（{@link ReplayObserver#onEventApplied} と同じ数）
 */
public record ReplayDecision(
    int seat,
    EngineDecisionKind kind,
    Choice choice,
    int chosenSlot,
    int causeIndex,
    int resolveIndex,
    int revision) {

  /** 牌譜から判断の手が分かるか。UNOBSERVED には牌譜の手が無いので教師にしない。 */
  public enum Choice {
    RECORDED_ACTION,
    /** 見送りが確か: 明示の見送りか、誰も鳴かずロンもしないまま牌譜が先へ進んだ。 */
    CONFIRMED_PASS,
    /** 牌譜から分からない: ほかの席が鳴くかロンした（塞がれた鳴きも）、または窓の途中で牌譜が終わった。 */
    UNOBSERVED
  }
}
