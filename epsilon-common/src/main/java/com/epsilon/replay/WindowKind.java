package com.epsilon.replay;

import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.replay.ReplayEvent.Chi;
import com.epsilon.replay.ReplayEvent.Daiminkan;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.Hora;
import com.epsilon.replay.ReplayEvent.None;
import com.epsilon.replay.ReplayEvent.Pon;
import com.epsilon.replay.ReplayEvent.ReachAccepted;

/** 応答の窓を開いた出来事の種類。どの出来事が窓を開いたまま進み、どれが窓への応答かを決める（source は窓を開いた席）。 */
enum WindowKind {
  DAHAI(EngineDecisionKind.RESPONSE),
  KAKAN(EngineDecisionKind.CHANKAN);

  private final EngineDecisionKind decisionKind;

  WindowKind(EngineDecisionKind decisionKind) {
    this.decisionKind = decisionKind;
  }

  static WindowKind of(TurnEvent.ResponseSource origin) {
    return switch (origin) {
      case TurnEvent.Discard ignored -> DAHAI;
      case TurnEvent.KanAttempt ignored -> KAKAN;
    };
  }

  EngineDecisionKind decisionKind() {
    return decisionKind;
  }

  /**
   * 窓を閉じずに適用する公開の更新か。牌譜は打牌と応答の間に、打牌の後のドラ表示と source のリーチの成立を書くことがある。加槓の新ドラは槍槓が無いと決まってから開くので、加槓の窓では
   * ドラ表示が窓を閉じる。
   */
  boolean keepsOpen(int source, ReplayEvent event) {
    return switch (event) {
      case Dora ignored -> this == DAHAI;
      case ReachAccepted accepted -> accepted.actor() == source;
      default -> false;
    };
  }

  /**
   * event がこの窓への応答の形なら応答した席、違えば -1。形は明示の見送り（None）、source へのロン、打牌の窓ならそれへのチー・ポン・大明槓。その席が判断を持つか、手が合法かは
   * 見ない。
   */
  int responder(int source, ReplayEvent event) {
    return switch (event) {
      case None pass -> pass.actor();
      case Hora win when win.isRon() && win.target() == source -> win.actor();
      case Chi call when this == DAHAI && call.target() == source -> call.actor();
      case Pon call when this == DAHAI && call.target() == source -> call.actor();
      case Daiminkan call when this == DAHAI && call.target() == source -> call.actor();
      default -> -1;
    };
  }
}
