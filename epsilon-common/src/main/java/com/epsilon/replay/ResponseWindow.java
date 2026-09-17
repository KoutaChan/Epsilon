package com.epsilon.replay;

import com.epsilon.core.Action;
import com.epsilon.engine.WinLegality.RonStatus;
import com.epsilon.replay.ReplayEvent.Chi;
import com.epsilon.replay.ReplayEvent.Daiminkan;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.Hora;
import com.epsilon.replay.ReplayEvent.None;
import com.epsilon.replay.ReplayEvent.Pon;
import com.epsilon.replay.ReplayEvent.ReachAccepted;
import java.util.List;

/** 1回の打牌または加槓に対して、各席の合法な応答と、応答が確定するまでの状態を管理する。 */
final class ResponseWindow {

  private enum Type {
    DAHAI,
    KAKAN
  }

  private static final int NUM_PLAYERS = 4;

  private Type type;
  private int sourcePlayer;
  private int causeEventIndex;

  @SuppressWarnings("unchecked")
  private final List<Action>[] legalByPlayer = new List[NUM_PLAYERS];

  private final boolean[] ronDeclineCandidate = new boolean[NUM_PLAYERS];

  /** 打牌への応答の受け付けを開始する。 */
  ResponseWindow openDahai(int sourcePlayer, int causeEventIndex) {
    return open(Type.DAHAI, sourcePlayer, causeEventIndex);
  }

  /** 加槓への応答の受け付けを開始する。 */
  ResponseWindow openKakan(int sourcePlayer, int causeEventIndex) {
    return open(Type.KAKAN, sourcePlayer, causeEventIndex);
  }

  private ResponseWindow open(Type type, int sourcePlayer, int causeEventIndex) {
    this.type = type;
    this.sourcePlayer = sourcePlayer;
    this.causeEventIndex = causeEventIndex;
    for (int player = 0; player < NUM_PLAYERS; player++) {
      legalByPlayer[player] = null;
      ronDeclineCandidate[player] = false;
    }
    return this;
  }

  /** 応答が確定するまで変更されない合法手を借用する。 */
  void put(int player, List<Action> legal, RonStatus ronStatus) {
    if (legal.size() > 1) {
      legalByPlayer[player] = legal;
    }
    ronDeclineCandidate[player] = ronStatus.createsFuritenWhenDeclined();
  }

  /** 記録する判断も、見送りによるフリテンもないかを返す。 */
  boolean isEmpty() {
    for (int player = 0; player < NUM_PLAYERS; player++) {
      if (legalByPlayer[player] != null || ronDeclineCandidate[player]) {
        return false;
      }
    }
    return true;
  }

  /** 指定席に未解決の判断または見送りフリテンがあるかを返す。 */
  boolean hasPendingPlayer(int player) {
    return legalByPlayer[player] != null || ronDeclineCandidate[player];
  }

  /** 指定席に判断行として記録する合法手の選択があるかを返す。 */
  boolean hasLegalDecision(int player) {
    return legalByPlayer[player] != null;
  }

  /** 応答開始時から借用している合法手を返す。 */
  List<Action> legalActions(int player) {
    return legalByPlayer[player];
  }

  /** 打牌または加槓をした席を返す。 */
  int sourcePlayer() {
    return sourcePlayer;
  }

  int causeEventIndex() {
    return causeEventIndex;
  }

  /** 閉じる際に槍槓状態の解消が必要な状態かを返す。 */
  boolean isKakanResponse() {
    return type == Type.KAKAN;
  }

  /** 応答前の公開情報更新かを判定する。加槓直後のドラは先に槍槓への応答を確定する。 */
  boolean preservesWindow(ReplayEvent event) {
    return switch (event) {
      case Dora ignored -> type == Type.DAHAI;
      case ReachAccepted accepted -> accepted.actor() == sourcePlayer;
      default -> false;
    };
  }

  /** 現在受け付けている合法な応答イベントかを返す。 */
  boolean accepts(ReplayEvent event) {
    int actor = actorOf(event);
    if (actor < 0 || legalByPlayer[actor] == null) {
      return false;
    }
    return switch (type) {
      case DAHAI -> acceptsDahaiResponse(event);
      case KAKAN -> acceptsKanResponse(event);
    };
  }

  /** 明示的な見送りを含む応答者を返す。 */
  int actorOf(ReplayEvent event) {
    return switch (event) {
      case None none -> none.actor();
      default -> event.responseActor();
    };
  }

  /** 牌譜が見送りを明示しているかを返す。 */
  boolean isExplicitPass(ReplayEvent event) {
    return event instanceof None;
  }

  /** 応答がロン和了かを返す。 */
  boolean isRon(ReplayEvent event) {
    return event instanceof Hora hora && hora.actor() != hora.target();
  }

  /** 応答を見送ると一時フリテンになる席かを返す。 */
  boolean createsFuritenWhenDeclined(int player) {
    return ronDeclineCandidate[player];
  }

  private boolean acceptsDahaiResponse(ReplayEvent event) {
    return switch (event) {
      case None ignored -> true;
      case Hora hora -> hora.actor() != hora.target() && hora.target() == sourcePlayer;
      case Chi chi -> chi.target() == sourcePlayer;
      case Pon pon -> pon.target() == sourcePlayer;
      case Daiminkan daiminkan -> daiminkan.target() == sourcePlayer;
      default -> false;
    };
  }

  private boolean acceptsKanResponse(ReplayEvent event) {
    return switch (event) {
      case None ignored -> true;
      case Hora hora -> hora.actor() != hora.target() && hora.target() == sourcePlayer;
      default -> false;
    };
  }
}
