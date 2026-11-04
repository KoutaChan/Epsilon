package com.epsilon.reviewer.engine.view;

import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.Ankan;
import com.epsilon.replay.ReplayEvent.Chi;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Daiminkan;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.EndGame;
import com.epsilon.replay.ReplayEvent.EndKyoku;
import com.epsilon.replay.ReplayEvent.Hora;
import com.epsilon.replay.ReplayEvent.Kakan;
import com.epsilon.replay.ReplayEvent.None;
import com.epsilon.replay.ReplayEvent.Pon;
import com.epsilon.replay.ReplayEvent.Reach;
import com.epsilon.replay.ReplayEvent.ReachAccepted;
import com.epsilon.replay.ReplayEvent.Ryukyoku;
import com.epsilon.replay.ReplayEvent.StartGame;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;

/** 手順になる出来事の種類（mjai の名前、protocol/result.ts の EVENT_KINDS）。start_game は手順にならない。 */
public enum EventKind {
  /** 局の始め。席を持たない。 */
  START_KYOKU("start_kyoku", false),
  TSUMO("tsumo", true),
  DAHAI("dahai", true),
  CHI("chi", true),
  PON("pon", true),
  DAIMINKAN("daiminkan", true),
  ANKAN("ankan", true),
  KAKAN("kakan", true),
  /** リーチの宣言。 */
  REACH("reach", true),
  /** リーチの成立（供託の支払い）。 */
  REACH_ACCEPTED("reach_accepted", true),
  /** 槓ドラ。席を持たない。 */
  DORA("dora", false),
  HORA("hora", true),
  /** 流局。席を持たない。 */
  RYUKYOKU("ryukyoku", false),
  /** 局の終わり。席を持たない。 */
  END_KYOKU("end_kyoku", false),
  /** 対局の終わり。席を持たない。 */
  END_GAME("end_game", false),
  /** 明示の見送り。 */
  NONE("none", true);

  private final String wireName;
  private final boolean hasActor;

  EventKind(String wireName, boolean hasActor) {
    this.wireName = wireName;
    this.hasActor = hasActor;
  }

  public String wireName() {
    return wireName;
  }

  /** 文書に actor を書くか。 */
  public boolean hasActor() {
    return hasActor;
  }

  /** @throws IllegalArgumentException 対局の始め（手順にならない） */
  public static EventKind of(ReplayEvent event) {
    return switch (event) {
      case StartGame ignored -> throw new IllegalArgumentException("start_game is not a step.");
      case StartKyoku ignored -> START_KYOKU;
      case Tsumo ignored -> TSUMO;
      case Dahai ignored -> DAHAI;
      case Chi ignored -> CHI;
      case Pon ignored -> PON;
      case Daiminkan ignored -> DAIMINKAN;
      case Ankan ignored -> ANKAN;
      case Kakan ignored -> KAKAN;
      case Reach ignored -> REACH;
      case ReachAccepted ignored -> REACH_ACCEPTED;
      case Dora ignored -> DORA;
      case Hora ignored -> HORA;
      case Ryukyoku ignored -> RYUKYOKU;
      case EndKyoku ignored -> END_KYOKU;
      case EndGame ignored -> END_GAME;
      case None ignored -> NONE;
    };
  }

  /** 出来事の席。席を持たない出来事は -1。 */
  public static int actor(ReplayEvent event) {
    return switch (event) {
      case Tsumo e -> e.actor();
      case Dahai e -> e.actor();
      case Chi e -> e.actor();
      case Pon e -> e.actor();
      case Daiminkan e -> e.actor();
      case Ankan e -> e.actor();
      case Kakan e -> e.actor();
      case Reach e -> e.actor();
      case ReachAccepted e -> e.actor();
      case Hora e -> e.actor();
      case None e -> e.actor();
      default -> -1;
    };
  }
}
