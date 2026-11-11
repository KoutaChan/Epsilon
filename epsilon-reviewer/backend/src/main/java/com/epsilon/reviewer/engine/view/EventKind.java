package com.epsilon.reviewer.engine.view;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.record.RecordEvent;

/**
 * 手順に書くイベントの種類。名前は mjai のもので、protocol/result.ts の EVENT_KINDS と同じ。立直の宣言牌の打牌では、その前に立直の
 * 宣言を 1 つの手順として書く。和了・流局・局の始めと終わり・終局は牌譜のイベントではないので、局の始まりと結果から書く。
 */
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
  /** 三人打ちの北抜き。名前は RiichiEnv の mjai に合わせる。 */
  NUKI("kita", true),
  /** 立直の宣言。 */
  REACH("reach", true),
  /** 立直の成立（供託の支払い）。 */
  REACH_ACCEPTED("reach_accepted", true),
  /** 槓ドラの表示。席を持たない。 */
  DORA("dora", false),
  HORA("hora", true),
  /** 流局。席を持たない。 */
  RYUKYOKU("ryukyoku", false),
  /** 局の終わり。席を持たない。 */
  END_KYOKU("end_kyoku", false),
  /** 対局の終わり。席を持たない。 */
  END_GAME("end_game", false);

  private final String wireName;
  private final boolean hasActor;

  EventKind(String wireName, boolean hasActor) {
    this.wireName = wireName;
    this.hasActor = hasActor;
  }

  public String wireName() {
    return wireName;
  }

  /** 結果の文書に {@code actor} を書くか。 */
  public boolean hasActor() {
    return hasActor;
  }

  /** 牌譜のイベントの種類を返す。立直の宣言牌も打牌として扱う。 */
  public static EventKind of(RecordEvent event) {
    return switch (event) {
      case RecordEvent.Draw ignored -> TSUMO;
      case RecordEvent.Discard ignored -> DAHAI;
      case RecordEvent.Call call ->
          switch (Meld.of(call.meld()).type()) {
            case CHI -> CHI;
            case PON -> PON;
            case DAIMINKAN -> DAIMINKAN;
            case ANKAN, KAKAN -> throw new IllegalArgumentException("Not a call: " + call);
          };
      case RecordEvent.ClosedKan ignored -> ANKAN;
      case RecordEvent.AddedKan ignored -> KAKAN;
      case RecordEvent.DoraReveal ignored -> DORA;
      case RecordEvent.RiichiAccepted ignored -> REACH_ACCEPTED;
      case RecordEvent.Nuki ignored -> NUKI;
    };
  }

  /** イベントを起こした席。席を持たないイベントは -1。 */
  public static int actor(RecordEvent event) {
    return switch (event) {
      case RecordEvent.Draw e -> e.seat();
      case RecordEvent.Discard e -> e.seat();
      case RecordEvent.Call e -> e.seat();
      case RecordEvent.ClosedKan e -> e.seat();
      case RecordEvent.AddedKan e -> e.seat();
      case RecordEvent.Nuki e -> e.seat();
      case RecordEvent.RiichiAccepted e -> e.seat();
      case RecordEvent.DoraReveal ignored -> -1;
    };
  }

  /** その前に立直の宣言を手順として書くイベント（立直の宣言牌の打牌）か。 */
  public static boolean declaresRiichi(RecordEvent event) {
    return event instanceof RecordEvent.Discard discard && discard.riichi();
  }
}
