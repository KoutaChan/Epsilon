package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import java.util.Objects;

/** 1 席から見た卓のビュー。公開情報と、全ての席の面子・抜いた北を読む。伏せた牌は、見る席の手牌だけを読む。 */
final class SeatReader implements SeatView {

  private final PublicState state;
  private final Hand[] hands;
  private final Decisions decisions;
  private final int viewer;

  /** 他家のツモを、引いた牌を伏せて見せるためのコピー。 */
  private final TurnEvent hiddenDraw = new TurnEvent();

  SeatReader(PublicState state, Hand[] hands, Decisions decisions, int viewer) {
    this.state = state;
    this.hands = hands;
    this.decisions = decisions;
    this.viewer = Objects.checkIndex(viewer, hands.length);
  }

  @Override
  public Variant variant() {
    return state.variant();
  }

  @Override
  public int seat() {
    return viewer;
  }

  @Override
  public EngineDecisionKind kind() {
    int slot = decisions.slotOf(viewer);
    if (slot < 0) {
      throw new IllegalStateException("Seat " + viewer + " has no decision");
    }
    return decisions.kind(slot);
  }

  @Override
  public int round() {
    return state.round();
  }

  @Override
  public int dealer() {
    return state.dealer();
  }

  @Override
  public int current() {
    return state.current();
  }

  @Override
  public int honba() {
    return state.honba();
  }

  @Override
  public int kyotaku() {
    return state.kyotaku();
  }

  @Override
  public int score(int seat) {
    return state.score(seat);
  }

  @Override
  public int liveLeft() {
    return state.liveLeft();
  }

  @Override
  public EventLog events() {
    return state.events();
  }

  @Override
  public TurnEvent last() {
    TurnEvent last = state.last();
    if (last.kind() != TurnEvent.Kind.DRAW || last.seat() == viewer) {
      return last;
    }
    hiddenDraw.setDraw(last.seat(), TurnEvent.NONE, false, last.rinshan(), last.doraPending());
    return hiddenDraw;
  }

  @Override
  public int count(int kind) {
    return hands[viewer].count(kind);
  }

  @Override
  public boolean holds(int physicalId) {
    return hands[viewer].holds(physicalId);
  }

  @Override
  public int meldCount(int seat) {
    return hands[seat].meldCount();
  }

  @Override
  public int meld(int seat, int index) {
    return hands[seat].meld(index);
  }

  @Override
  public int nuki(int seat) {
    return hands[seat].nuki();
  }

  @Override
  public int indicatorCount() {
    return state.dora().count();
  }

  @Override
  public int indicator(int index) {
    return state.dora().indicator(index);
  }

  @Override
  public int indicatorEvent(int index) {
    return state.dora().revealedAt(index);
  }

  @Override
  public int doraWeight(int kind) {
    return state.dora().weight(kind);
  }

  @Override
  public RiichiState riichi(int seat) {
    return state.riichi(seat);
  }

  @Override
  public boolean ippatsu(int seat) {
    return state.ippatsu(seat);
  }

  @Override
  public int visible(int kind) {
    return state.visible(kind);
  }

  @Override
  public long riverKinds(int seat) {
    return state.riverKinds(seat);
  }

  @Override
  public long riichiSafe(int seat) {
    return state.riichiSafe(seat);
  }

  @Override
  public boolean furiten() {
    return hands[viewer].missedRon();
  }

  @Override
  public long waits() {
    return hands[viewer].waits();
  }
}
