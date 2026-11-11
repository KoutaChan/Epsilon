package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinContext.Source;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Objects;

/** 伏せた牌と運で決まるイベントも読めるビュー（{@link FullView}）。状態をコピーせずに読み、席ごとのビューを 1 つずつ持つ。 */
final class FullReader implements FullView {

  private final PublicState state;
  private final Hand[] hands;
  private final ChanceLog chance;
  private final SeatReader[] seats;

  FullReader(PublicState state, Hand[] hands, Decisions decisions, ChanceLog chance) {
    this.state = state;
    this.hands = hands;
    this.chance = chance;
    seats = new SeatReader[hands.length];
    for (int seat = 0; seat < seats.length; seat++) {
      seats[seat] = new SeatReader(state, hands, decisions, seat);
    }
  }

  @Override
  public Variant variant() {
    return state.variant();
  }

  @Override
  public SeatView seat(int seat) {
    return seats[Objects.checkIndex(seat, seats.length)];
  }

  @Override
  public ChanceLog chance() {
    return chance;
  }

  @Override
  public int unseen(int kind) {
    return chance.unseen(kind);
  }

  @Override
  public long waits(int seat) {
    return hands[seat].waits();
  }

  @Override
  public boolean furiten(int seat) {
    return hands[seat].missedRon();
  }

  @Override
  public void copyHand(int seat, HandShape out) {
    out.copyFrom(hands[Objects.checkIndex(seat, hands.length)].shape());
  }

  /**
   * その牌がいま打たれたとして、ロンできるかを調べる。山の残りが 0 なら河底になる。判定は打牌への応答と同じで、点に裏ドラは
   * 数えない。false のとき、書く先の中身に意味は無い。
   */
  @Override
  public boolean ronScore(int seat, int tile, int from, RuleWorkspace ws, WinScore out) {
    Variant variant = state.variant();
    Objects.checkIndex(seat, hands.length);
    Objects.checkIndex(from, hands.length);
    if (seat == from) {
      throw new IllegalArgumentException("Seat " + seat + " cannot ron its own discard");
    }
    if (!variant.tiles().contains(tile)) {
      throw new IllegalArgumentException(variant + " has no physical tile " + tile);
    }
    if (ws.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + ws.variant());
    }
    Source source = state.liveLeft() == 0 ? Source.HOUTEI : Source.RON;
    return WinLegality.ronWithoutUra(
        state, hands[seat], seat, Tiles.kind(tile), variant.tiles().red(tile), source, ws, out);
  }

  @Override
  public boolean hidden(int physicalId) {
    return chance.hidden(physicalId);
  }

  @Override
  public boolean tsumoScore(int seat, int kind, RuleWorkspace ws, WinScore out) {
    requireDrawn(seat, ws);
    return WinLegality.tsumoAs(state, hands[seat], kind, false, ws, out);
  }

  @Override
  public boolean tsumoScoreTile(int seat, int physicalId, RuleWorkspace ws, WinScore out) {
    requireDrawn(seat, ws);
    if (!chance.hidden(physicalId) && physicalId != chance.tile(chance.size() - 1)) {
      throw new IllegalArgumentException(
          "Tile " + physicalId + " is neither hidden nor the tile seat " + seat + " drew");
    }
    boolean red = state.variant().tiles().red(physicalId);
    return WinLegality.tsumoAs(state, hands[seat], Tiles.kind(physicalId), red, ws, out);
  }

  /** 直前のイベントがその席のツモで、作業領域のルールが局のルールと合うことを確かめる。 */
  private void requireDrawn(int seat, RuleWorkspace ws) {
    TurnEvent last = state.last();
    if (last.kind() != TurnEvent.Kind.DRAW || last.seat() != Objects.checkIndex(seat, hands.length)) {
      throw new IllegalStateException(
          "Seat " + seat + " did not draw last: " + last.kind() + " of seat " + last.seat());
    }
    if (ws.variant() != state.variant()) {
      throw new IllegalArgumentException(
          "Variant mismatch: expected " + state.variant() + " but found " + ws.variant());
    }
  }
}
