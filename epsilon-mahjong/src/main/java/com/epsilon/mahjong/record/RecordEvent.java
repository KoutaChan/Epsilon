package com.epsilon.mahjong.record;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatTable;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 牌譜の局のイベント。牌譜の形式には依らない。席は起家を 0 とする絶対席で、牌は物理 id で表す。和了と流局はイベントにせず、
 * 局の結果（{@code RoundResult}）に持たせる。イベントの並び方は {@code RecordedRound} に書いた決まりに従う。
 */
public sealed interface RecordEvent {

  /** 1 つの席から見た卓に、このイベントを反映する。 */
  void applyTo(SeatTable table, RuleWorkspace ws);

  /** ツモ（嶺上ツモと北抜きの補充を含む）。1 つの席から見た記録では、他家がツモった牌は {@code TurnEvent.NONE}。 */
  record Draw(int seat, int tile) implements RecordEvent {

    public Draw {
      requireSeat(seat);
      if (tile != TurnEvent.NONE) {
        requireTile(tile);
      }
    }

    @Override
    public void applyTo(SeatTable table, RuleWorkspace ws) {
      table.draw(seat, tile, ws);
    }
  }

  /**
   * 打牌。
   *
   * @param riichi 立直の宣言牌か
   */
  record Discard(int seat, int tile, boolean tsumogiri, boolean riichi) implements RecordEvent {

    public Discard {
      requireSeat(seat);
      requireTile(tile);
    }

    @Override
    public void applyTo(SeatTable table, RuleWorkspace ws) {
      table.discard(seat, tile, tsumogiri, riichi, ws);
    }
  }

  /**
   * 直前の打牌へのチー・ポン・大明槓。
   *
   * @param meld 面子を表す番号（{@code Meld.id}）
   */
  record Call(int seat, int meld) implements RecordEvent {

    public Call {
      requireSeat(seat);
      Meld.Type type = Meld.of(meld).type();
      if (type == Meld.Type.ANKAN || type == Meld.Type.KAKAN) {
        throw new IllegalArgumentException("Not a call: " + Meld.of(meld));
      }
    }

    @Override
    public void applyTo(SeatTable table, RuleWorkspace ws) {
      table.call(seat, meld);
    }
  }

  /**
   * 暗槓。
   *
   * @param kind 槓した牌の種類
   */
  record ClosedKan(int seat, int kind) implements RecordEvent {

    public ClosedKan {
      requireSeat(seat);
      requireKind(kind);
    }

    @Override
    public void applyTo(SeatTable table, RuleWorkspace ws) {
      table.closedKan(seat, kind);
    }
  }

  /**
   * ポンへの加槓。
   *
   * @param kind 加槓した牌の種類
   */
  record AddedKan(int seat, int kind) implements RecordEvent {

    public AddedKan {
      requireSeat(seat);
      requireKind(kind);
    }

    @Override
    public void applyTo(SeatTable table, RuleWorkspace ws) {
      table.addedKan(seat, kind, ws);
    }
  }

  /**
   * 北抜き。1 つの席から見た卓は、抜いた北の物理 id を使わない。自分の北抜きでも、抜く北は卓が自分の手から選ぶ。
   *
   * @param seat 抜いた席
   * @param tile 抜いた北の物理 id
   */
  record Nuki(int seat, int tile) implements RecordEvent {

    public Nuki {
      requireSeat(seat);
      requireTile(tile);
      if (Tiles.kind(tile) != Tiles.PEI) {
        throw new IllegalArgumentException("Only a north is taken out, not tile " + tile);
      }
    }

    @Override
    public void applyTo(SeatTable table, RuleWorkspace ws) {
      table.nuki(seat, ws);
    }
  }

  /** 局の途中でめくったドラ表示牌。最初の表示牌は含まず、めくった順に並ぶ。 */
  record DoraReveal(int tile) implements RecordEvent {

    public DoraReveal {
      requireTile(tile);
    }

    @Override
    public void applyTo(SeatTable table, RuleWorkspace ws) {
      table.revealDora(tile, ws);
    }
  }

  /**
   * 立直の成立（宣言牌がロンされずに通り、供託を払った）。
   *
   * @param seat 立直した席
   */
  record RiichiAccepted(int seat) implements RecordEvent {

    public RiichiAccepted {
      requireSeat(seat);
    }

    @Override
    public void applyTo(SeatTable table, RuleWorkspace ws) {
      table.riichiAccepted(seat);
    }
  }

  private static void requireSeat(int seat) {
    if (seat < 0) {
      throw new IllegalArgumentException("Seat must not be negative: " + seat);
    }
  }

  private static void requireTile(int tile) {
    if (tile < 0 || tile >= Tiles.PHYSICAL_IDS) {
      throw new IllegalArgumentException("Physical tile out of range: " + tile);
    }
  }

  private static void requireKind(int kind) {
    if (kind < 0 || kind >= Tiles.KINDS) {
      throw new IllegalArgumentException("Tile kind out of range: " + kind);
    }
  }
}
