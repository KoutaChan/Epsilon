package com.epsilon.reviewer.engine.view;

import com.epsilon.mahjong.action.Meld.AkaSource;
import com.epsilon.mahjong.action.Meld.Type;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.List;

/**
 * 牌譜の 1 局をイベントの順に進める、表示用の卓。牌譜の記録だけで動く。全ての席の手牌（物理牌）、河の牌と印（ツモ切り・立直の
 * 宣言牌・鳴かれた牌）、面子、抜いた北、ドラ表示牌、点・本場・供託、残りのツモの数、手番を持つ。どの席の伏せた牌も持つので、推論には
 * 渡さない。抜いた北は、副露の後に 1 枚ずつ並べて見せる。
 *
 * <p>鳴きの記録は手から出した物理牌を持たない。そこで、手の赤 5 を使う鳴きでは赤の写しを、それ以外では赤でない写しのうち番号の最も
 * 小さいものを手から出す。
 */
public final class RecordTable {
  private final Variant variant;
  private final TileSet tiles;
  private final int seats;
  private final int round;
  private final int[] scores;
  private final int honba;
  private int kyotaku;

  /** [席][物理牌の id]。その牌がその席の純手牌にあれば true。 */
  private final boolean[][] hands;

  private final List<List<RiverTile>> rivers = new ArrayList<>();

  /** 席ごとの面子の番号（鳴いた順。加槓は元のポンの位置で置き換える）。 */
  private final List<List<Integer>> melds = new ArrayList<>();

  /** ドラ表示牌の物理牌の id（めくった順）。 */
  private final List<Integer> indicators = new ArrayList<>();

  private final boolean[] riichi;

  /** 席ごとの抜いた北の数。 */
  private final int[] nuki;

  /** この局で立直が成立した席のビットマスク。 */
  private int accepted;

  /** ツモの数。嶺上牌（槓や北抜きの後の補充）も含める。 */
  private int draws;

  private int current;

  /** 見えているツモ牌の席と物理牌の id。無ければ -1。 */
  private int drawSeat = -1;

  private int drawTile = -1;

  /** 最後にツモった牌、打った牌、加槓で足した牌、または抜いた北の物理牌の id。 */
  private int lastTile = -1;

  public RecordTable(RecordedRound recorded) {
    variant = recorded.start().variant();
    tiles = variant.tiles();
    seats = variant.seats();
    round = recorded.start().round();
    scores = recorded.start().scores();
    honba = recorded.start().honba();
    kyotaku = recorded.start().kyotaku();
    hands = new boolean[seats][Tiles.PHYSICAL_IDS];
    riichi = new boolean[seats];
    nuki = new int[seats];
    for (int seat = 0; seat < seats; seat++) {
      for (int index = 0; index < Variant.HAND; index++) hands[seat][recorded.deal(seat, index)] = true;
      rivers.add(new ArrayList<>());
      melds.add(new ArrayList<>());
    }
    indicators.add(recorded.indicator());
    current = variant.dealer(round);
  }

  public Variant variant() {
    return variant;
  }

  public int round() {
    return round;
  }

  public int honba() {
    return honba;
  }

  public int dealer() {
    return variant.dealer(round);
  }

  /**
   * イベントを 1 つ適用する。
   *
   * @throws IllegalArgumentException 手に無い牌を出すか、面子を作れない（牌譜が局と合わない）
   */
  public void apply(RecordEvent event) {
    switch (event) {
      case RecordEvent.Draw draw -> {
        if (draw.tile() == TurnEvent.NONE)
          throw new IllegalArgumentException("A hidden draw cannot be shown: " + draw);
        take(draw.seat(), draw.tile());
        draws++;
        current = draw.seat();
        drawSeat = draw.seat();
        drawTile = draw.tile();
        lastTile = draw.tile();
      }
      case RecordEvent.Discard discard -> {
        release(discard.seat(), discard.tile());
        rivers
            .get(discard.seat())
            .add(
                new RiverTile(
                    TileCode.of(tiles, discard.tile()),
                    discard.tsumogiri(),
                    discard.riichi(),
                    false));
        if (discard.riichi()) riichi[discard.seat()] = true;
        current = discard.seat();
        clearDraw();
        lastTile = discard.tile();
      }
      case RecordEvent.Call call -> {
        var meld = com.epsilon.mahjong.action.Meld.of(call.meld());
        markCalled(meld.source().seat(call.seat(), seats));
        List<Integer> consumed = new ArrayList<>(meld.size());
        for (int index = 0; index < meld.size(); index++) consumed.add(meld.tile(index));
        consumed.remove(Integer.valueOf(meld.called()));
        boolean redFromHand = meld.aka() == AkaSource.CONSUMED_HAND_TILE;
        for (int kind : consumed) {
          boolean red = redFromHand && Tiles.contains(Tiles.FIVES, kind);
          release(call.seat(), red ? Tiles.physicalId(kind, Tiles.RED_COPY) : plain(call.seat(), kind));
          redFromHand &= !red;
        }
        melds.get(call.seat()).add(call.meld());
        current = call.seat();
        clearDraw();
      }
      case RecordEvent.ClosedKan kan -> {
        boolean red = false;
        for (int copy = 0; copy < Tiles.COPIES; copy++) {
          int tile = Tiles.physicalId(kan.kind(), copy);
          red |= tiles.red(tile);
          release(kan.seat(), tile);
        }
        melds.get(kan.seat()).add(com.epsilon.mahjong.action.Meld.ankan(kan.kind(), red).id());
        current = kan.seat();
        clearDraw();
      }
      case RecordEvent.AddedKan kan -> {
        int tile = held(kan.seat(), kan.kind());
        release(kan.seat(), tile);
        List<Integer> own = melds.get(kan.seat());
        int index = ponIndex(own, kan.seat(), kan.kind());
        var pon = com.epsilon.mahjong.action.Meld.of(own.get(index));
        own.set(index, com.epsilon.mahjong.action.Meld.kakan(pon, tiles.red(tile)).id());
        current = kan.seat();
        clearDraw();
        lastTile = tile;
      }
      case RecordEvent.Nuki nuki -> {
        release(nuki.seat(), nuki.tile());
        this.nuki[nuki.seat()]++;
        current = nuki.seat();
        clearDraw();
        lastTile = nuki.tile();
      }
      case RecordEvent.DoraReveal reveal -> indicators.add(reveal.tile());
      case RecordEvent.RiichiAccepted riichiAccepted -> {
        scores[riichiAccepted.seat()] -= Variant.DEPOSIT;
        kyotaku++;
        accepted |= 1 << riichiAccepted.seat();
      }
    }
  }

  /** 和了牌の記号。ツモ和了なら最後のツモ、ロンなら最後の打牌、槍槓された加槓の牌、または抜いた北。 */
  public String winningTile() {
    if (lastTile < 0) throw new IllegalStateException("No tile has been drawn or discarded.");
    return TileCode.of(tiles, lastTile);
  }

  /**
   * 局末の和了 1 つ分の点の移動を卓に足す。本場と供託は最初の和了者が受け取る。そのため最初の和了では、局全体の点の移動から他の和了の
   * 分を引いた残りを足し、場の供託を 0 にする。
   *
   * @param end エンジンの局末
   * @param index 何番目の和了か（結果の和了者の順）
   */
  public void settleWin(RoundEnd end, int index) {
    RoundOutcome outcome = end.outcome();
    int[] share = new int[seats];
    if (index == 0) {
      for (int seat = 0; seat < seats; seat++) share[seat] = change(end, seat);
      for (int other = 1; other < outcome.winnerCount(); other++) {
        int[] taken = ron(outcome, other);
        for (int seat = 0; seat < seats; seat++) share[seat] -= taken[seat];
      }
      kyotaku = 0;
    } else share = ron(outcome, index);
    for (int seat = 0; seat < seats; seat++) scores[seat] += share[seat];
  }

  /** 流局の点の移動を卓に足す。場の供託は次の局へ残る。 */
  public void settleDraw(RoundEnd end) {
    for (int seat = 0; seat < seats; seat++) scores[seat] += change(end, seat);
  }

  /** 今の表示用の局面。 */
  public TableState state() {
    List<PlayerState> players = new ArrayList<>(seats);
    for (int seat = 0; seat < seats; seat++) {
      List<String> hand = new ArrayList<>(14);
      String draw = null;
      for (int tile = 0; tile < Tiles.PHYSICAL_IDS; tile++) {
        if (!hands[seat][tile]) continue;
        if (seat == drawSeat && tile == drawTile) draw = TileCode.of(tiles, tile);
        else hand.add(TileCode.of(tiles, tile));
      }
      List<Meld> shown = new ArrayList<>(melds.get(seat).size() + nuki[seat]);
      for (int id : melds.get(seat)) shown.add(MeldLayout.of(id, seat, seats));
      for (int index = 0; index < nuki[seat]; index++) shown.add(MeldLayout.nuki());
      players.add(
          new PlayerState(
              scores[seat],
              riichi[seat],
              List.copyOf(hand),
              draw,
              List.copyOf(rivers.get(seat)),
              List.copyOf(shown)));
    }
    List<String> dora = new ArrayList<>(indicators.size());
    for (int tile : indicators) dora.add(TileCode.of(tiles, tile));
    return new TableState(
        List.copyOf(players), List.copyOf(dora), honba, kyotaku, variant.liveWall() - draws, current);
  }

  /**
   * 判断する席が鳴いた後の局面を作る。席から見える卓は他家の伏せた牌を持たないので、取る元を分ける。鳴いた席の手牌・面子・手番と、
   * 点・供託は席から見える卓から取る（鳴かれた立直の宣言牌の立直棒は、鳴く前に払われている）。他家と河は判断の時点の表示から取り、
   * 鳴かれた牌に印を付ける。
   *
   * @param view 鳴いた席から見える、鳴いた後の卓
   * @param shown 鳴く前の、判断の時点の表示用の局面
   */
  public static TableState afterCall(SeatView view, TableState shown) {
    int seat = view.seat();
    int seats = view.variant().seats();
    TileSet tiles = view.variant().tiles();
    List<String> hand = new ArrayList<>(14);
    for (int index = 0; index < tiles.size(); index++) {
      int tile = tiles.physicalId(index);
      if (view.holds(tile)) hand.add(TileCode.of(tiles, tile));
    }
    List<Meld> shownMelds = new ArrayList<>(view.meldCount(seat) + view.nuki(seat));
    for (int index = 0; index < view.meldCount(seat); index++)
      shownMelds.add(MeldLayout.of(view.meld(seat, index), seat, seats));
    for (int index = 0; index < view.nuki(seat); index++) shownMelds.add(MeldLayout.nuki());
    int called = view.meld(seat, view.meldCount(seat) - 1);
    int from = com.epsilon.mahjong.action.Meld.of(called).source().seat(seat, seats);
    List<PlayerState> players = new ArrayList<>(seats);
    for (int player = 0; player < seats; player++)
      players.add(shown.players().get(player).withScore(view.score(player)));
    PlayerState own = players.get(seat);
    players.set(
        seat,
        new PlayerState(
            own.score(), own.riichi(), List.copyOf(hand), null, own.river(), List.copyOf(shownMelds)));
    PlayerState discarder = players.get(from);
    int last = discarder.river().size() - 1;
    RiverTile tile = discarder.river().get(last);
    players.set(
        from,
        discarder.withRiverTile(
            last, new RiverTile(tile.tile(), tile.tsumogiri(), tile.riichi(), true)));
    return new TableState(
        List.copyOf(players),
        shown.dora(),
        shown.honba(),
        view.kyotaku(),
        shown.remaining(),
        seat);
  }

  /** 局の始めからの点の移動のうち、まだ卓に足していない分。立直の供託は、成立したときに引いてある。 */
  private int change(RoundEnd end, int seat) {
    return end.delta(seat) + ((accepted >> seat & 1) != 0 ? Variant.DEPOSIT : 0);
  }

  /** 2 番目以降のロンの点の移動（本場と供託を除く）。 */
  private int[] ron(RoundOutcome outcome, int index) {
    int[] delta = new int[seats];
    Settlement.ron(
        variant,
        outcome.base(index),
        outcome.winner(index),
        outcome.loser(),
        dealer(),
        0,
        outcome.pao(index),
        delta);
    return delta;
  }

  private void markCalled(int seat) {
    List<RiverTile> river = rivers.get(seat);
    if (river.isEmpty()) throw new IllegalArgumentException("Seat " + seat + " has no discard to call.");
    RiverTile tile = river.getLast();
    river.set(river.size() - 1, new RiverTile(tile.tile(), tile.tsumogiri(), tile.riichi(), true));
  }

  private void take(int seat, int tile) {
    if (!tiles.contains(tile)) throw new IllegalArgumentException(variant + " does not use tile " + tile);
    hands[seat][tile] = true;
  }

  private void release(int seat, int tile) {
    if (!hands[seat][tile])
      throw new IllegalArgumentException("Seat " + seat + " does not hold tile " + tile);
    hands[seat][tile] = false;
  }

  /** 手にあるその種類の赤でない写しのうち、番号の最も小さいもの。 */
  private int plain(int seat, int kind) {
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int tile = Tiles.physicalId(kind, copy);
      if (hands[seat][tile] && !tiles.red(tile)) return tile;
    }
    throw new IllegalArgumentException("Seat " + seat + " holds no plain tile of kind " + kind);
  }

  /** 手にあるその種類の牌（加槓で足す 1 枚）。 */
  private int held(int seat, int kind) {
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int tile = Tiles.physicalId(kind, copy);
      if (hands[seat][tile]) return tile;
    }
    throw new IllegalArgumentException("Seat " + seat + " holds no tile of kind " + kind);
  }

  private static int ponIndex(List<Integer> melds, int seat, int kind) {
    for (int index = 0; index < melds.size(); index++) {
      var meld = com.epsilon.mahjong.action.Meld.of(melds.get(index));
      if (meld.type() == Type.PON && meld.base() == kind) return index;
    }
    throw new IllegalArgumentException("Seat " + seat + " has no pon of kind " + kind);
  }

  private void clearDraw() {
    drawSeat = -1;
    drawTile = -1;
  }
}
