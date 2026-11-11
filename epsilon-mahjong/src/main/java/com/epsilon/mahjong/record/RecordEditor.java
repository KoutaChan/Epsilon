package com.epsilon.mahjong.record;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 牌譜の局を 1 つの席から見た記録に直し、牌や点を書き換えて、もしもの局面を作る。記録に入るのは判断の時点より前のイベントだけ。
 * 自分のツモは物理牌のまま残し、他家のツモは牌を伏せる（{@code TurnEvent.NONE}）。打牌・鳴き・槓・北抜き・表示牌・立直の成立は
 * そのまま残す。書き換えられるのは物理牌（自分の配牌とツモ、打牌、表示牌）と、局の始まりの本場・供託・点だけ。イベントの種類と
 * 並びは変えない。{@code SeatTable} をこの局の始まりと自分の配牌、最初の表示牌で始め、イベントを順に渡すと、判断の時点の局面になる。
 *
 * <p>局の中で物理牌が重ならないようにするのは、呼び出す側の役目（物理牌は {@code TileCopies} の割り当て方で選ぶ）。書き換えた
 * 記録がルールに合うか（打つ牌や、鳴き・槓に使う牌が手にあるか）は、{@code SeatTable} が受けるときに確かめる。点・本場・供託の
 * 範囲は確かめない。同時に使えるのは 1 つのスレッドだけ。
 */
public final class RecordEditor {

  /** 手に入った所と手から出た所の対応が無いことを表す。 */
  private static final int NONE = -1;

  private final Variant variant;
  private final TileSet set;
  private final int seat;
  private final int round;
  private int honba;
  private int kyotaku;
  private final int[] scores;

  /** 自分の配牌の物理 id。 */
  private final int[] hand;

  /** 最初の表示牌の物理 id。 */
  private int indicator;

  /** 判断の時点までの、その席から見たイベント。 */
  private final List<RecordEvent> events;

  /**
   * 自分の牌が手に入った所（配牌かツモ）と、その牌を手から出した所（打牌か北抜き）の対応。片方の位置に、もう片方の位置を入れる。
   * 位置は、配牌の i 枚目なら i、p 番目のイベントなら {@code Variant.HAND + p}。対応の無い所は {@link #NONE}。
   */
  private final int[] partner;

  /**
   * 牌譜の局から、判断の時点より前のイベントを判断する席から見た形で取り出し、記録を作る。
   *
   * @param round 牌譜の局
   * @param position 判断の時点（{@code ReplayDriver.cursor} の値）。これより前のイベントを、その席から見た形にして持つ
   * @param seat 判断する席
   * @throws IllegalArgumentException 記録がその席のツモを伏せているか、その席が持っていない牌を打つか抜く
   */
  public RecordEditor(RecordedRound round, int position, int seat) {
    RoundStart start = round.start();
    variant = start.variant();
    set = variant.tiles();
    this.seat = Objects.checkIndex(seat, variant.seats());
    List<RecordEvent> recorded = round.events();
    Objects.checkIndex(position, recorded.size() + 1);
    this.round = start.round();
    honba = start.honba();
    kyotaku = start.kyotaku();
    scores = start.scores();
    hand = new int[Variant.HAND];
    for (int index = 0; index < Variant.HAND; index++) {
      hand[index] = round.deal(seat, index);
    }
    indicator = round.indicator();
    events = new ArrayList<>(position);
    partner = new int[Variant.HAND + position];
    Arrays.fill(partner, NONE);
    int[] entered = new int[Tiles.PHYSICAL_IDS];
    Arrays.fill(entered, NONE);
    for (int index = 0; index < Variant.HAND; index++) {
      entered[hand[index]] = index;
    }
    for (int at = 0; at < position; at++) {
      RecordEvent event = recorded.get(at);
      switch (event) {
        case RecordEvent.Draw draw when draw.seat() == seat -> {
          if (draw.tile() == TurnEvent.NONE) {
            throw new IllegalArgumentException("The record hides a draw of seat " + seat + " at " + at);
          }
          entered[draw.tile()] = Variant.HAND + at;
          events.add(draw);
        }
        case RecordEvent.Draw draw -> events.add(new RecordEvent.Draw(draw.seat(), TurnEvent.NONE));
        case RecordEvent.Discard discard when discard.seat() == seat -> {
          link(entered, discard.tile(), at);
          events.add(discard);
        }
        case RecordEvent.Nuki nuki when nuki.seat() == seat -> {
          link(entered, nuki.tile(), at);
          events.add(nuki);
        }
        default -> events.add(event);
      }
    }
  }

  public Variant variant() {
    return variant;
  }

  /** 判断する席。 */
  public int seat() {
    return seat;
  }

  /** 局の番号。ルールで決まる局の並びで何番目か。 */
  public int round() {
    return round;
  }

  public int honba() {
    return honba;
  }

  public int kyotaku() {
    return kyotaku;
  }

  /** 局の始まりの点（絶対席の順）のコピー。 */
  public int[] scores() {
    return scores.clone();
  }

  /** 自分の配牌（物理 id 13 枚、配牌の位置の順）のコピー。 */
  public int[] hand() {
    return hand.clone();
  }

  /** 最初の表示牌の物理 id。 */
  public int indicator() {
    return indicator;
  }

  /** 判断の時点までの、その席から見たイベント（書き換えた後のコピー）。 */
  public List<RecordEvent> events() {
    return List.copyOf(events);
  }

  /**
   * 自分の配牌の 1 枚を替える。
   *
   * @param index 配牌の位置
   * @param tile 新しい物理 id
   * @throws IllegalArgumentException このルールで使わない牌か、元の牌を判断の時点より前に打ったか抜いた（そのときは打牌の方を
   *     替える）
   */
  public void replaceDealt(int index, int tile) {
    Objects.checkIndex(index, Variant.HAND);
    requireTile(tile);
    requireHeld(index);
    hand[index] = tile;
  }

  /**
   * 自分のツモの牌を替える。
   *
   * @param position イベントの位置
   * @param tile 新しい物理 id
   * @throws IllegalArgumentException 自分のツモでないか、このルールで使わない牌か、元の牌を判断の時点より前に打ったか抜いた
   *     （そのときは打牌の方を替える）
   */
  public void replaceDrawn(int position, int tile) {
    if (!(event(position) instanceof RecordEvent.Draw draw) || draw.seat() != seat) {
      throw new IllegalArgumentException("Position " + position + " is not a draw of seat " + seat);
    }
    requireTile(tile);
    requireHeld(Variant.HAND + position);
    events.set(position, new RecordEvent.Draw(seat, tile));
  }

  /**
   * 打った牌を替える。自分の打牌なら、その牌が手に入った所（配牌の位置かツモ）も同じ牌にする。ツモ切りと立直の印は変えない。
   * 判断の時点より前に鳴かれた打牌は、種類も赤かどうかも元の牌と同じ牌にしか替えられない。
   *
   * @param position イベントの位置
   * @param tile 新しい物理 id
   * @throws IllegalArgumentException 打牌でないか、このルールで使わない牌か、鳴かれた打牌で元の牌と種類か赤かどうかが違う
   */
  public void replaceDiscarded(int position, int tile) {
    if (!(event(position) instanceof RecordEvent.Discard discard)) {
      throw new IllegalArgumentException("Position " + position + " is not a discard");
    }
    requireTile(tile);
    RecordEvent.Call call = callOf(position);
    if (call != null) {
      Meld meld = Meld.of(call.meld());
      if (Tiles.kind(tile) != meld.called() || set.red(tile) != (meld.aka() == Meld.AkaSource.CALLED_TILE)) {
        throw new IllegalArgumentException(
            "Seat " + call.seat() + " calls the discard at " + position + ", so it stays a tile of its meld");
      }
    }
    events.set(position, new RecordEvent.Discard(discard.seat(), tile, discard.tsumogiri(), discard.riichi()));
    int entry = partner[Variant.HAND + position];
    if (entry == NONE) {
      return;
    }
    if (entry < Variant.HAND) {
      hand[entry] = tile;
    } else {
      events.set(entry - Variant.HAND, new RecordEvent.Draw(seat, tile));
    }
  }

  /**
   * 表示牌を替える。
   *
   * @param index 表示牌の番号。0 は局の始めの表示牌、1 からは判断の時点より前にめくった順
   * @param tile 新しい物理 id
   * @throws IllegalArgumentException このルールで使わない牌か、判断の時点までにめくっていない表示牌
   */
  public void replaceIndicator(int index, int tile) {
    requireTile(tile);
    if (index == 0) {
      indicator = tile;
      return;
    }
    int revealed = 0;
    for (int position = 0; position < events.size(); position++) {
      if (events.get(position) instanceof RecordEvent.DoraReveal && ++revealed == index) {
        events.set(position, new RecordEvent.DoraReveal(tile));
        return;
      }
    }
    throw new IllegalArgumentException(
        "Indicator " + index + " is not revealed before the decision, only " + (revealed + 1) + " are");
  }

  /**
   * 局の始まりの本場・供託・点を替える。
   *
   * @param honba 本場
   * @param kyotaku 供託の本数
   * @param scores 絶対席の順の点
   * @throws IllegalArgumentException 点の数が席の数と違う
   */
  public void replaceStart(int honba, int kyotaku, int[] scores) {
    if (scores.length != this.scores.length) {
      throw new IllegalArgumentException(variant + " needs " + this.scores.length + " scores, got " + scores.length);
    }
    this.honba = honba;
    this.kyotaku = kyotaku;
    System.arraycopy(scores, 0, this.scores, 0, scores.length);
  }

  /**
   * 自分が打ったか抜いた牌を、その牌が手に入った所と対応付ける。
   *
   * @param entered 物理 id ごとに、その牌が手に入った所（まだ手から出ていない牌だけ）
   * @param tile 打ったか抜いた牌の物理 id
   * @param at 打牌か北抜きのイベントの位置
   * @throws IllegalArgumentException その牌を持っていない
   */
  private void link(int[] entered, int tile, int at) {
    int entry = entered[tile];
    if (entry == NONE) {
      throw new IllegalArgumentException("Seat " + seat + " gives up tile " + tile + " at " + at + " without holding it");
    }
    partner[entry] = Variant.HAND + at;
    partner[Variant.HAND + at] = entry;
    entered[tile] = NONE;
  }

  /**
   * 手に入った所の牌が、判断の時点にまだ手にあることを確かめる。
   *
   * @param slot 手に入った所（配牌の位置か、{@code Variant.HAND} ＋ ツモの位置）
   * @throws IllegalArgumentException 判断の時点より前に、その牌を打ったか抜いた
   */
  private void requireHeld(int slot) {
    if (partner[slot] != NONE) {
      throw new IllegalArgumentException(
          "Seat " + seat + " gives up that tile at " + (partner[slot] - Variant.HAND) + ", so replace that event");
    }
  }

  /**
   * 指定した打牌が判断の時点より前に鳴かれていれば、その鳴きのイベントを返す。
   *
   * @param position 打牌の位置
   * @return 鳴いたイベント。鳴かれていなければ null
   */
  private RecordEvent.Call callOf(int position) {
    for (int at = position + 1; at < events.size(); at++) {
      switch (events.get(at)) {
        case RecordEvent.DoraReveal ignored -> {}
        case RecordEvent.RiichiAccepted ignored -> {}
        case RecordEvent.Call call -> {
          return call;
        }
        default -> {
          return null;
        }
      }
    }
    return null;
  }

  private RecordEvent event(int position) {
    return events.get(Objects.checkIndex(position, events.size()));
  }

  private void requireTile(int tile) {
    if (!set.contains(tile)) {
      throw new IllegalArgumentException(variant + " has no physical tile " + tile);
    }
  }
}
