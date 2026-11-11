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

/**
 * テスト用に手で組み立てる牌譜の局（四人打ちと三人打ち）。牌は数字の後に種類の文字（m・p・s・z）を付けて書き、0 は赤の 5 を表す。
 * 物理牌はまだ使っていないものから選び、赤と書いていない牌には赤でないものを使う。打牌は、その席がツモった牌と同じ表記ならツモ切りに
 * なり、違えば手の中から打つ。
 */
final class RoundBuilder {

  /** 四人打ちのテストで使うルール。 */
  static final Variant VARIANT = Variant.TENHOU_4P;

  /** 表記の値（種類 × 2 ＋ 赤なら 1）で、赤のときに足す分。 */
  private static final int RED = 1;

  private final TileSet set;
  private final int seats;
  private final RoundStart start;
  private final int[][] deal;
  private final List<List<Integer>> hands = new ArrayList<>();
  private final boolean[] used = new boolean[Tiles.PHYSICAL_IDS];
  private final List<RecordEvent> events = new ArrayList<>();
  private int indicator = TurnEvent.NONE;
  private int drawn = TurnEvent.NONE;
  private int discarded = TurnEvent.NONE;
  private int discarder = TurnEvent.NONE;

  RoundBuilder(RoundStart start) {
    this.start = start;
    set = start.variant().tiles();
    seats = start.variant().seats();
    deal = new int[seats][];
    for (int seat = 0; seat < seats; seat++) {
      hands.add(new ArrayList<>());
    }
  }

  /** 席に配牌（13 枚）を配る。 */
  RoundBuilder hand(int seat, String tiles) {
    int[] faces = faces(tiles);
    deal[seat] = new int[faces.length];
    for (int index = 0; index < faces.length; index++) {
      deal[seat][index] = take(faces[index]);
      hands.get(seat).add(deal[seat][index]);
    }
    return this;
  }

  RoundBuilder indicator(String tile) {
    indicator = take(faces(tile)[0]);
    return this;
  }

  RoundBuilder draw(int seat, String tile) {
    drawn = take(faces(tile)[0]);
    hands.get(seat).add(drawn);
    events.add(new RecordEvent.Draw(seat, drawn));
    return this;
  }

  RoundBuilder discard(int seat, String tile) {
    return discard(seat, tile, false);
  }

  /** 席の立直。宣言牌の打牌と、宣言牌が通った後の立直の成立を書く。 */
  RoundBuilder riichi(int seat, String tile) {
    discard(seat, tile, true);
    events.add(new RecordEvent.RiichiAccepted(seat));
    return this;
  }

  /** 席の北抜き。ツモった牌が北ならその北を、違えば手の中の北を抜く。 */
  RoundBuilder nuki(int seat) {
    int north = drawn != TurnEvent.NONE && Tiles.kind(drawn) == Tiles.PEI ? drawn : held(seat, Tiles.PEI * 2);
    hands.get(seat).remove(Integer.valueOf(north));
    events.add(new RecordEvent.Nuki(seat, north));
    drawn = TurnEvent.NONE;
    return this;
  }

  /** 席が直前の打牌を、手の中の赤でない 2 枚でポンする。 */
  RoundBuilder pon(int seat) {
    int kind = Tiles.kind(discarded);
    for (int taken = 0; taken < 2; taken++) {
      hands.get(seat).remove(Integer.valueOf(held(seat, kind * 2)));
    }
    Meld.AkaSource aka = set.red(discarded) ? Meld.AkaSource.CALLED_TILE : Meld.AkaSource.NONE;
    Meld meld = Meld.pon(kind, Meld.RelativeSource.of(seat, discarder, seats), aka);
    events.add(new RecordEvent.Call(seat, meld.id()));
    return this;
  }

  /** ここまでのイベントのコピーを返す（記録を書き換えるテストで使う）。 */
  List<RecordEvent> events() {
    return new ArrayList<>(events);
  }

  RecordedRound round(RoundResult result) {
    return round(events, result);
  }

  RecordedRound round(List<RecordEvent> recorded, RoundResult result) {
    return new RecordedRound(start, deal, indicator, recorded, result);
  }

  /** 書いた牌それぞれの表記の値（種類 × 2 ＋ 赤なら 1）を、書いた順に返す。 */
  static int[] faces(String text) {
    int[] faces = new int[text.length()];
    int size = 0;
    int digits = 0;
    for (int i = 0; i < text.length(); i++) {
      char c = text.charAt(i);
      if (Character.isDigit(c)) {
        digits++;
        continue;
      }
      int first =
          switch (c) {
            case 'm' -> Tiles.M1;
            case 'p' -> Tiles.P1;
            case 's' -> Tiles.S1;
            case 'z' -> Tiles.TON;
            default -> throw new IllegalArgumentException("Unknown suit " + c + " in " + text);
          };
      for (int digit = i - digits; digit < i; digit++) {
        int number = text.charAt(digit) - '0';
        faces[size++] = number == 0 ? (first + Tiles.FIVE) * 2 + RED : (first + number - 1) * 2;
      }
      digits = 0;
    }
    return Arrays.copyOf(faces, size);
  }

  private RoundBuilder discard(int seat, String tile, boolean riichi) {
    int face = faces(tile)[0];
    boolean tsumogiri = drawn != TurnEvent.NONE && face(drawn) == face;
    int physicalId = tsumogiri ? drawn : held(seat, face);
    hands.get(seat).remove(Integer.valueOf(physicalId));
    events.add(new RecordEvent.Discard(seat, physicalId, tsumogiri, riichi));
    drawn = TurnEvent.NONE;
    discarded = physicalId;
    discarder = seat;
    return this;
  }

  /** 物理牌の表記の値を返す。 */
  private int face(int physicalId) {
    return Tiles.kind(physicalId) * 2 + (set.red(physicalId) ? RED : 0);
  }

  /**
   * 席の手にある、その表記の物理牌を返す。直前にツモった牌は選ばない。
   *
   * @param face 表記の値
   */
  private int held(int seat, int face) {
    for (int physicalId : hands.get(seat)) {
      if (physicalId != drawn && face(physicalId) == face) {
        return physicalId;
      }
    }
    throw new IllegalStateException("Seat " + seat + " holds no tile " + face);
  }

  /** 表記の値に当たる物理牌を、まだ使っていないものから 1 枚取る。 */
  private int take(int face) {
    int kind = face / 2;
    boolean red = face % 2 == RED;
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int physicalId = Tiles.physicalId(kind, copy);
      if (!used[physicalId] && set.contains(physicalId) && set.red(physicalId) == red) {
        used[physicalId] = true;
        return physicalId;
      }
    }
    throw new IllegalStateException("No free " + (red ? "red " : "") + "copy of kind " + kind);
  }
}
