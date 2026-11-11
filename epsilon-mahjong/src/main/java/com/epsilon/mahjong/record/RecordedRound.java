package com.epsilon.mahjong.record;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.Objects;

/**
 * 牌譜の 1 局。局の始まりの卓、配牌（絶対席ごとに物理 id 13 枚）、最初の表示牌の物理 id、起きた順のイベント、局の結果を持つ。配列は
 * コピーして持つ。
 *
 * <p>イベントは、エンジンが局を進める順に並ぶ。ツモ、手番の手（打牌・暗槓・加槓・北抜き）、打牌への鳴き、鳴いた席の打牌の順になる。
 * 暗槓・加槓・大明槓・北抜きの次のツモは、嶺上か補充のツモ。和了・流局はイベントにせず、局の結果が持つ。
 *
 * <p>表示牌は、エンジンがめくる所に置く。めくる所は次の 3 つ。
 *
 * <ul>
 *   <li>暗槓の直後。保留していた表示牌、この槓の表示牌の順に置く。
 *   <li>大明槓・加槓の後の打牌の直後。
 *   <li>加槓・北抜きへの応答が出そろった直後。保留していた表示牌を、続く嶺上・補充のツモの前に置く。
 * </ul>
 *
 * <p>立直の成立は、宣言牌が通った所に置く。宣言牌とその後の表示牌より後、続く鳴きとツモより前になる。
 */
public record RecordedRound(
    RoundStart start, int[][] deal, int indicator, List<RecordEvent> events, RoundResult result) {

  public RecordedRound {
    Objects.requireNonNull(start, "start");
    Objects.requireNonNull(result, "result");
    int seats = start.variant().seats();
    if (deal.length != seats) {
      throw new IllegalArgumentException(start.variant() + " deals to " + seats + " seats, got " + deal.length);
    }
    int[][] copy = new int[seats][];
    for (int seat = 0; seat < seats; seat++) {
      if (deal[seat].length != Variant.HAND) {
        throw new IllegalArgumentException("A deal is " + Variant.HAND + " tiles, got " + deal[seat].length);
      }
      copy[seat] = deal[seat].clone();
    }
    deal = copy;
    events = List.copyOf(events);
    if (result.deltas().length != seats) {
      throw new IllegalArgumentException(seats + " seats need " + seats + " deltas, got " + result.deltas().length);
    }
  }

  /**
   * 牌譜の形式のままの並びで読んだ局のイベントを、このクラスの説明に書いた並びに直す。動かすのは表示牌と立直の成立だけで、ほかの
   * イベントの順は変えない。表示牌は、読んだ順にエンジンがめくる所へ置く。めくる所まで進まずに局が終わったときは、そこに置く表示牌が
   * 無くてもよい。立直の成立は、その席の宣言牌と、その直後の表示牌の後へ置く。
   *
   * <p>めくる所の無い表示牌、宣言牌の無い立直の成立、同じ席に 2 つある立直の成立があれば IllegalArgumentException を投げる。
   * 北抜きでめくる保留の表示牌が、読んだ並びで補充のツモより後にあるときも同じ。エンジンより遅くめくった記録なので、正しい並びに
   * 直せない。
   */
  public static List<RecordEvent> contractOrder(List<RecordEvent> read) {
    Deque<Integer> indicators = new ArrayDeque<>();
    int accepted = 0;
    List<RecordEvent> moves = new ArrayList<>(read.size());
    // 表示牌と立直の成立を除いた各イベントについて、読んだ並びでそれより前にある表示牌の数。
    int[] readBefore = new int[read.size()];
    for (RecordEvent event : read) {
      switch (event) {
        case RecordEvent.DoraReveal reveal -> indicators.addLast(reveal.tile());
        case RecordEvent.RiichiAccepted riichi -> {
          if ((accepted & 1 << riichi.seat()) != 0) {
            throw new IllegalArgumentException("Seat " + riichi.seat() + " has its riichi accepted twice");
          }
          accepted |= 1 << riichi.seat();
        }
        default -> {
          readBefore[moves.size()] = indicators.size();
          moves.add(event);
        }
      }
    }
    List<RecordEvent> ordered = new ArrayList<>(read.size());
    int pending = 0;
    int placed = 0;
    for (int index = 0; index < moves.size(); index++) {
      RecordEvent move = moves.get(index);
      if (move instanceof RecordEvent.Draw
          && index > 0
          && moves.get(index - 1) instanceof RecordEvent.Nuki nuki
          && readBefore[index] < placed) {
        throw new IllegalArgumentException(
            "The dora indicator due at the nuki of seat " + nuki.seat() + " is read after its replacement draw");
      }
      ordered.add(move);
      int due = 0;
      switch (move) {
        case RecordEvent.ClosedKan ignored -> {
          due = pending + 1;
          pending = 0;
        }
        case RecordEvent.AddedKan ignored -> {
          due = pending;
          pending = 1;
        }
        case RecordEvent.Call call when Meld.of(call.meld()).type() == Meld.Type.DAIMINKAN -> pending++;
        case RecordEvent.Discard ignored -> {
          due = pending;
          pending = 0;
        }
        case RecordEvent.Nuki ignored -> {
          due = pending;
          pending = 0;
        }
        default -> {}
      }
      for (; due > 0 && !indicators.isEmpty(); due--) {
        ordered.add(new RecordEvent.DoraReveal(indicators.removeFirst()));
        placed++;
      }
      if (move instanceof RecordEvent.Discard discard && discard.riichi() && (accepted & 1 << discard.seat()) != 0) {
        ordered.add(new RecordEvent.RiichiAccepted(discard.seat()));
        accepted &= ~(1 << discard.seat());
      }
    }
    if (!indicators.isEmpty()) {
      throw new IllegalArgumentException("No kan reveals the dora indicators " + indicators);
    }
    if (accepted != 0) {
      throw new IllegalArgumentException(
          "Riichi is accepted without a riichi discard: " + Integer.toBinaryString(accepted));
    }
    return ordered;
  }

  /** 配牌のうち、指定した席・位置の牌の物理 id。 */
  public int deal(int seat, int index) {
    return deal[seat][index];
  }

  /** コピーを返す。 */
  @Override
  public int[][] deal() {
    int[][] copy = new int[deal.length][];
    for (int seat = 0; seat < deal.length; seat++) {
      copy[seat] = deal[seat].clone();
    }
    return copy;
  }

  /** この局で立直が成立した席（{@code RiichiAccepted} のある席）のビットマスク。 */
  public int riichiMask() {
    int mask = 0;
    for (RecordEvent event : events) {
      if (event instanceof RecordEvent.RiichiAccepted accepted) {
        mask |= 1 << accepted.seat();
      }
    }
    return mask;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof RecordedRound round
        && start.equals(round.start)
        && Arrays.deepEquals(deal, round.deal)
        && indicator == round.indicator
        && events.equals(round.events)
        && result.equals(round.result);
  }

  @Override
  public int hashCode() {
    return Objects.hash(start, Arrays.deepHashCode(deal), indicator, events, result);
  }

  @Override
  public String toString() {
    return "RecordedRound[" + start + " deal=" + Arrays.deepToString(deal) + " indicator=" + indicator
        + " events=" + events + " " + result + "]";
  }
}
