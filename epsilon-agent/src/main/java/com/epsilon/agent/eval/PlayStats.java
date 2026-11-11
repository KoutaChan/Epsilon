package com.epsilon.agent.eval;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;

/**
 * 対局の内訳（診断用）。スロットごとに、局の結果と判断の傾向を席あたりで数える。どの席がどのスロットかは、半荘ごとの割り当てで渡す。
 *
 * <p>判断の傾向として数えるのは次の 4 つ。自分は立直しておらず他家の立直を受けて打牌するとき、全ての立直者の現物があればそれを切ったか。
 * 鳴ける場面で鳴いたか。立直できる場面で立直したか。抜いた北の数。
 *
 * <p>1 つのインスタンスに同時に書けるのは 1 つのスレッドだけ。半荘ごとに数えて、牌山や卓でまとめるときに {@link #add} で足す。数は全て整数
 * なので、結果は足す順に依らない。
 */
public final class PlayStats {

  private enum Count {
    HANDS,
    WINS,
    TSUMO,
    DEAL_INS,
    DEAL_INS_TO_RIICHI,
    RIICHI_HANDS,
    OPEN_HANDS,
    DRAWS,
    TENPAI_AT_DRAW,
    WIN_POINTS,
    DEAL_IN_POINTS,
    NET_POINTS,
    THREAT_DISCARDS,
    THREAT_WITH_SAFE,
    THREAT_CHOSE_SAFE,
    CALL_CHANCES,
    CALLS_TAKEN,
    RIICHI_CHANCES,
    RIICHI_TAKEN,
    NUKIS
  }

  private static final int COUNTS = Count.values().length;

  private final int slots;

  /** 添字は [スロット × 数の種類の数 ＋ 数の種類]。 */
  private final long[] counts;

  /** @param slots 卓のスロットの数（1 以上） */
  public PlayStats(int slots) {
    if (slots < 1) {
      throw new IllegalArgumentException("Play stats count at least one slot: " + slots);
    }
    this.slots = slots;
    counts = new long[slots * COUNTS];
  }

  /** @return スロットの数 */
  public int slots() {
    return slots;
  }

  /**
   * 主経路の判断を数える。合法手が 1 つだけの判断は数えない。
   *
   * @param slotOfSeat 席ごとのスロット
   * @param decision 判断の番号
   * @param choice 選んだ手（合法手の何番目か）
   */
  public void decided(int[] slotOfSeat, SeatView view, Decisions decisions, int decision, int choice) {
    int n = decisions.legalCount(decision);
    if (n < 2) {
      return;
    }
    int player = decisions.seat(decision);
    int seats = view.variant().seats();
    int slot = slotOfSeat[player];
    boolean canCall = false;
    boolean canRiichi = false;
    boolean canDiscard = false;
    for (int index = 0; index < n; index++) {
      switch (Action.of(decisions.legal(decision, index)).type()) {
        case CHI, PON -> canCall = true;
        case RIICHI_DAHAI -> canRiichi = true;
        case DAHAI -> canDiscard = true;
        default -> {}
      }
    }
    Action action = Action.of(decisions.legal(decision, choice));
    if (canCall) {
      add(slot, Count.CALL_CHANCES, 1);
      if (action.type() == Action.Type.CHI || action.type() == Action.Type.PON) {
        add(slot, Count.CALLS_TAKEN, 1);
      }
    }
    if (canRiichi) {
      add(slot, Count.RIICHI_CHANCES, 1);
      if (action.type() == Action.Type.RIICHI_DAHAI) {
        add(slot, Count.RIICHI_TAKEN, 1);
      }
    }
    if (action.type() == Action.Type.NUKI) {
      add(slot, Count.NUKIS, 1);
    }
    if (!canDiscard || riichi(view, player)) {
      return;
    }
    int threats = 0;
    for (int seat = 0; seat < seats; seat++) {
      if (seat != player && riichi(view, seat)) {
        threats |= 1 << seat;
      }
    }
    if (threats == 0) {
      return;
    }
    add(slot, Count.THREAT_DISCARDS, 1);
    boolean safeAvailable = false;
    for (int index = 0; index < n; index++) {
      Action candidate = Action.of(decisions.legal(decision, index));
      if (candidate.type() == Action.Type.DAHAI && safe(view, threats, candidate.tile())) {
        safeAvailable = true;
      }
    }
    if (!safeAvailable) {
      return;
    }
    add(slot, Count.THREAT_WITH_SAFE, 1);
    if ((action.type() == Action.Type.DAHAI || action.type() == Action.Type.RIICHI_DAHAI)
        && safe(view, threats, action.tile())) {
      add(slot, Count.THREAT_CHOSE_SAFE, 1);
    }
  }

  /**
   * 主経路の局末を数える。
   *
   * @param slotOfSeat 席ごとのスロット
   */
  public void roundEnded(int[] slotOfSeat, FullView full, RoundEnd end) {
    int seats = full.variant().seats();
    SeatView view = full.seat(0);
    RoundOutcome outcome = end.outcome();
    for (int seat = 0; seat < seats; seat++) {
      int slot = slotOfSeat[seat];
      add(slot, Count.HANDS, 1);
      add(slot, Count.NET_POINTS, end.delta(seat));
      if (riichi(view, seat)) {
        add(slot, Count.RIICHI_HANDS, 1);
      }
      for (int meld = 0; meld < view.meldCount(seat); meld++) {
        if (Meld.of(view.meld(seat, meld)).type() != Meld.Type.ANKAN) {
          add(slot, Count.OPEN_HANDS, 1);
          break;
        }
      }
    }
    switch (outcome.kind()) {
      case TSUMO -> {
        int winner = outcome.winner(0);
        int slot = slotOfSeat[winner];
        add(slot, Count.WINS, 1);
        add(slot, Count.TSUMO, 1);
        add(slot, Count.WIN_POINTS, end.delta(winner));
      }
      case RON -> {
        int loser = outcome.loser();
        int loserSlot = slotOfSeat[loser];
        add(loserSlot, Count.DEAL_INS, 1);
        add(loserSlot, Count.DEAL_IN_POINTS, -end.delta(loser));
        boolean toRiichi = false;
        for (int index = 0; index < outcome.winnerCount(); index++) {
          int winner = outcome.winner(index);
          add(slotOfSeat[winner], Count.WINS, 1);
          add(slotOfSeat[winner], Count.WIN_POINTS, end.delta(winner));
          toRiichi |= riichi(view, winner);
        }
        if (toRiichi) {
          add(loserSlot, Count.DEAL_INS_TO_RIICHI, 1);
        }
      }
      case EXHAUSTIVE -> {
        for (int seat = 0; seat < seats; seat++) {
          add(slotOfSeat[seat], Count.DRAWS, 1);
          if ((outcome.tenpaiMask() & 1 << seat) != 0) {
            add(slotOfSeat[seat], Count.TENPAI_AT_DRAW, 1);
          }
        }
      }
      case ABORTIVE -> {}
    }
  }

  /**
   * ほかの内訳を足す。
   *
   * @param other スロットの数が同じ内訳
   */
  public void add(PlayStats other) {
    if (other.slots != slots) {
      throw new IllegalArgumentException("Play stats of " + other.slots + " slots added to " + slots);
    }
    for (int index = 0; index < counts.length; index++) {
      counts[index] += other.counts[index];
    }
  }

  /** 全ての数を 0 に戻す。 */
  public void clear() {
    Arrays.fill(counts, 0);
  }

  /** @return 列の数（スロットの数 × 数の種類の数） */
  public int columns() {
    return counts.length;
  }

  /** 全ての数を、スロットの順、スロットの中は数の種類の順に、それぞれ前にタブを付けて行に足す（決闘の games.tsv の列）。 */
  public void appendTo(StringBuilder line) {
    for (long count : counts) {
      line.append('\t').append(count);
    }
  }

  /**
   * {@link #appendTo} と同じ順に並べた数を足す。
   *
   * @param columns 数（{@link #columns} 個）
   * @throws IllegalArgumentException 数の個数が合わないとき
   */
  public void addColumns(long[] columns) {
    if (columns.length != counts.length) {
      throw new IllegalArgumentException("Play stats have " + counts.length + " columns, not " + columns.length);
    }
    for (int index = 0; index < counts.length; index++) {
      counts[index] += columns[index];
    }
  }

  /** @return そのスロットの席で打った局の数（席ごとに数えた局の和） */
  public long seatHands(int slot) {
    return get(slot, Count.HANDS);
  }

  /** @return そのスロットの率（1 席・1 局あたり。局が無ければ 0） */
  public Rates rates(int slot) {
    double hands = Math.max(1, get(slot, Count.HANDS));
    return new Rates(
        get(slot, Count.WINS) / hands,
        get(slot, Count.DEAL_INS) / hands,
        get(slot, Count.RIICHI_HANDS) / hands,
        get(slot, Count.OPEN_HANDS) / hands,
        get(slot, Count.NUKIS) / hands);
  }

  /**
   * 1 席・1 局あたりの率。
   *
   * @param win 和了
   * @param dealIn 放銃
   * @param riichi 立直した局
   * @param open 鳴いた局（暗槓だけの局は数えない）
   * @param nuki 抜いた北
   */
  public record Rates(double win, double dealIn, double riichi, double open, double nuki) {

    /** @return この率 − ほかの率 */
    public Rates minus(Rates other) {
      return new Rates(win - other.win, dealIn - other.dealIn, riichi - other.riichi, open - other.open, nuki - other.nuki);
    }
  }

  private void add(int slot, Count count, long amount) {
    counts[slot * COUNTS + count.ordinal()] += amount;
  }

  private long get(int slot, Count count) {
    return counts[slot * COUNTS + count.ordinal()];
  }

  private static boolean riichi(SeatView view, int seat) {
    return view.riichi(seat) != RiichiState.NONE;
  }

  /** 全ての立直者の現物か（どの立直者の河にもある牌種か）。 */
  private static boolean safe(SeatView view, int threats, int kind) {
    for (int seat = 0; seat < view.variant().seats(); seat++) {
      if ((threats & (1 << seat)) != 0 && !Tiles.contains(view.riverKinds(seat), kind)) {
        return false;
      }
    }
    return true;
  }
}
