package com.epsilon.replay;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.WinLegality.RonStatus;
import com.epsilon.replay.ReplayEvent.Chi;
import com.epsilon.replay.ReplayEvent.Daiminkan;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.Hora;
import com.epsilon.replay.ReplayEvent.None;
import com.epsilon.replay.ReplayEvent.Pon;
import java.util.Arrays;
import java.util.List;

/**
 * 打牌か加槓への応答の窓。窓の状態はここだけが持つ。{@link #open} で開き、{@link #resolve} か {@link #endOfRecord} が必ず同じ close で
 * 閉じる（判断する席が無ければ open の中で閉じる）。加槓の窓は閉じる時に、槓の成立か槍槓かが決まる。
 */
final class ResponsePhase {
  /** 牌譜が応答の並びで示した、その席の応答。並びを読んでから窓を閉じるまでだけ意味を持つ。 */
  private enum Reply {
    UNSEEN,
    PASSED,
    ACTED
  }

  private final List<ReplayEvent> events;
  private final ReplayTable table;
  private final LegalActions legal;
  private final ReplayFeed feed;

  /** PASS のほかに合法な応答がある席。この席だけが判断し、応答を受け付ける。 */
  private final boolean[] decides = new boolean[GameState.NUM_PLAYERS];

  /** ロンが合法な席。ロンしなければ一時振聴になる（ロンが合法なら PASS もあるので decides も真）。 */
  private final boolean[] ronOffered = new boolean[GameState.NUM_PLAYERS];

  private final Reply[] replies = new Reply[GameState.NUM_PLAYERS];

  /** 開いている窓の種類。閉じていれば null。 */
  private WindowKind kind;

  private int source;
  private int cause;

  ResponsePhase(List<ReplayEvent> events, ReplayTable table, LegalActions legal, ReplayFeed feed) {
    this.events = events;
    this.table = table;
    this.legal = legal;
    this.feed = feed;
    Arrays.fill(replies, Reply.UNSEEN);
  }

  boolean isOpen() {
    return kind != null;
  }

  /**
   * 元の席のほかの席の応答を作って窓を開く。役の無いロンの形の席はここで一時振聴になる（改訂は窓を開いた出来事の分に含む）。
   *
   * @param origin 打牌か加槓の宣言（GameState が使い回す物）
   * @param cause 窓を開いた出来事の位置
   */
  void open(TurnEvent.ResponseSource origin, int cause) {
    kind = WindowKind.of(origin);
    source = origin.player();
    this.cause = cause;
    boolean anyone = false;
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      if (seat == source) continue;
      RonStatus status = legal.respond(seat, origin);
      if (status.createsImmediateFuriten()) table.enterFuriten(seat);
      // 生成器は PASS をほかの手がある時だけ足すので、判断があるのは 2 手以上の席。
      if (legal.responses(seat).size() > 1) {
        decides[seat] = true;
        ronOffered[seat] = status.createsFuritenWhenDeclined();
        anyone = true;
      }
    }
    if (!anyone) close(false);
  }

  /**
   * 窓が開いている間の 1 歩。窓を閉じない公開の更新は適用して窓を開いたままにし、応答の並びは解いてから窓を閉じる。応答でない出来事は読まずに窓を閉じる（次に手番の段がその出来事を適用する）。
   *
   * @return 次に読む出来事の位置
   */
  int resolve(int index) {
    ReplayEvent first = events.get(index);
    if (kind.keepsOpen(source, first)) {
      // keepsOpen が通すのは打牌の窓のドラ表示と、source のリーチの成立だけ。
      if (first instanceof Dora dora) table.revealDora(dora);
      else table.payRiichi(source);
      feed.applied(index, first);
      return index + 1;
    }
    int end = index;
    for (; end < events.size(); end++) {
      ReplayEvent event = events.get(end);
      int seat = kind.responder(source, event);
      if (seat < 0 || !decides[seat]) break;
      replies[seat] = event instanceof None ? Reply.PASSED : Reply.ACTED;
    }
    if (end == index) {
      resolveUnanswered(index, true);
      close(false);
      return index;
    }
    resolveUnanswered(index, end < events.size() && !anyActed());
    // 複数ロンは全員の判断を出してから、記録の順に 1 つずつ精算して通知する。最初のロンより後の明示の見送りも位置の順にその間で通知する。
    int firstRon = end;
    for (int at = index; at < end; at++) {
      ReplayEvent event = events.get(at);
      answer(at, event, firstRon < at);
      if (event instanceof Hora && firstRon == end) firstRon = at;
    }
    for (int at = firstRon; at < end; at++) settle(at, events.get(at));
    close(firstRon < end);
    return end;
  }

  /** 牌譜が窓の中で終わった。判断の残る席は出来事の数の位置で未観測にする。 */
  void endOfRecord(int eventCount) {
    resolveUnanswered(eventCount, false);
    close(false);
  }

  /**
   * 牌譜の手を示さなかった判断の席を、席の順に見送りか未観測で解く。ロンできた席はその判断の直後に一時振聴にするので、後の席の判断は新しい改訂を見る。
   *
   * @param othersPassed 明示しなかった席も見送りと確かめられる（並びに鳴き・ロンが無く、牌譜が続く）
   */
  private void resolveUnanswered(int resolveIndex, boolean othersPassed) {
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      if (!decides[seat] || replies[seat] == Reply.ACTED) continue;
      List<Action> offered = legal.responses(seat);
      if (replies[seat] == Reply.PASSED || othersPassed)
        feed.passed(seat, kind.decisionKind(), offered, cause, resolveIndex);
      else feed.unobserved(seat, kind.decisionKind(), offered, cause, resolveIndex);
      declineRon(seat);
    }
  }

  /**
   * 並びの 1 つの応答を解く。鳴きはここで適用して通知し、ロンは判断だけを出す（精算は {@link #settle}）。
   *
   * @param afterRon 並びのこれより前にロンがある（明示の見送りは {@link #settle} が通知する）
   */
  private void answer(int index, ReplayEvent event, boolean afterRon) {
    switch (event) {
      case None pass -> {
        if (!afterRon) feed.applied(index, pass);
      }
      case Hora ron -> {
        int seat = ron.actor();
        feed.recorded(
            seat, kind.decisionKind(), Action.ronAgari(), legal.responses(seat), cause, index);
      }
      case Chi chi -> {
        Action action = RecordedActions.chi(chi);
        call(index, chi.actor(), action);
        table.chi(index, chi, action);
        feed.applied(index, chi);
      }
      case Pon pon -> {
        Action action = RecordedActions.pon(pon);
        call(index, pon.actor(), action);
        table.pon(index, pon, action);
        feed.applied(index, pon);
      }
      case Daiminkan kan -> {
        call(index, kan.actor(), RecordedActions.daiminkan(kan));
        table.daiminkan(index, kan);
        feed.applied(index, kan);
      }
      default -> throw new IllegalStateException("Unexpected response event: " + event);
    }
  }

  /** 並びの最初のロンから後の 1 つの出来事を通知する。ロンは精算して、明示の見送りはそのまま（鳴きは適用した時に通知した）。 */
  private void settle(int index, ReplayEvent event) {
    switch (event) {
      case Hora win -> feed.applied(index, table.settleWin(win));
      case None pass -> feed.applied(index, pass);
      default -> {}
    }
  }

  /** 鳴きの判断。ロンできた席は鳴く前に一時振聴になる。 */
  private void call(int index, int seat, Action action) {
    feed.recorded(seat, kind.decisionKind(), action, legal.responses(seat), cause, index);
    declineRon(seat);
  }

  private void declineRon(int seat) {
    if (ronOffered[seat] && table.enterFuriten(seat)) feed.changed();
  }

  private void close(boolean robbed) {
    if (kind == WindowKind.KAKAN) {
      if (robbed) table.robKakan();
      else table.completeKakan(source);
      feed.changed();
    }
    Arrays.fill(decides, false);
    Arrays.fill(ronOffered, false);
    Arrays.fill(replies, Reply.UNSEEN);
    kind = null;
  }

  private boolean anyActed() {
    for (Reply reply : replies) if (reply == Reply.ACTED) return true;
    return false;
  }
}
