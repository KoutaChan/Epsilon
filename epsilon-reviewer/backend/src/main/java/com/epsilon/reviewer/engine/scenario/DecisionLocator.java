package com.epsilon.reviewer.engine.scenario;

import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.view.RecordTable;
import com.epsilon.reviewer.engine.view.TableState;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 解析した 1 つの判断を探す。牌譜のその局だけを牌譜の手のとおりにエンジンで進め、目当ての判断とその時点の局面を取り出す。局の
 * 記録は局の始まりの状態を持つので、その局だけを再生しても、対局の始めから再生したときと同じ状態になる。
 */
public final class DecisionLocator {
  private DecisionLocator() {}

  /**
   * その席が局の中で何番目にした判断かで探す。見つけた判断の合法手の集合が、文書の候補と同じかも確かめる。
   *
   * @param ordinal その席の判断を局の始めから数えて何番目か（0 から）
   * @param candidates 文書にある候補の行動の番号（順は問わない）。確かめなければ null
   * @throws ReviewerException 判断が見つからないか、合法手が候補と違うとき（decision_not_found）
   */
  public static LocatedDecision findNth(
      LoadedRecord record, int round, int seat, int ordinal, int[] candidates) {
    if (round < 0 || round >= record.rounds().size() || ordinal < 0) throw notFound();
    return search(record, round, seat, ordinal, candidates);
  }

  private static ReviewerException notFound() {
    return new ReviewerException(ErrorCode.DECISION_NOT_FOUND);
  }

  /**
   * 局を目当ての判断まで進めて、その判断と局面を取り出す。
   *
   * @param ordinal その席の判断を局の始めから数えて何番目か（0 から）
   * @param candidates 確かめる候補。確かめなければ null
   */
  private static LocatedDecision search(
      LoadedRecord record, int round, int seat, int ordinal, int[] candidates) {
    Variant variant = record.game().variant();
    if (seat < 0 || seat >= variant.seats()) throw notFound();
    RecordedRound recorded = record.rounds().get(round);
    List<RecordEvent> events = recorded.events();
    var driver = new ReplayDriver(variant);
    var ws = new RuleWorkspace(variant);
    var table = new RecordTable(recorded);
    int applied = 0;
    int seen = 0;
    Boundary boundary = driver.start(recorded, ws);
    while (boundary == Boundary.DECISIONS) {
      int cursor = driver.cursor();
      for (; applied < cursor; applied++) table.apply(events.get(applied));
      int slot = driver.engine().decisions().slotOf(seat);
      if (slot >= 0 && seen++ == ordinal)
        return located(driver, round, seat, slot, cursor, table.state(), recorded, candidates);
      boundary = driver.step(ws);
    }
    throw notFound();
  }

  private static LocatedDecision located(
      ReplayDriver driver,
      int round,
      int seat,
      int slot,
      int position,
      TableState table,
      RecordedRound recorded,
      int[] candidates) {
    Engine engine = driver.engine();
    Decisions decisions = engine.decisions();
    int[] legal = new int[decisions.legalCount(slot)];
    for (int index = 0; index < legal.length; index++) legal[index] = decisions.legal(slot, index);
    if (candidates != null && !sameSet(legal, candidates)) throw notFound();
    int choice = driver.choice(slot);
    int recordedCode = choice == ReplayDriver.UNOBSERVED ? -1 : legal[choice];
    int drawn = table.players().get(seat).draw() == null ? -1 : lastDraw(recorded, position, seat);
    return new LocatedDecision(
        round,
        position,
        seat,
        LocatedDecision.Kind.of(decisions.kind(slot)),
        table,
        recordedCode,
        legal,
        concealed(engine.view(seat), recorded, position, seat, drawn),
        drawn);
  }

  /** 判断する席の伏せた手牌の物理牌を、手に入った順（配牌、続いてツモ）に並べる。表示するツモ牌は除く。 */
  private static List<Integer> concealed(
      SeatView view, RecordedRound recorded, int position, int seat, int drawn) {
    List<Integer> hand = new ArrayList<>(Variant.HAND + 1);
    for (int index = 0; index < Variant.HAND; index++) {
      int tile = recorded.deal(seat, index);
      if (tile != drawn && view.holds(tile)) hand.add(tile);
    }
    for (int at = 0; at < position; at++)
      if (recorded.events().get(at) instanceof RecordEvent.Draw draw
          && draw.seat() == seat
          && draw.tile() != drawn
          && view.holds(draw.tile())) hand.add(draw.tile());
    return List.copyOf(hand);
  }

  /** 判断する席が、判断の時点より前に最後にツモった物理牌。 */
  private static int lastDraw(RecordedRound recorded, int position, int seat) {
    for (int at = position - 1; at >= 0; at--)
      if (recorded.events().get(at) instanceof RecordEvent.Draw draw && draw.seat() == seat)
        return draw.tile();
    throw new IllegalStateException("Seat " + seat + " shows a drawn tile without a draw");
  }

  private static boolean sameSet(int[] legal, int[] candidates) {
    int[] a = legal.clone();
    int[] b = candidates.clone();
    Arrays.sort(a);
    Arrays.sort(b);
    return Arrays.equals(a, b);
  }
}
