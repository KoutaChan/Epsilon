package com.epsilon.mahjong.record;

import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 判断の時点の記録の位置（{@code ReplayDriver.cursor}）のテスト。自己対局の記録（四人打ちと三人打ち）の全ての判断で、位置の直前の
 * イベントは判断を起こしたイベントで、位置のイベントは判断で選ぶ手から生まれるイベントになる。位置より前にめくった表示牌の数は
 * エンジンと同じになる。局末では、位置はイベントの数になる。
 */
public class ReplayCursorTest {

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test(dataProvider = "variants")
  public void theCursorSplitsTheRecordAtEveryDecision(Variant variant) {
    List<RecordedGame> games = new ArrayList<>();
    for (int seed = 0; seed < 20; seed++) {
      games.add(RecordedGames.play(variant, seed, "self-" + seed));
    }
    ReplayDriver driver = new ReplayDriver(variant);
    RuleWorkspace ws = new RuleWorkspace(variant);
    int decisions = 0;
    for (RecordedGame game : games) {
      for (RecordedRound round : game.rounds()) {
        List<RecordEvent> events = round.events();
        int previous = 0;
        Boundary boundary = driver.start(round, ws);
        while (boundary == Boundary.DECISIONS) {
          int cursor = driver.cursor();
          String where = game.meta().id() + " " + round.start() + " at " + cursor;
          Assert.assertTrue(cursor > previous, where);
          Assert.assertTrue(cursor <= events.size(), where);
          Decisions decided = driver.engine().decisions();
          int seat = decided.seat(0);
          SeatView view = driver.engine().view(seat);
          int actor = view.last().seat();
          RecordEvent before = events.get(cursor - 1);
          RecordEvent after = cursor < events.size() ? events.get(cursor) : null;
          boolean fits =
              switch (decided.kind(0)) {
                case TURN -> before instanceof RecordEvent.Draw draw
                    && draw.seat() == seat
                    && (after == null || turnAction(after, seat));
                case POST_CALL_DAHAI -> before instanceof RecordEvent.Call call
                    && call.seat() == seat
                    && after instanceof RecordEvent.Discard discard
                    && discard.seat() == seat;
                case RESPONSE -> lastMove(events, cursor) instanceof RecordEvent.Discard discard
                    && discard.seat() == actor
                    && (after == null
                        || after instanceof RecordEvent.RiichiAccepted
                        || after instanceof RecordEvent.Call
                        || after instanceof RecordEvent.Draw);
                case CHANKAN -> before instanceof RecordEvent.AddedKan kan
                    && kan.seat() == actor
                    && (after == null || after instanceof RecordEvent.DoraReveal || after instanceof RecordEvent.Draw);
                case NUKI_RESPONSE -> before instanceof RecordEvent.Nuki nuki
                    && nuki.seat() == actor
                    && (after == null || after instanceof RecordEvent.DoraReveal || after instanceof RecordEvent.Draw);
              };
          Assert.assertTrue(fits, where + " " + decided.kind(0) + " between " + before + " and " + after);
          Assert.assertEquals(reveals(events, cursor), view.indicatorCount() - 1, where);
          previous = cursor;
          decisions++;
          boundary = driver.step(ws);
        }
        Assert.assertEquals(driver.cursor(), events.size(), game.meta().id() + " " + round.start());
      }
    }
    Assert.assertTrue(decisions > 5_000, decisions + " decisions");
  }

  /** 局を始める前は位置が無い。 */
  @Test
  public void thereIsNoCursorBeforeARound() {
    ReplayDriver driver = new ReplayDriver(Variant.TENHOU_4P);
    Assert.assertThrows(IllegalStateException.class, driver::cursor);
  }

  /**
   * イベントが、手番の席の手（打牌・暗槓・加槓・北抜き）かを返す。
   *
   * @param seat 手番の席
   */
  private static boolean turnAction(RecordEvent event, int seat) {
    return switch (event) {
      case RecordEvent.Discard discard -> discard.seat() == seat;
      case RecordEvent.ClosedKan kan -> kan.seat() == seat;
      case RecordEvent.AddedKan kan -> kan.seat() == seat;
      case RecordEvent.Nuki nuki -> nuki.seat() == seat;
      default -> false;
    };
  }

  /**
   * 位置より前にある、表示牌でない最後のイベントを返す。
   *
   * @param end 位置（この位置のイベントは含めない）
   * @return そのイベント。無ければ null
   */
  private static RecordEvent lastMove(List<RecordEvent> events, int end) {
    for (int index = end - 1; index >= 0; index--) {
      if (!(events.get(index) instanceof RecordEvent.DoraReveal)) {
        return events.get(index);
      }
    }
    return null;
  }

  /**
   * 位置より前にある表示牌の数を返す。
   *
   * @param end 位置（この位置のイベントは含めない）
   */
  private static int reveals(List<RecordEvent> events, int end) {
    int count = 0;
    for (int index = 0; index < end; index++) {
      count += events.get(index) instanceof RecordEvent.DoraReveal ? 1 : 0;
    }
    return count;
  }
}
