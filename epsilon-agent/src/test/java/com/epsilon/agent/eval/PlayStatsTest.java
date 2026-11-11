package com.epsilon.agent.eval;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.GameObserver;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.LongAdder;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 対局の集計のテスト。候補の局の数・和了率・放銃率・立直した局の割合と、スロットごとの点の増減、鳴ける場面・鳴いた回数、立直できる場面・
 * 立直した回数は、観測のフックで数え直した値と一致する。
 */
public class PlayStatsTest {

  /** スロットごとに数え直す数と、{@link PlayStats#appendTo} の列でスロットの中の何番目に書かれるか。 */
  enum Column {
    NET_POINTS(11),
    CALL_CHANCES(15),
    CALLS_TAKEN(16),
    RIICHI_CHANCES(17),
    RIICHI_TAKEN(18);

    static final int COUNT = values().length;

    final int index;

    Column(int index) {
      this.index = index;
    }
  }

  /** 候補の席（牌山の番号を 4 で割った余り）について局の結果を、スロットごと（候補は 0、相手は 1）に点の増減と判断を数え直す。 */
  static final class Recount implements GameObserver {
    final LongAdder hands = new LongAdder(), wins = new LongAdder(), dealIns = new LongAdder(), riichi = new LongAdder();

    /** 添字は [スロット × 数の種類の数 ＋ 数の種類]。 */
    final AtomicLongArray columns = new AtomicLongArray(2 * Column.COUNT);

    long count(int slot, Column column) {
      return columns.get(slot * Column.COUNT + column.ordinal());
    }

    private void add(GameKey game, int seat, Column column, long amount) {
      int slot = seat == Math.floorMod(game.wall(), 4) ? 0 : 1;
      columns.addAndGet(slot * Column.COUNT + column.ordinal(), amount);
    }

    @Override
    public void gameEnded(GameKey game, TableState last) {}

    @Override
    public void decided(GameKey game, SeatView view, Decisions decisions, int slot, int choice) {
      if (decisions.legalCount(slot) < 2) return;
      int seat = decisions.seat(slot);
      boolean canCall = false, canRiichi = false;
      for (int a = 0; a < decisions.legalCount(slot); a++) {
        Action.Type type = Action.of(decisions.legal(slot, a)).type();
        canCall |= type == Action.Type.CHI || type == Action.Type.PON;
        canRiichi |= type == Action.Type.RIICHI_DAHAI;
      }
      Action.Type chosen = Action.of(decisions.legal(slot, choice)).type();
      if (canCall) {
        add(game, seat, Column.CALL_CHANCES, 1);
        if (chosen == Action.Type.CHI || chosen == Action.Type.PON) add(game, seat, Column.CALLS_TAKEN, 1);
      }
      if (canRiichi) {
        add(game, seat, Column.RIICHI_CHANCES, 1);
        if (chosen == Action.Type.RIICHI_DAHAI) add(game, seat, Column.RIICHI_TAKEN, 1);
      }
    }

    @Override
    public void roundEnded(GameKey game, FullView full, RoundEnd end) {
      int seat = Math.floorMod(game.wall(), 4);
      hands.increment();
      for (int other = 0; other < 4; other++) add(game, other, Column.NET_POINTS, end.delta(other));
      if (full.seat(seat).riichi(seat) != RiichiState.NONE) riichi.increment();
      RoundOutcome outcome = end.outcome();
      if (outcome.kind() == EndKind.TSUMO || outcome.kind() == EndKind.RON)
        for (int i = 0; i < outcome.winnerCount(); i++) if (outcome.winner(i) == seat) wins.increment();
      if (outcome.kind() == EndKind.RON && outcome.loser() == seat) dealIns.increment();
    }
  }

  /** 見送りを選べるときは必ず見送り、それ以外はハッシュで手を選ぶ打ち手。鳴かないので門前のまま進み、立直の機会が出る。 */
  record Closed(ArenaTest.Hashed hashed) implements InlinePolicy {

    @Override
    public Variant variant() {
      return TENHOU_4P;
    }

    @Override
    public String label() {
      return "closed";
    }

    @Override
    public int act(Observation observation) {
      for (int a = 0; a < observation.legalActions(); a++)
        if (observation.action(a).type() == Action.Type.PASS) return a;
      return hashed.act(observation);
    }

    @Override
    public Evaluation evaluate(Observation observation) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }

  @Test(timeOut = 300_000)
  public void candidateRatesMatchARecount() {
    var recount = new Recount();
    var stats = new PlayStats(2);
    try (var table = new TableLoop(new TableSettings(4, 2, 1000L, 4))) {
      var plan = ArenaTest.byGame(new Closed(new ArenaTest.Hashed(3)), new ArenaTest.Hashed(4), 16, 77, 0);
      Arena.play(
          table,
          plan,
          (wall, scores, games) -> {
            for (PlayStats game : games) {
              stats.add(game);
            }
            return true;
          },
          recount);
    }
    long hands = recount.hands.sum();
    Assert.assertTrue(hands >= 16, "hands " + hands);
    Assert.assertEquals(stats.seatHands(0), hands);
    Assert.assertEquals(stats.seatHands(1), 3 * hands);
    PlayStats.Rates candidate = stats.rates(0);
    Assert.assertEquals(candidate.win(), recount.wins.sum() / (double) hands, 1e-12);
    Assert.assertEquals(candidate.dealIn(), recount.dealIns.sum() / (double) hands, 1e-12);
    Assert.assertEquals(candidate.riichi(), recount.riichi.sum() / (double) hands, 1e-12);
    Assert.assertTrue(recount.riichi.sum() > 0, "no riichi hand");
    StringBuilder line = new StringBuilder();
    stats.appendTo(line);
    String[] written = line.substring(1).split("\t");
    int perSlot = stats.columns() / stats.slots();
    for (int slot = 0; slot < 2; slot++)
      for (Column column : Column.values())
        Assert.assertEquals(
            Long.parseLong(written[slot * perSlot + column.index]),
            recount.count(slot, column),
            "slot " + slot + " " + column);
    Assert.assertTrue(
        recount.count(0, Column.CALL_CHANCES) > 0
            && recount.count(0, Column.RIICHI_CHANCES) > 0
            && recount.count(1, Column.CALLS_TAKEN) > 0,
        "no call or riichi chances");
  }
}
